"""Main training loop: self-play -> replay buffer -> gradient step -> TensorBoard -> checkpoint.

Run: python3 -m training.train [--config training/configs/default.yaml] [--num-iterations N] ...
Any CLI flag overrides the same key from --config, so you can e.g. take a saved config and just
bump --num-iterations for a longer run without editing the file.
"""
import argparse
import inspect
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from railroad_env import RailroadGymEnv

from .mcts import MCTS
from .model import TrainUNet
from .replay_buffer import ReplayBuffer
from .self_play import play_episode, play_episode_no_search, play_episode_topk, self_play_worker


def masked_policy_loss(policy_logits: torch.Tensor, policy_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between the model's policy distribution and the MCTS visit-count target
    distribution, both flattened over (2, H, W). `policy_logits` must already be illegal-action
    masked - i.e. produced by calling the model WITH an action_mask, not masked again here."""
    log_probs = F.log_softmax(policy_logits.flatten(1), dim=1)
    targets_flat = policy_targets.flatten(1)
    return -(targets_flat * log_probs).sum(dim=1).mean()


def train(
    num_iterations: int = 1000,
    games_per_iteration: int = 1,
    train_steps_per_iteration: int = 20,
    batch_size: int = 128,
    buffer_capacity: int = 100_000,
    use_mcts: bool = True,
    topk: int = 0,
    num_simulations: int = 100,
    eval_batch_size: int = 32,
    c_puct: float = 1.5,
    temperature_moves: int = 30,
    lr: float = 1e-3,
    checkpoint_every: int = 20,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = None,
    device: str = None,
    grid_height: int = None,
    max_turns: int = 100,
    opponent_strategy: str = "passive",
    seed: int = None,
    num_workers: int = 1,
):
    # Every parameter this run actually used (defaults included) - captured before any other
    # local variable exists, so locals() is exactly the resolved config. Saved to log_dir below
    # so every run directory is self-documenting: `training/configs/<name>.yaml` files are
    # starting points, this is the record of what a specific run was actually launched with.
    effective_config = dict(locals())

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir or f"runs/{run_name}"
    effective_config["log_dir"] = log_dir
    effective_config["device"] = device

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(log_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(effective_config, f, sort_keys=False)

    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # topk > 0 switches to the simplified single-forward-pass-per-turn strategy
    # (play_episode_topk): DISRUPT is dropped entirely, so the policy head only needs 1
    # channel (PLACE) instead of 2 - overrides use_mcts, which is meaningless without a
    # DISRUPT-aware 2-channel policy to search over.
    model = TrainUNet(policy_channels=1 if topk > 0 else 2).to(device)
    # torch.compile(model) was tried and is NOT enabled here: it failed outright in this sandbox
    # (PermissionError invoking `nvcc` from inside inductor's compilation path), and separately,
    # MCTS calls the model with varying batch sizes (remainder batches when num_simulations isn't
    # a multiple of eval_batch_size) which would trigger repeated recompilation even with
    # dynamic=True. Worth re-testing in an unrestricted environment, but not a safe default here.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Mixed precision only helps (and is only supported) on CUDA - a no-op autocast/scaler on CPU.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    buffer = ReplayBuffer(capacity=buffer_capacity)
    rng = np.random.RandomState(seed)

    # --- parallel self-play setup ---
    # num_workers > 1 plays games_per_iteration games across a pool of worker PROCESSES
    # (real OS-level parallelism, not threads - Python's GIL would serialize the game logic
    # otherwise) instead of one after another in this process. The pool is created once, up
    # front, and reused for the whole run - spawning fresh processes every iteration would cost
    # far more than the games themselves at this model size. "spawn" (not the default "fork" on
    # Linux) is required for safety: workers must start a clean Python interpreter rather than
    # forking this process's already-initialized CUDA context, which is unsafe to share.
    # Workers always run on CPU regardless of `device` - see self_play_worker's docstring.
    strategy = "topk" if topk > 0 else ("mcts" if use_mcts else "no_search")
    strategy_kwargs = {
        "topk": topk,
        "c_puct": c_puct,
        "num_simulations": num_simulations,
        "eval_batch_size": eval_batch_size,
        "temperature_moves": temperature_moves,
    }
    env_kwargs = dict(
        grid_height=grid_height,
        max_turns=max_turns,
        opponent_strategy=opponent_strategy,
    )
    policy_channels = 1 if topk > 0 else 2

    pool = mp.get_context("spawn").Pool(processes=num_workers) if num_workers > 1 else None

    try:
        for iteration in range(num_iterations):
            iter_start = time.time()

            # Seeds are always drawn sequentially from the single `rng` here in the main
            # process (never inside a worker), so a given top-level `seed` reproduces the same
            # per-game seeds regardless of num_workers.
            game_seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(games_per_iteration)]

            if pool is not None:
                state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                game_results = pool.starmap(
                    self_play_worker,
                    [(state_dict_cpu, policy_channels, strategy, strategy_kwargs, env_kwargs, s) for s in game_seeds],
                )
            else:
                # use_mcts=False skips search entirely (num_simulations/eval_batch_size/c_puct
                # are unused in that case): one forward pass per atomic decision instead of
                # num_simulations-many, at the cost of a much weaker policy training signal -
                # see play_episode_no_search's docstring. Useful for fast pipeline iteration.
                mcts = MCTS(model, device=device, c_puct=c_puct, num_simulations=num_simulations, eval_batch_size=eval_batch_size) if strategy == "mcts" else None
                game_results = []
                for s in game_seeds:
                    env = RailroadGymEnv(seed=s, **env_kwargs)
                    env.reset()
                    if strategy == "topk":
                        game_results.append(play_episode_topk(env, model, device=device, k=topk))
                    elif strategy == "mcts":
                        game_results.append(
                            play_episode(env, mcts, temperature=1.0, temperature_moves=temperature_moves, rng=rng)
                        )
                    else:
                        game_results.append(
                            play_episode_no_search(
                                env, model, device=device, temperature=1.0, temperature_moves=temperature_moves, rng=rng
                            )
                        )

            score_margins, episode_lengths = [], []
            player0_scores, player1_scores = [], []
            for examples, final_scores in game_results:
                buffer.push_many(examples)
                score_margins.append(final_scores[0] - final_scores[1])
                episode_lengths.append(len(examples))
                player0_scores.append(final_scores[0])
                player1_scores.append(final_scores[1])

            self_play_time = time.time() - iter_start

            writer.add_scalar("self_play/score_margin", np.mean(score_margins), iteration)
            writer.add_scalar("self_play/player0_score", np.mean(player0_scores), iteration)
            writer.add_scalar("self_play/player1_score", np.mean(player1_scores), iteration)
            writer.add_scalar("self_play/episode_length", np.mean(episode_lengths), iteration)
            writer.add_scalar("self_play/buffer_size", len(buffer), iteration)
            writer.add_scalar("self_play/seconds", self_play_time, iteration)

            if len(buffer) < batch_size:
                print(f"[iter {iteration}] buffer too small ({len(buffer)}/{batch_size}), skipping training step")
                continue

            # --- training: gradient steps on sampled minibatches ---
            model.train()
            train_start = time.time()
            total_losses, policy_losses, value_losses = [], [], []

            for _ in range(train_steps_per_iteration):
                states, policies, values, masks = buffer.sample(batch_size)
                states_t = torch.from_numpy(states).to(device)
                policies_t = torch.from_numpy(policies).to(device)
                values_t = torch.from_numpy(values).to(device).unsqueeze(1)
                masks_t = torch.from_numpy(masks).to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    policy_logits, value_pred = model(states_t, masks_t)
                    p_loss = masked_policy_loss(policy_logits, policies_t)
                    v_loss = F.mse_loss(value_pred, values_t)
                    loss = p_loss + v_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_losses.append(loss.item())
                policy_losses.append(p_loss.item())
                value_losses.append(v_loss.item())

            train_time = time.time() - train_start

            writer.add_scalar("train/total_loss", np.mean(total_losses), iteration)
            writer.add_scalar("train/policy_loss", np.mean(policy_losses), iteration)
            writer.add_scalar("train/value_loss", np.mean(value_losses), iteration)
            writer.add_scalar("train/seconds", train_time, iteration)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], iteration)

            print(
                f"[iter {iteration}] score_margin={np.mean(score_margins):.1f} "
                f"loss={np.mean(total_losses):.4f} (policy={np.mean(policy_losses):.4f}, "
                f"value={np.mean(value_losses):.4f}) "
                f"self_play={self_play_time:.1f}s train={train_time:.1f}s buffer={len(buffer)}"
            )

            if (iteration + 1) % checkpoint_every == 0:
                ckpt_file = checkpoint_path / f"{run_name}_iter{iteration + 1}.pt"
                torch.save(
                    {
                        "iteration": iteration + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_file,
                )
                print(f"[iter {iteration}] checkpoint saved: {ckpt_file}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    writer.close()


def _build_parser():
    parser = argparse.ArgumentParser(description="Train the Railroad Tycoon AlphaZero-style agent")
    parser.add_argument("--config", type=str, default=None, help="YAML file of defaults (see training/configs/); any other flag overrides its value")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--games-per-iteration", type=int, default=1)
    parser.add_argument("--train-steps-per-iteration", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--use-mcts", action=argparse.BooleanOptionalAction, default=True, help="--no-use-mcts skips search entirely - one network forward pass per decision instead of num_simulations-many (faster, weaker training signal)")
    parser.add_argument("--topk", type=int, default=0, help="If >0, ignore DISRUPT and MCTS entirely: one forward pass per turn, place the top-k legal cells by score directly (overrides --use-mcts)")
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature-moves", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grid-height", type=int, default=None, help="Pin the map height (width follows the 1.5 aspect ratio). Omit for the authentic random 14-20.")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--opponent-strategy", type=str, default="passive", choices=["passive", "boss", "random", "greedy"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1, help="Play games_per_iteration games across this many parallel worker processes instead of one after another (1 = no multiprocessing)")
    return parser


def _parse_args():
    parser = _build_parser()

    # Two-pass parse: find --config first, load it, and use its values as new argparse
    # defaults - so any flag actually typed on the CLI still overrides the config file's value.
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            overrides = yaml.safe_load(f) or {}

        valid_keys = set(inspect.signature(train).parameters)
        unknown = set(overrides) - valid_keys
        if unknown:
            raise ValueError(f"Unknown key(s) in {pre_args.config}: {sorted(unknown)} - check for typos against train()'s parameters")

        parser.set_defaults(**overrides)

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.pop("config")
    return args_dict


if __name__ == "__main__":
    train(**_parse_args())

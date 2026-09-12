"""PPO training loop for the Railroad Tycoon env.

RailroadUNet is the shared actor-critic - a 2-down/2-up U-Net with a convolution before and
after, a spatial policy head (PLACE / DISRUPT / SKIP_DISRUPT channels) and a critic reading the
bottleneck. See training/ppo.py for the rollout/update mechanics and
training/configs/ppo_full.yaml for the intended env/PPO settings together.

Run: python3 -m training.train_ppo [--config training/configs/ppo.yaml] [--num-iterations N] ...
Any CLI flag overrides the same key from --config, matching training/train.py's convention.
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
import yaml
from torch.utils.tensorboard import SummaryWriter

from railroad_env.game_state import GameState
from railroad_env.opponent import BOSS_TIERS, OPPONENT_STRATEGIES

from .agent_model import RailroadUNet
from .evaluate import evaluate_all_bosses, format_eval_table, log_eval_to_tensorboard
from .ppo import DEFAULT_SCORE_NORM, SELF_PLAY, collect_rollout, ppo_update


def train_ppo(
    num_iterations: int = 1000,
    games_per_iteration: int = 8,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.05,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 4,
    minibatch_size: int = 128,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    lr: float = 3e-4,
    lr_end: float = None,
    base_channels: int = 64,
    channels=None,
    value_hidden: int = 256,
    score_norm: float = DEFAULT_SCORE_NORM,
    allow_skip_disrupt: bool = True,
    reward_mode: str = "own",
    win_bonus: float = 0.0,       # terminal +/- bonus in raw score units; see ppo.py::_play_one_game
    force_disrupt: bool = False,
    action_selection: str = "epsilon_greedy",
    opponent_pool: dict = None,
    self_play_epsilon: float = 0.0,
    eval_episodes: int = 100,
    eval_max_turns: int = None,
    eval_bosses: str = ",".join(BOSS_TIERS),
    checkpoint_every: int = 20,
    checkpoint_dir: str = "checkpoints",
    init_checkpoint: str = None,
    log_dir: str = None,
    device: str = None,
    grid_height: int = None,
    max_turns: int = 100,
    opponent_strategy: str = "passive",
    opponent_wait_probability: float = 0.0,
    opponent_disrupt_probability: float = 0.5,
    seed: int = None,
    num_workers: int = 1,
):
    # See training/train.py's train() for why this pattern (locals() captured before any other
    # local variable exists) makes every run directory self-documenting.
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

    # 3 policy channels when disrupting is optional (PLACE / DISRUPT / SKIP_DISRUPT), 2 when it
    # is compulsory.
    model_kwargs = dict(
        in_channels=GameState.NUM_CHANNELS,
        policy_channels=3 if allow_skip_disrupt else 2,
        base_channels=base_channels,
        value_hidden=value_hidden,
    )
    # An explicit (c1, c2, c3) overrides the doubling progression base_channels implies. This is
    # what a fine-tune of a distilled student needs: the student is a deliberately flat, small
    # shape, and without this the warm start would build a (64,128,256) net and then throw a
    # shape mismatch on load_state_dict. Checkpoints record whichever shape was used, so
    # bake_agent.py and the evaluators rebuild it correctly with no further flags.
    if channels is not None:
        model_kwargs["channels"] = tuple(int(c) for c in channels)
    model = RailroadUNet(**model_kwargs).to(device)
    if init_checkpoint is not None:
        # Weights only, not optimizer state - this is a warm start (fresh Adam moments), not a
        # resume. Iteration numbers in this run's logs/checkpoints start back at 0 regardless
        # of how far init_checkpoint's own run got - they're this run's own count, not a
        # continuation of the source run's numbering.
        init_ckpt = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"Initialized weights from {init_checkpoint} (trained for {init_ckpt.get('iteration', '?')} iterations)")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"RailroadUNet: {sum(p.numel() for p in model.parameters()):,} parameters")

    # grid_height=None keeps the authentic random map size (height 14-20, width from the 1.5
    # aspect ratio); the encoding pads whatever comes out into the fixed network canvas.
    env_kwargs = dict(
        grid_height=grid_height,
        max_turns=max_turns,
        opponent_strategy=opponent_strategy,
        opponent_kwargs={
            "wait_probability": opponent_wait_probability,
            "disrupt_probability": opponent_disrupt_probability,
        },
    )

    if action_selection not in ("epsilon_greedy", "onpolicy"):
        raise ValueError(
            f"action_selection must be 'epsilon_greedy' or 'onpolicy', got {action_selection!r}"
        )
    if opponent_pool:
        unknown = [n for n in opponent_pool if n != SELF_PLAY and n not in OPPONENT_STRATEGIES]
        if unknown:
            raise ValueError(
                f"opponent_pool has unknown opponents {unknown}. "
                f"Available: {sorted(OPPONENT_STRATEGIES) + [SELF_PLAY]}"
            )
        print(f"Opponent pool: {opponent_pool}"
              + (f"  (self-play epsilon {self_play_epsilon})" if SELF_PLAY in opponent_pool else ""))
    if force_disrupt and not allow_skip_disrupt:
        raise ValueError(
            "force_disrupt with allow_skip_disrupt=False is redundant: the head is already 2 "
            "channels and SKIP_DISRUPT does not exist. Use allow_skip_disrupt=True to keep the "
            "3-channel head loadable from an existing checkpoint."
        )

    eval_boss_list = [b.strip() for b in eval_bosses.split(",") if b.strip()]

    rng = np.random.RandomState(seed)

    # num_workers > 1 plays games_per_iteration games across a pool of worker PROCESSES
    # instead of one after another in this process - see training/train.py's identical
    # pattern/reasoning (real OS-level parallelism, a persistent "spawn" pool created once
    # up front, workers always on CPU).
    pool = mp.get_context("spawn").Pool(processes=num_workers) if num_workers > 1 else None

    try:
        for iteration in range(num_iterations):
            iter_start = time.time()

            # Linear decay from epsilon_start (iteration 0) to epsilon_end (the final iteration).
            decay_progress = iteration / max(1, num_iterations - 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * decay_progress

            # Linearly anneal the learning rate alongside epsilon. Late in a run the policy
            # should be refining, not still taking full-size steps - leaving lr flat is what let
            # the previous run drift back down after peaking.
            if lr_end is not None:
                current_lr = lr + (lr_end - lr) * decay_progress
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            else:
                current_lr = lr

            transitions, final_scores_list, episode_lengths = collect_rollout(
                env_kwargs, model, model_kwargs, device, gamma, gae_lambda,
                games_per_iteration, rng, epsilon=epsilon, pool=pool,
                allow_skip_disrupt=allow_skip_disrupt, score_norm=score_norm,
                reward_mode=reward_mode, force_disrupt=force_disrupt,
                action_selection=action_selection, opponent_pool=opponent_pool,
                self_play_epsilon=self_play_epsilon, win_bonus=win_bonus,
            )
            rollout_time = time.time() - iter_start

            score_margins = [s[0] - s[1] for s in final_scores_list]
            player0_scores = [s[0] for s in final_scores_list]
            player1_scores = [s[1] for s in final_scores_list]
            wins = float(np.mean([1.0 if s[0] > s[1] else 0.0 for s in final_scores_list]))

            writer.add_scalar("self_play/score_margin", np.mean(score_margins), iteration)
            writer.add_scalar("self_play/player0_score", np.mean(player0_scores), iteration)
            writer.add_scalar("self_play/player1_score", np.mean(player1_scores), iteration)
            writer.add_scalar("self_play/win_rate", wins, iteration)
            writer.add_scalar("self_play/episode_length", np.mean(episode_lengths), iteration)
            # Only meaningful under epsilon_greedy. Logging the annealing schedule under
            # onpolicy, where nothing reads it, draws a confident curve for a dead knob.
            if action_selection == "epsilon_greedy":
                writer.add_scalar("self_play/epsilon", epsilon, iteration)
            writer.add_scalar("train/learning_rate", current_lr, iteration)
            writer.add_scalar("self_play/seconds", rollout_time, iteration)

            train_start = time.time()
            stats = ppo_update(
                model, optimizer, transitions, ppo_epochs, minibatch_size,
                clip_epsilon, value_coef, entropy_coef, max_grad_norm, device,
            )
            train_time = time.time() - train_start

            writer.add_scalar("train/policy_loss", stats["policy_loss"], iteration)
            writer.add_scalar("train/value_loss", stats["value_loss"], iteration)
            writer.add_scalar("train/entropy", stats["entropy"], iteration)
            writer.add_scalar("train/clip_frac", stats["clip_frac"], iteration)
            writer.add_scalar("train/approx_kl", stats["approx_kl"], iteration)
            writer.add_scalar("train/explained_variance", stats["explained_variance"], iteration)
            writer.add_scalar("train/seconds", train_time, iteration)

            print(
                f"[iter {iteration}] score={np.mean(player0_scores):.0f} "
                f"margin={np.mean(score_margins):+.0f} win_rate={wins:.2f} "
                + (f"eps={epsilon:.3f} " if action_selection == "epsilon_greedy" else "")
                + f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} clip_frac={stats['clip_frac']:.3f} "
                f"ev={stats['explained_variance']:+.2f} "
                f"rollout={rollout_time:.1f}s train={train_time:.1f}s transitions={len(transitions)}"
            )

            if (iteration + 1) % checkpoint_every == 0:
                ckpt_file = checkpoint_path / f"{run_name}_iter{iteration + 1}.pt"
                torch.save(
                    {
                        "iteration": iteration + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_kwargs": model_kwargs,
                        # Recorded so bake_agent.py and evaluate.py can reproduce the action
                        # space this checkpoint was actually trained on. With force_disrupt the
                        # head keeps its SKIP_DISRUPT channel but that channel never receives a
                        # gradient, so offering the action downstream would run on frozen
                        # weights - see play_eval_game's docstring.
                        "allow_skip_disrupt": allow_skip_disrupt,
                        "force_disrupt": force_disrupt,
                    },
                    ckpt_file,
                )
                print(f"[iter {iteration}] checkpoint saved: {ckpt_file}")

                # Benchmark the freshly-saved weights against every boss tier. Greedy, on a
                # fixed set of maps, so successive evaluations differ because the agent changed
                # rather than because the maps did. Reuses the rollout worker pool.
                if eval_episodes > 0 and eval_boss_list:
                    eval_start = time.time()
                    results = evaluate_all_bosses(
                        model, model_kwargs, eval_episodes,
                        eval_max_turns if eval_max_turns is not None else max_turns,
                        allow_skip_disrupt, bosses=eval_boss_list, pool=pool,
                        force_disrupt=force_disrupt,
                    )
                    log_eval_to_tensorboard(writer, results, iteration)
                    print(format_eval_table(
                        results,
                        title=f"\n  eval @ iter {iteration + 1} - {eval_episodes} greedy episodes vs each boss "
                              f"({time.time() - eval_start:.0f}s)",
                    ))
                    print()
                    model.train()  # evaluate_all_bosses leaves nothing on the model, but the
                                   # next PPO update expects train mode regardless
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    writer.close()


def _build_parser():
    parser = argparse.ArgumentParser(description="Train the Railroad Tycoon agent with PPO")
    parser.add_argument("--config", type=str, default=None, help="YAML file of defaults (see training/configs/); any other flag overrides its value")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.5, help="Exploration rate (probability of a real stochastic sample vs argmax) at iteration 0")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Exploration rate at the final iteration - decays linearly from epsilon_start")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=None, help="Linearly anneal the learning rate to this by the final iteration (omit to keep it flat)")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channels", type=int, nargs=3, default=None, metavar=("C1", "C2", "C3"), help="Explicit stage widths, overriding the doubling progression from --base-channels (e.g. --channels 34 34 36 to fine-tune a distilled student)")
    parser.add_argument("--value-hidden", type=int, default=256)
    parser.add_argument("--score-norm", type=float, default=DEFAULT_SCORE_NORM, help="Per-turn rewards are divided by this before GAE")
    parser.add_argument("--allow-skip-disrupt", action=argparse.BooleanOptionalAction, default=True, help="Let the agent decline to disrupt (adds a third policy channel)")
    parser.add_argument("--reward-mode", type=str, default="own", choices=["own", "margin"], help="'own' = points gained this turn; 'margin' = own gain minus the opponent's")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Greedy benchmark games per boss after each checkpoint (0 disables)")
    parser.add_argument("--eval-max-turns", type=int, default=None, help="Turns per evaluation game (defaults to --max-turns)")
    parser.add_argument("--eval-bosses", type=str, default=",".join(BOSS_TIERS))
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Initialize model weights from this .pt file before training (fresh optimizer state - a warm start, not a resume)")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grid-height", type=int, default=None, help="Pin the map height (width follows the 1.5 aspect ratio). Omit for the authentic random 14-20.")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--opponent-strategy", type=str, default="passive", choices=["passive", "boss", "random", "greedy"])
    parser.add_argument("--opponent-wait-probability", type=float, default=0.0, help="For the random opponent: per-turn chance it skips placing")
    parser.add_argument("--opponent-disrupt-probability", type=float, default=0.5, help="For the random opponent: per-turn chance it disrupts")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1, help="Play games_per_iteration games across this many parallel worker processes instead of one after another (1 = no multiprocessing)")
    return parser


def _parse_args():
    parser = _build_parser()

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            overrides = yaml.safe_load(f) or {}

        valid_keys = set(inspect.signature(train_ppo).parameters)
        unknown = set(overrides) - valid_keys
        if unknown:
            raise ValueError(f"Unknown key(s) in {pre_args.config}: {sorted(unknown)} - check for typos against train_ppo()'s parameters")

        parser.set_defaults(**overrides)

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.pop("config")
    return args_dict


if __name__ == "__main__":
    train_ppo(**_parse_args())

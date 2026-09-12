"""PPO training loop for the lite game (railroad_lite_env).

LiteUNet is the shared actor-critic: a 2-down/2-up U-Net with a convolution before and after,
its single-channel policy head producing a PLACE heatmap and its critic reading the bottleneck.
The PPO update itself is shared with the full-game trainer (training/ppo.py::ppo_update).

Run: python3 -m training.train_lite_ppo [--config training/configs/ppo_lite.yaml] [--num-iterations N]
Any CLI flag overrides the same key from --config.
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

from .lite import collect_lite_rollout
from .lite_model import LiteUNet
from .ppo import ppo_update


def train_lite_ppo(
    num_iterations: int = 1000,
    games_per_iteration: int = 16,
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
    base_channels: int = 32,
    value_hidden: int = 128,
    checkpoint_every: int = 50,
    checkpoint_dir: str = "checkpoints",
    init_checkpoint: str = None,
    log_dir: str = None,
    device: str = None,
    grid_height: int = None,
    max_turns: int = 25,
    opponent_wait_probability: float = 0.5,
    seed: int = None,
    num_workers: int = 1,
):
    # locals() captured before any other local exists, so this is exactly the resolved config -
    # written into log_dir so every run directory is self-documenting.
    effective_config = dict(locals())

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    run_name = time.strftime("%Y%m%d-%H%M%S") + "-lite"
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

    model_kwargs = dict(base_channels=base_channels, value_hidden=value_hidden)
    model = LiteUNet(**model_kwargs).to(device)
    if init_checkpoint is not None:
        # Weights only, fresh optimizer state - a warm start, not a resume. This run's iteration
        # numbers start at 0 regardless of how far the source run got.
        init_ckpt = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"Initialized weights from {init_checkpoint} (trained for {init_ckpt.get('iteration', '?')} iterations)")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"LiteUNet: {sum(p.numel() for p in model.parameters()):,} parameters")

    env_kwargs = dict(
        grid_height=grid_height,
        max_turns=max_turns,
        opponent_wait_probability=opponent_wait_probability,
    )

    rng = np.random.RandomState(seed)

    # Worker processes (never threads - the GIL would serialize the game logic) with the "spawn"
    # start method, since forking a process that already holds a CUDA context is unsafe.
    pool = mp.get_context("spawn").Pool(processes=num_workers) if num_workers > 1 else None

    try:
        for iteration in range(num_iterations):
            iter_start = time.time()

            decay_progress = iteration / max(1, num_iterations - 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * decay_progress

            transitions, final_scores_list, episode_lengths = collect_lite_rollout(
                env_kwargs, model, model_kwargs, device, gamma, gae_lambda,
                games_per_iteration, rng, epsilon=epsilon, pool=pool,
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
            writer.add_scalar("self_play/epsilon", epsilon, iteration)
            writer.add_scalar("self_play/seconds", rollout_time, iteration)

            train_start = time.time()
            stats = ppo_update(
                model, optimizer, transitions, ppo_epochs, minibatch_size,
                clip_epsilon, value_coef, entropy_coef, max_grad_norm, device,
            )
            train_time = time.time() - train_start

            for key in ("policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl"):
                writer.add_scalar(f"train/{key}", stats[key], iteration)
            writer.add_scalar("train/seconds", train_time, iteration)

            print(
                f"[iter {iteration}] score={np.mean(player0_scores):.1f} "
                f"margin={np.mean(score_margins):+.1f} win_rate={wins:.2f} eps={epsilon:.3f} "
                f"policy_loss={stats['policy_loss']:+.4f} value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.3f} clip_frac={stats['clip_frac']:.3f} "
                f"rollout={rollout_time:.1f}s train={train_time:.1f}s"
            )

            if (iteration + 1) % checkpoint_every == 0:
                ckpt_file = checkpoint_path / f"{run_name}_iter{iteration + 1}.pt"
                torch.save(
                    {
                        "iteration": iteration + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_kwargs": model_kwargs,
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
    parser = argparse.ArgumentParser(description="Train a PPO agent on the lite Railroad game")
    parser.add_argument("--config", type=str, default=None, help="YAML file of defaults; any other flag overrides its value")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.5)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--grid-height", type=int, default=None, help="Pin the map height; omit for the authentic random 14-20")
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--opponent-wait-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def _parse_args():
    parser = _build_parser()

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            overrides = yaml.safe_load(f) or {}
        valid_keys = set(inspect.signature(train_lite_ppo).parameters)
        unknown = set(overrides) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown key(s) in {pre_args.config}: {sorted(unknown)} - check for typos "
                "against train_lite_ppo()'s parameters"
            )
        parser.set_defaults(**overrides)

    args_dict = vars(parser.parse_args())
    args_dict.pop("config")
    return args_dict


if __name__ == "__main__":
    train_lite_ppo(**_parse_args())

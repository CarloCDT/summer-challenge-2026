#!/usr/bin/env python3
"""Benchmark every checkpoint against each boss tier and print the results.

Plays greedily (epsilon = 0, i.e. deployment behaviour) so the numbers reflect how the agent
would actually be run, and uses the SAME map seeds for every checkpoint/boss pair so the
comparison isn't confounded by map-difficulty luck.

    python3 evaluate_checkpoints.py
    python3 evaluate_checkpoints.py --episodes 100 --max-turns 100 --workers 16
    python3 evaluate_checkpoints.py --checkpoints "checkpoints/20260910-*.pt" --bosses level2,level2Pro
"""
import argparse
import glob
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from railroad_env.opponent import BOSS_TIERS
from training.evaluate import evaluate_vs_boss


def main():
    parser = argparse.ArgumentParser(description="Benchmark checkpoints against each boss tier")
    parser.add_argument("--checkpoints", type=str, default="checkpoints/*.pt",
                        help="Glob of checkpoints to evaluate (lite ones are skipped)")
    parser.add_argument("--bosses", type=str, default=",".join(BOSS_TIERS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=100)  # Game.MAX_TURNS
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the shared map set")
    parser.add_argument("--skip-disrupt", dest="skip_disrupt", default=None,
                        action=argparse.BooleanOptionalAction,
                        help="Whether the agent may decline to disrupt. Defaults to what each "
                             "checkpoint recorded. A run trained with force_disrupt never gave "
                             "its SKIP_DISRUPT channel a gradient, so offering the action here "
                             "scores frozen weights. Pass --no-skip-disrupt for checkpoints "
                             "written before that field was recorded.")
    args = parser.parse_args()

    bosses = [b.strip() for b in args.bosses.split(",") if b.strip()]
    paths = sorted(
        (p for p in glob.glob(args.checkpoints) if "lite" not in Path(p).name),
        key=lambda p: (Path(p).name.split("_iter")[0], int(Path(p).name.split("_iter")[-1].split(".")[0])),
    )
    if not paths:
        print(f"No checkpoints matched {args.checkpoints!r}")
        return 1

    # Same maps for every checkpoint and every boss, so differences are the agent, not the draw.
    seeds = [args.seed + i for i in range(args.episodes)]

    print(f"{len(paths)} checkpoint(s) x {len(bosses)} boss tier(s) x {args.episodes} episodes, "
          f"{args.max_turns} turns, greedy (epsilon=0)\n")

    pool = mp.get_context("spawn").Pool(processes=args.workers) if args.workers > 1 else None
    try:
        header = f"{'checkpoint':<34}{'boss':<16}{'score':>9}{'opp':>9}{'margin':>10}{'win':>7}{'conn':>7}{'zero':>7}"
        print(header)
        print("-" * len(header))
        for path in paths:
            blob = torch.load(path, map_location="cpu")
            model_kwargs = blob["model_kwargs"]
            state_dict = {k: v.cpu() for k, v in blob["model_state_dict"].items()}
            allow_skip = model_kwargs.get("policy_channels", 2) == 3
            # The head having three channels does not mean the third one was ever trained.
            force_disrupt = (not args.skip_disrupt if args.skip_disrupt is not None
                             else blob.get("force_disrupt", False))

            for i, boss in enumerate(bosses):
                t0 = time.time()
                r = evaluate_vs_boss(state_dict, model_kwargs, boss, seeds, args.max_turns,
                                     allow_skip, pool, force_disrupt=force_disrupt)
                label = Path(path).name.replace(".pt", "") if i == 0 else ""
                print(f"{label:<34}{boss:<16}{r['score']:>9.0f}{r['opp']:>9.0f}{r['margin']:>+10.0f}"
                      f"{r['win']:>7.2f}{r['conn']:>7.0%}{r['zero']:>7.0%}"
                      f"   ({time.time() - t0:.0f}s)")
            print()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print("score/opp/margin are per-game means; win = fraction won outright; "
          "conn = desired connections completed; zero = games scoring nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())

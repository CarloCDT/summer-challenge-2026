#!/usr/bin/env python3
"""Head-to-head round robin: every agent against every other, both seats.

WHY THIS EXISTS. The boss table (evaluate_checkpoints.py) scores agents against fixed opponents,
and it demonstrably does not predict CodinGame rank - a student that finished LAST here at 0.290
ranked HIGHER in the arena than the successor beating it 0.68 head-to-head. Playing candidates
directly against each other is the sharper instrument: it separates checkpoints a fixed-boss
table shows as identical, and it is the only way to see a quantization or distillation gap that
a +18,000 margin against level2 completely masks.

Each pairing plays N seeds with the seat alternating on seed parity, so seat advantage cancels;
results are reported from the first-named agent's point of view. Both players are asked for their
move BEFORE the turn resolves, which is how the Java referee does it (Referee.gameTurn sends
state to both and calls execute() on both before collecting either output) - nobody sees the
other's move first.

Agents are given as name=path. A .pt is a torch checkpoint; a .py is a baked submission, driven
as a subprocess over the real wire protocol, so "what we would actually send" is directly
comparable against "what we trained".

    python3 compare_agents.py teacher=checkpoints/<run>_iter300.pt shipped=submission.py
    python3 compare_agents.py a=checkpoints/x.pt b=checkpoints/y.pt --episodes 25 --workers 8

A note on reading the output: two agents with identical weights score 0:0 on every seed. They
pick the same cell each turn, every track resolves to NEUTRAL, and neutral track scores for
nobody. That is a correct result and a useful sanity check, not a bug.
"""
import argparse
import itertools
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from railroad_env import RailroadGymEnv
from railroad_env.opponent import BakedSubmissionOpponent


def build(spec):
    """spec is (kind, path, force_disrupt). Built inside the worker - models do not pickle well."""
    kind, path, force_disrupt = spec
    if kind == "baked":
        return BakedSubmissionOpponent(submission_path=path)
    from training.agent_model import RailroadUNet
    from training.self_play_opponent import SelfPlayOpponent
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = RailroadUNet(**blob["model_kwargs"])
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return SelfPlayOpponent(model, epsilon=0.0, allow_skip_disrupt=True,
                            force_disrupt=force_disrupt)


def one_game(spec_seat0, spec_seat1, seed, max_turns):
    torch.set_num_threads(1)
    env = RailroadGymEnv(max_turns=max_turns, opponent_strategy="passive", seed=seed)
    env.reset()
    gs = env.game_state
    a, b = build(spec_seat0), build(spec_seat1)
    try:
        while not gs.is_done():
            acts0 = a.get_actions(gs, 0)
            acts1 = b.get_actions(gs, 1)
            gs.resolve_turn([acts0, acts1])
            gs.do_income()
    finally:
        for agent in (a, b):
            if hasattr(agent, "close"):
                agent.close()
    return int(gs.scores[0]), int(gs.scores[1])


def _job(args):
    sa, sb, seed, flip, max_turns = args
    if flip:
        s1, s0 = one_game(sb, sa, seed, max_turns)
        return s0, s1
    return one_game(sa, sb, seed, max_turns)


def resolve(path):
    """A .pt is a torch checkpoint; anything else is treated as a baked submission file.

    force_disrupt is read from the checkpoint. It must match training: a run trained with it
    never gave its SKIP_DISRUPT channel a gradient, so offering the action scores frozen weights.
    Baked files carry the decision in the file itself (bake_agent.py bakes the branch in or out).
    """
    p = str(Path(path).resolve())
    if p.endswith(".pt"):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        return ("torch", p, bool(blob.get("force_disrupt", False)))
    return ("baked", p, False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="+", metavar="name=path",
                    help=".pt for a torch checkpoint, .py for a baked submission")
    ap.add_argument("--episodes", type=int, default=25, help="games per pairing")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7000, help="base seed for the shared map set")
    args = ap.parse_args()

    agents = {}
    for entry in args.agents:
        if "=" not in entry:
            ap.error(f"expected name=path, got {entry!r}")
        name, path = entry.split("=", 1)
        if not Path(path).exists():
            ap.error(f"{name}: no such file {path}")
        agents[name] = resolve(path)
    if len(agents) < 2:
        ap.error("need at least two agents")

    for name, (kind, path, fd) in agents.items():
        print(f"  {name:<16} {kind:<6} {Path(path).name}"
              + (f"   force_disrupt={fd}" if kind == "torch" else ""))

    jobs, index = [], []
    for x, y in itertools.combinations(agents, 2):
        for i in range(args.episodes):
            jobs.append((agents[x], agents[y], args.seed + i, i % 2 == 1, args.max_turns))
            index.append((x, y))

    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_job, jobs)

    agg = defaultdict(list)
    for key, value in zip(index, results):
        agg[key].append(value)

    print(f"\nHEAD-TO-HEAD, {args.episodes} episodes per pairing, {args.max_turns} turns, "
          f"seats alternated\n")
    print(f"{'matchup':<30}{'score':>16}{'margin':>10}{'win':>7}{'draws':>7}")
    print("-" * 70)
    wins, played = defaultdict(float), defaultdict(int)
    for (x, y), rows in agg.items():
        sx = np.array([r[0] for r in rows], float)
        sy = np.array([r[1] for r in rows], float)
        draws = int(np.sum(sx == sy))
        print(f"{x + ' vs ' + y:<30}{sx.mean():>7.0f} :{sy.mean():>7.0f}"
              f"{(sx - sy).mean():>+10.0f}{np.mean(sx > sy):>7.2f}{draws:>7}")
        wins[x] += float(np.sum(sx > sy)) + 0.5 * draws
        wins[y] += float(np.sum(sy > sx)) + 0.5 * draws
        played[x] += len(rows)
        played[y] += len(rows)

    print(f"\n{'overall win rate across all its games':<70}")
    print("-" * 70)
    for name in sorted(agents, key=lambda n: -wins[n] / played[n]):
        se = (0.25 / played[name]) ** 0.5
        print(f"  {name:<24}{wins[name] / played[name]:>8.3f}  "
              f"({wins[name]:.1f}/{played[name]}, SE {se:.3f})")
    print("\nSE is the standard error of a single win rate - differences smaller than about "
          "two of them are noise.")


if __name__ == "__main__":
    main()

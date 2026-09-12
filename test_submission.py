#!/usr/bin/env python3
"""Play a baked submission as a real subprocess, speaking the referee's exact wire protocol.

bake_agent.py --verify checks that the numpy network reproduces the torch one. This checks the
other half: that the generated file parses the referee's input format, emits actions the
referee's regexes accept, and scores what the torch agent scores.

    python3 test_submission.py submission.py --episodes 20
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from railroad_env import RailroadGymEnv
# One definition of the referee's protocol, shared with railroad_env.opponent's level2Silver
# boss - which drives a baked file exactly the way this script does.
from railroad_env.wire import frame_lines, init_lines, parse_actions


def play(path, seed, max_turns, opponent):
    env = RailroadGymEnv(max_turns=max_turns, opponent_strategy=opponent, seed=seed)
    env.reset()
    gs = env.game_state

    proc = subprocess.Popen([sys.executable, "-u", str(path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        for line in init_lines(gs, 0):
            proc.stdin.write(line + "\n")
        proc.stdin.flush()

        illegal = 0
        while not gs.is_done():
            for line in frame_lines(gs, 0):
                proc.stdin.write(line + "\n")
            proc.stdin.flush()
            reply = proc.stdout.readline()
            if not reply:
                err = proc.stderr.read()
                raise RuntimeError(f"submission died on seed {seed}:\n{err[-2000:]}")
            actions = parse_actions(reply)

            # Legality, judged the way the referee judges it.
            budget = gs.paint_points[0]
            for a in actions:
                if a[0] == "PLACE":
                    if not gs._is_placeable(a[1], a[2]) or gs.get_track_cost(a[1], a[2]) > budget:
                        illegal += 1
                    budget -= gs.get_track_cost(a[1], a[2])
                elif a[0] == "DISRUPT" and not gs.can_disrupt(a[1]):
                    illegal += 1

            env.step({"actions": actions})
        return gs.scores[0], gs.scores[1], illegal
    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=100)  # Game.MAX_TURNS
    ap.add_argument("--opponent", default="level2")
    args = ap.parse_args()

    own, opp, bad = [], [], 0
    for seed in range(args.episodes):
        a, b, ill = play(args.submission, seed, args.max_turns, args.opponent)
        own.append(a); opp.append(b); bad += ill
        print(f"  seed {seed:>3}  score {a:>6}  opp {b:>6}  {'ILLEGAL x%d' % ill if ill else ''}")

    own, opp = np.array(own, float), np.array(opp, float)
    print(f"\n{args.episodes} games vs {args.opponent}, {args.max_turns} turns")
    print(f"  score {own.mean():.0f}  opp {opp.mean():.0f}  "
          f"margin {(own - opp).mean():+.0f}  win {np.mean(own > opp):.2f}")
    print(f"  illegal actions: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

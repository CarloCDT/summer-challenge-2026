#!/usr/bin/env python3
"""Standalone scripted-player demo with pygame visualization (no neural network involved).

Run directly, from anywhere:
  $ python3 docs/play_with_rendering.py
  $ python3 docs/play_with_rendering.py --opponent boss --sleep 0.2

Player 0 uses AUTOPLACE to work through each town's desired connections in turn, which is a
decent scripted baseline and exercises the env's full action set.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railroad_env import RailroadGymEnv


def player_strategy(game_state):
    """Pick the first desired connection that isn't active yet and AUTOPLACE toward it."""
    for town in game_state.towns:
        for other in town.desired_connections:
            if other.id in town.paths:
                continue  # already connected
            return [("AUTOPLACE", town.coord[0], town.coord[1], other.coord[0], other.coord[1])]
    return [("WAIT",)]


def main():
    parser = argparse.ArgumentParser(description="Watch a scripted player play a full game")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["passive", "boss", "random", "greedy"])
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--grid-height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = RailroadGymEnv(
        grid_height=args.grid_height,
        opponent_strategy=args.opponent,
        seed=args.seed,
        render_mode="human",
    )
    obs, info = env.reset()
    print(f"Map {info['width']}x{info['height']}, {len(info['towns'])} towns")
    print(f"Player 0 = scripted AUTOPLACE, player 1 = {args.opponent}")
    print("Close the window to stop\n")

    try:
        while True:
            actions = player_strategy(env.game_state)
            obs, reward, terminated, _, info = env.step({"actions": actions})

            if info["turn"] % 10 == 0 or reward:
                print(
                    f"Turn {info['turn']:3d} | Score {info['scores'][0]:5d}-{info['scores'][1]:<5d} "
                    f"| reward {reward:+.0f} | placed {len(info['player_actions']['placed'])}"
                )

            if not info["window_open"]:
                print("\nWindow closed - stopping early.")
                break
            if terminated:
                print(f"\nGame ended at turn {info['turn']}")
                break
            time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        scores = env.game_state.scores
        winner = "Player" if scores[0] > scores[1] else ("Opponent" if scores[1] > scores[0] else "Tie")
        print(f"\nFinal score: {scores[0]} - {scores[1]} ({winner})")
        env.close()


if __name__ == "__main__":
    main()

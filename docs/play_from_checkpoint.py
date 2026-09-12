#!/usr/bin/env python3
"""Watch a trained checkpoint play a full game, with pygame visualization.

Run from anywhere:
  $ python3 docs/play_from_checkpoint.py --checkpoint checkpoints/<run>_iter500.pt
  $ python3 docs/play_from_checkpoint.py --checkpoint ... --opponent boss --sleep 0.3

The agent acts one atomic decision at a time (a PLACE_TRACKS cell or a DISRUPT target) through
GameSimulator, which accumulates them into whole turns exactly as training does.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railroad_env import RailroadGymEnv
from training.model import TrainUNet
from training.simulator import GameSimulator


def choose_action(model, sim, device, epsilon, rng):
    """One forward pass -> one atomic action, epsilon-greedy over the masked policy."""
    state = sim.get_encoded_state()
    mask = sim.get_action_mask()

    with torch.no_grad():
        x = torch.from_numpy(state).unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        logits, value = model(x, m)
        flat = logits.flatten(1)
        if rng.random_sample() < epsilon:
            index = int(torch.distributions.Categorical(logits=flat).sample().item())
        else:
            index = int(torch.argmax(flat, dim=1).item())

    _, height, width = mask.shape
    channel, remainder = divmod(index, height * width)
    row, col = divmod(remainder, width)
    pad_top, pad_left = sim.pad_offsets
    kind = "PLACE" if channel == 0 else "DISRUPT"
    return (kind, col - pad_left, row - pad_top), float(value.item())


def main():
    parser = argparse.ArgumentParser(description="Watch a checkpoint play a full game")
    parser.add_argument("--checkpoint", type=str, required=True, help="A .pt saved by training/train_ppo.py or training/train.py")
    parser.add_argument("--opponent", type=str, default="passive", choices=["passive", "boss", "random", "greedy"])
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds to pause between turns")
    parser.add_argument("--epsilon", type=float, default=0.0, help="0 = greedy/deployment; >0 samples from the policy")
    parser.add_argument("--grid-height", type=int, default=None, help="Pin the map height; omit for the authentic random 14-20")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--policy-channels", type=int, default=2, help="2 for PPO/MCTS checkpoints, 1 for a topk checkpoint")
    parser.add_argument("--bounded-value", action=argparse.BooleanOptionalAction, default=False,
                        help="Tanh value head - use --bounded-value for AlphaZero/MCTS checkpoints")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = TrainUNet(policy_channels=args.policy_channels, bounded_value=args.bounded_value).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded (trained for {ckpt.get('iteration', '?')} iterations), running on {device}\n")

    env = RailroadGymEnv(
        grid_height=args.grid_height,
        max_turns=args.max_turns,
        opponent_strategy=args.opponent,
        seed=args.seed,
        render_mode="human",
    )
    env.reset()
    gs = env.game_state
    print(f"Map {gs.width}x{gs.height}, {len(gs.towns)} towns")
    print(f"Player 0 = checkpoint agent, player 1 = {args.opponent} opponent")
    print("Close the pygame window to stop\n")

    sim = GameSimulator.from_env(env)

    try:
        while not sim.is_game_over():
            turn = sim.game_state.turn
            actions_this_turn = []
            value = 0.0
            while not sim.is_game_over() and sim.game_state.turn == turn:
                action, value = choose_action(model, sim, device, args.epsilon, rng)
                actions_this_turn.append(action)
                sim.apply(action)

            # Mirror the simulator's resolved state into the rendered env.
            env.game_state = sim.game_state
            if env.renderer is not None:
                env.renderer.game_state = sim.game_state

            scores = sim.game_state.scores
            print(
                f"Turn {sim.game_state.turn:3d} | Score {scores[0]:5d}-{scores[1]:<5d} | "
                f"value={value:+.2f} | {', '.join(f'{k} {x},{y}' for k, x, y in actions_this_turn)}"
            )

            if not env.render():
                print("\nWindow closed - stopping early.")
                break
            time.sleep(args.sleep)

        if sim.is_game_over():
            scores = sim.game_state.scores
            winner = "Agent" if scores[0] > scores[1] else ("Opponent" if scores[1] > scores[0] else "Tie")
            print(f"\nGame finished at turn {sim.game_state.turn}. Final score: {scores} - {winner}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()

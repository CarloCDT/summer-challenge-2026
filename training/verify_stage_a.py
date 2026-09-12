"""End-to-end verification of the *training* stack: encoding, GameSimulator, TrainUNet and MCTS
working together against the real environment. Not training - just correctness.

For the game rules themselves (map generation, scoring, disruption, ...) see `verify_rules.py`
at the repo root, which checks them against the original Java referee.

Run: python3 training/verify_stage_a.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from railroad_env import RailroadGymEnv
from training.encoding import BOARD_HEIGHT, BOARD_WIDTH, compute_pad_offsets
from training.mcts import MCTS, visit_count_policy
from training.model import IN_CHANNELS, TrainUNet
from training.simulator import GameSimulator


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise AssertionError(label)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    env = RailroadGymEnv(grid_height=17, opponent_strategy="random", seed=42, opponent_seed=7)
    env.reset()
    sim = GameSimulator.from_env(env)
    gs = sim.game_state

    # --- Encoding / mask shapes ---
    encoded = sim.get_encoded_state()
    check(
        f"encoded state shape is ({IN_CHANNELS}, {BOARD_HEIGHT}, {BOARD_WIDTH})",
        encoded.shape == (IN_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
    )
    mask = sim.get_action_mask()
    check(f"action mask shape is (2, {BOARD_HEIGHT}, {BOARD_WIDTH})",
          mask.shape == (2, BOARD_HEIGHT, BOARD_WIDTH))

    # --- Padding: the real board is centered, everything outside reads as inked ---
    pad_top, pad_left = compute_pad_offsets(gs.height, gs.width)
    check("pad offsets center the board in the canvas",
          (pad_top, pad_left) == ((BOARD_HEIGHT - gs.height) // 2, (BOARD_WIDTH - gs.width) // 2))
    from railroad_env.game_state import GameState
    inked_channel = encoded[GameState.INKED_CHANNEL]
    off_board_inked = inked_channel[:pad_top, :].all() if pad_top else True
    check("off-board padding is marked as inked", bool(off_board_inked))

    # --- Legal actions line up with the mask ---
    legal = sim.get_legal_actions()
    check("legal actions are non-empty at the start", len(legal) > 0)
    check("every legal action is marked in the mask",
          all(mask[0 if kind == "PLACE" else 1, pad_top + y, pad_left + x] == 1.0
              for kind, x, y in legal))
    check("mask marks exactly the legal actions", int(mask.sum()) == len(legal))

    # --- Illegal cells really are excluded ---
    town_coord = gs.towns[0].coord
    check("a town cell is not a legal PLACE target",
          mask[0, pad_top + town_coord[1], pad_left + town_coord[0]] == 0.0)
    town_zone = gs.get_region(*town_coord)
    check("a town's region is not a legal DISRUPT target",
          all(mask[1, pad_top + y, pad_left + x] == 0.0
              for (x, y) in gs.zones[town_zone].coords))

    # --- Model forward pass on the real encoded state ---
    model = TrainUNet(policy_channels=2).to(device)
    x = torch.from_numpy(encoded).unsqueeze(0).to(device)
    m = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = model(x, m)
    check("model forward pass produces finite policy/value",
          bool(torch.isfinite(policy_logits).all() and torch.isfinite(value).all()))
    check("masked-illegal logits are hugely negative", bool((policy_logits[m == 0] <= -1e8).all()))

    # --- MCTS on the first decision ---
    mcts = MCTS(model, device=device, c_puct=1.5, num_simulations=20)
    t0 = time.time()
    visit_counts = mcts.run(sim)
    print(f"\nMCTS.run(20 simulations) took {time.time() - t0:.2f}s")

    check("visit_counts only covers legal actions", set(visit_counts).issubset(set(legal)))
    check("visit_counts is non-empty", len(visit_counts) > 0)
    check("total visits respects the simulation budget", sum(visit_counts.values()) <= 20)

    policy = visit_count_policy(visit_counts, temperature=1.0)
    check("visit_count_policy sums to ~1.0", abs(sum(policy.values()) - 1.0) < 1e-6)
    print(f"Most-visited root action: {max(visit_counts, key=visit_counts.get)}")

    # --- Drive a few real turns through the simulator ---
    print("\nPlaying 3 real turns via MCTS-selected actions...")
    turns_completed = 0
    while turns_completed < 3 and not sim.is_game_over():
        visit_counts = mcts.run(sim)
        sim.apply(max(visit_counts, key=visit_counts.get))
        if sim.last_turn_reward is not None:
            turns_completed += 1
            print(f"  turn {turns_completed}: reward={sim.last_turn_reward}, scores={sim.game_state.scores}")
            sim.last_turn_reward = None

    check("completed 3 real turns without error", turns_completed == 3)
    check("the original env is untouched by the simulator", env.game_state.turn == 0)

    print("\nAll training-stack checks passed.")


if __name__ == "__main__":
    main()

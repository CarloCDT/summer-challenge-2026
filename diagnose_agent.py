#!/usr/bin/env python3
"""Behavioural diagnostics for a checkpoint: what is it actually DOING, not just scoring?

    python3 diagnose_agent.py checkpoints/<a>.pt checkpoints/<b>.pt 8

The two columns that matter and are invisible in the eval table:

  disrupts / skips - the leading indicator of collapse. A healthy policy spends ~96% of its
    disruption opportunities; one that has drifted declines ~48% of them. Disruption points are
    1/turn and never carry over, so a declined disrupt destroys the resource outright. This
    moved well before the eval table noticed.

  peak vs end connections - `conn` in the eval table is measured at GAME END, so against an
    inking opponent ~0% is normal and means "everything got inked", NOT "it never built".
    Peak-during-game is the real build metric; it runs 93-100% even when end-of-game is 0%.

level2ProMax inks the region where ENEMY track most outnumbers its own, once that lead is > 1.
From our agent's side that means: a region where (our tracks - boss tracks) > 1 is a live ink
target. Counting those per turn measures how much of a target the policy makes of itself.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/carlo/summer-challenge-2026")
from railroad_env import RailroadGymEnv
from training.agent_model import RailroadUNet
from training.simulator import GameSimulator
from training.ppo import _decode_action


def load(p):
    b = torch.load(p, map_location="cpu", weights_only=False)
    m = RailroadUNet(**b["model_kwargs"])
    m.load_state_dict(b["model_state_dict"])
    m.eval()
    return m


def run(model, seed, max_turns=100):
    env = RailroadGymEnv(max_turns=max_turns, opponent_strategy="level2ProMax", seed=seed)
    env.reset()
    sim = GameSimulator.from_env(env, allow_skip_disrupt=True, reward_mode="margin")
    gs = sim.game_state

    targets, peak_conn, kinds = [], 0, {"PLACE": 0, "DISRUPT": 0, "SKIP_DISRUPT": 0}
    turn_seen = -1
    while not sim.is_game_over():
        st, mk = sim.get_encoded_state(), sim.get_action_mask()
        with torch.no_grad():
            lg, _ = model(torch.from_numpy(st).unsqueeze(0), torch.from_numpy(mk).unsqueeze(0))
            a = int(torch.argmax(lg.flatten(1), dim=1))
        act = _decode_action(a, mk.shape, sim.pad_offsets)
        kinds[act[0]] = kinds.get(act[0], 0) + 1
        sim.apply(act)

        if gs.turn != turn_seen:
            turn_seen = gs.turn
            own = np.bincount(gs.regions.ravel(), weights=(gs.tracks == 0).ravel(),
                              minlength=len(gs.zones))
            foe = np.bincount(gs.regions.ravel(), weights=(gs.tracks == 1).ravel(),
                              minlength=len(gs.zones))
            live = [not z.inked and not z.contained_towns for z in gs.zones]
            targets.append(int(sum(1 for i in range(len(gs.zones))
                                   if live[i] and (own[i] - foe[i]) > 1)))
            peak_conn = max(peak_conn, sum(len(t.paths) for t in gs.towns))

    wanted = sum(len(t.desired_connections) for t in gs.towns)
    return {
        "score": gs.scores[0], "opp": gs.scores[1], "margin": gs.scores[0] - gs.scores[1],
        "end_conn": sum(len(t.paths) for t in gs.towns), "peak_conn": peak_conn, "wanted": wanted,
        "inked": sum(1 for z in gs.zones if z.inked),
        "own_track": int((gs.tracks == 0).sum()), "foe_track": int((gs.tracks == 1).sum()),
        "ink_targets_per_turn": float(np.mean(targets)),
        "disrupts": kinds.get("DISRUPT", 0), "skips": kinds.get("SKIP_DISRUPT", 0),
    }


def main(paths, n):
    for p in paths:
        model = load(p)
        rows = [run(model, s) for s in range(n)]
        agg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        made = sum(r["end_conn"] for r in rows); want = sum(r["wanted"] for r in rows)
        peak = sum(r["peak_conn"] for r in rows)
        print(f"\n{p.split('/')[-1]}  ({n} games vs level2ProMax, 100 turns)")
        print(f"  margin {agg['margin']:+8.0f}   score {agg['score']:7.0f}  opp {agg['opp']:7.0f}"
              f"   win {np.mean([r['margin'] > 0 for r in rows]):.2f}")
        print(f"  connections  end {made}/{want} ({made/want:.0%})   peak-during-game "
              f"{peak}/{want} ({peak/want:.0%})")
        print(f"  track laid   own {agg['own_track']:5.1f}   boss {agg['foe_track']:5.1f}")
        print(f"  regions inked {agg['inked']:.1f}   live ink-targets per turn "
              f"{agg['ink_targets_per_turn']:.2f}")
        print(f"  disrupts {agg['disrupts']:.0f}  skips {agg['skips']:.0f} "
              f"({agg['skips']/max(1,agg['disrupts']+agg['skips']):.0%} declined)")


if __name__ == "__main__":
    main(sys.argv[1:-1], int(sys.argv[-1]))

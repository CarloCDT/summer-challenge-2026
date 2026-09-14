#!/usr/bin/env python3
"""Prove the baked runtime observes the board exactly the way training does.

`bake_agent.py --verify` checks the numpy NETWORK against torch, but it feeds both sides the
training encoding, so it can never catch a bug in the runtime's own `observe()`. This checks that
half. It execs the runtime of a real bake with a scripted stdin, feeds it the referee's wire
frames for BOTH seats of real games, and compares every channel the runtime builds against
`encode_state` for the same player.

Channels 28-37 matter most here. The runtime rebuilds them from wire data alone - connection
lists per cell, and the opponent's disruption inferred from instability deltas - while training
reads them straight off GameState, so this is where railroad_env/features.py and its copy inside
bake_agent.RUNTIME would drift apart.

    python3 verify_bake_parity.py                      # a random 38-channel net, 6 games
    python3 verify_bake_parity.py checkpoints/<run>.pt --games 12
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bake_agent as BA
from railroad_env import RailroadGymEnv
from railroad_env.game_state import GameState
from railroad_env.opponent import make_opponent
from railroad_env.wire import frame_lines, init_lines
from training.agent_model import RailroadUNet
from training.encoding import encode_state

# Builders, a random disruptor and a targeted inker, so neutral track, instability, inked regions,
# early game-over and the opponent-disruption memory all get exercised from both seats.
PAIRINGS = [("level1Pro", "level2ProMax"), ("level2ProMax", "greedy"), ("greedy", "level1Pro")]
TOLERANCE = 1e-5


def load_runtime(src, init):
    """Exec everything above the runtime's main loop, with `input()` reading from a queue."""
    if BA.MAIN_LOOP_MARKER not in src:
        raise SystemExit("verify_bake_parity: the runtime has no main-loop marker - "
                         "bake_agent.RUNTIME changed shape; update this script with it.")
    feed = deque(init)

    def scripted_input():
        if not feed:
            raise EOFError
        return feed.popleft()

    ns = {"input": scripted_input, "__name__": "baked_runtime"}
    exec(compile(src.split(BA.MAIN_LOOP_MARKER)[0], "<baked runtime>", "exec"), ns)
    return ns, feed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="bake this checkpoint (default: a random net)")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=31000)
    args = ap.parse_args()

    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        kwargs, sd = blob["model_kwargs"], blob["model_state_dict"]
    else:
        kwargs = dict(in_channels=GameState.NUM_CHANNELS, policy_channels=3,
                      channels=(8, 8, 8), value_hidden=8)
        sd = RailroadUNet(**kwargs).state_dict()
    nch = kwargs["in_channels"]
    payload, shapes = BA.quantize(BA.fold_batchnorm(sd), "int4")
    src = BA.build_source(payload, shapes, "int4", kwargs["policy_channels"], 3,
                          allow_skip=False, in_channels=nch)

    worst = np.zeros(nch)
    frames = 0
    for g in range(args.games):
        names = PAIRINGS[g % len(PAIRINGS)]
        env = RailroadGymEnv(max_turns=100, opponent_strategy="passive", seed=args.seed + g)
        env.reset()
        gs = env.game_state
        players = [make_opponent(names[0], seed=g), make_opponent(names[1], seed=g + 1)]
        runtimes = [load_runtime(src, init_lines(gs, pid)) for pid in (0, 1)]
        while not gs.is_done():
            for pid, (ns, feed) in enumerate(runtimes):
                feed.extend(frame_lines(gs, pid))
                my_score, foe_score, track, inst, inked_z, active, conn, pairs = ns["read_frame"]()
                ns["note_instability"](inst)
                got = ns["observe"](track, inst, inked_z, active, conn, pairs,
                                    my_score - foe_score, gs.turn)
                want = encode_state(gs, player=pid)[:nch]
                worst = np.maximum(worst, np.abs(got - want).reshape(nch, -1).max(axis=1))
                frames += 1
            actions = [p.get_actions(gs, pid) for pid, p in enumerate(players)]
            # Tell each runtime what it disrupted, the way its own main loop records it. Only the
            # first legal DISRUPT spends the point, exactly as in GameState._do_actions.
            for pid, (ns, _) in enumerate(runtimes):
                ns["LAST_DZ"] = next((int(a[1]) for a in actions[pid]
                                      if a[0] == "DISRUPT" and gs.can_disrupt(int(a[1]))), -1)
            gs.resolve_turn(actions)
            gs.do_income()
        print(f"  game {g}: {names[0]} vs {names[1]}, {gs.turn} turns, score {gs.scores}")

    bad = np.flatnonzero(worst > TOLERANCE)
    print(f"\n{frames} observations compared ({nch} channels, both seats)")
    for ch in range(nch):
        flag = "  MISMATCH" if worst[ch] > TOLERANCE else ""
        print(f"  ch {ch:>2}  max |runtime - encode_state| = {worst[ch]:.3g}{flag}")
    if len(bad):
        print(f"\nFAIL: channels {bad.tolist()} differ between the bake and training.")
        return 1
    print("\nPASS: the baked runtime reproduces encode_state on every channel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

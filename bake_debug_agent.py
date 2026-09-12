#!/usr/bin/env python3
"""Bake a checkpoint into a submission that ALSO reports what the critic thinks, on stderr.

This is `bake_agent.py` plus the value head, for debugging only. The shipped bake deliberately
drops the critic - PPO's value function is training-only and the policy head is what plays - so
`submission.py` can tell you what it did but never why it thought it was winning. This file keeps
the critic and prints, every turn:

    T 42 | 3120-2890 (+230) | V +1.832 | P(win) me 0.71 foe 0.29 | PLACE_TRACKS 12 7;DISRUPT 5

THE POLICY IS UNCHANGED. Same layers, same quantization, same decision order as `bake_agent.py`,
so at the same --quant this file picks the same cells as the real submission and is a faithful
stand-in for it. The only additions are the two value-head Linears and the stderr line.

WHAT P(win) ACTUALLY IS, because it is easy to over-read. RailroadUNet's value head is UNBOUNDED:
it regresses the discounted future return, in score_norm units, not a probability. There is no
principled way to read a probability off it, so this script FITS one - it plays `--calibrate N`
games with the torch model, collects (value, did-player-0-win) pairs, fits a two-parameter
logistic P = sigmoid(a*V + b), and bakes `a` and `b` into the file. Consequences:

  - The fit is only as good as N. Each game contributes ~250 values but exactly ONE outcome, so
    the effective sample size is N, not 250*N. N=25 is a sketch; use a few hundred if you care.
  - It is calibrated for the OPPONENT MIX it was calibrated against (--calibrate-opponent).
    Against a different opponent the ordering still holds but the numbers will be off.
  - With --calibrate 0 the file still runs and still prints V, but P(win) is a raw sigmoid of the
    value and the runtime says so on its first line. Trust the ordering, not the number.
  - Checkpoints trained with `win_bonus` carry the match outcome in the return directly, so their
    critic should calibrate far better than one trained on margin alone. Worth re-fitting after
    the first win_bonus run and comparing the Brier scores.

STDERR IS NOT FREE. A referee spawns this with stderr on a pipe and may not drain it until the
process exits (`railroad_env/opponent.py` only reads it on failure). At ~110 bytes a turn over 100
turns that is ~11 KB against a typical 64 KB pipe buffer, so it fits - but do not add a per-cell
dump to this line without checking that arithmetic again, or the agent will deadlock mid-game
rather than fail cleanly.

    python3 bake_debug_agent.py checkpoints/<run>_iterN.pt -o debug_submission.py --calibrate 25
    python3 test_submission.py debug_submission.py          # stderr shows the critic's read

QUANTIZATION IS A CHOICE BETWEEN THE TWO HEADS, and you cannot have both. Measured on
`114307_iter300` over 30 turns, baked value against torch value on the same state:

    --quant fp16    max relative error  0.20%, mean 0.03%    <- the default, trust the critic
    --quant int4    max relative error 42.78%, mean 5.48%    <- matches submission.py's moves

int4 is what ships, so it is the only mode whose POLICY is bit-identical to `submission.py` - but
4-bit weights wreck a scalar regression in a way they demonstrably do not wreck an argmax over
~1800 cells (see bake_agent.quantize: int4 costs no score despite only ~62% move agreement). So
fp16 is the default here: this tool exists to read the critic, and a 43% error would make that
reading fiction. Reach for int4 only when you are debugging the POLICY and need the exact moves
the arena sees, and ignore V when you do. Size does not matter either way - this file is never
submitted, and fp16 runs about 5 MB.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bake_agent as BA

# The critic, as two 1x1 "convolutions". A Linear weight is (out, in); reshaping it to
# (out, in, 1, 1) makes it indistinguishable from a 1x1 conv, which means the baked loader's
# existing kh==1 branch already turns it back into an (out, in) matrix. No new payload format,
# no new dequantization path - the quantizer's per-output-channel scale is per output unit here,
# which is exactly what it should be for a Linear.
VALUE_LAYERS = [("v1", "value_head.2"), ("v2", "value_head.5")]


def value_params(sd):
    """The two Linears, shaped as 1x1 convs. No BatchNorm in the value head, so nothing to fold,
    and Dropout is an identity in eval mode - which is the mode that ships."""
    out = []
    for name, prefix in VALUE_LAYERS:
        w = sd[f"{prefix}.weight"].double().numpy()
        b = sd[f"{prefix}.bias"].double().numpy()
        out.append((name, w[:, :, None, None].astype(np.float32), b.astype(np.float32)))
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_calibration(values, wins, iters=4000, lr=0.05):
    """Two-parameter logistic fit, P(win) = sigmoid(a*v + b), by plain gradient descent.

    Deliberately not sklearn: this is two parameters and the repo has no such dependency.
    `values` is standardised internally and the coefficients are mapped back, because raw values
    span a wide range and an unscaled fit converges badly."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(wins, dtype=np.float64)
    mu, sd = v.mean(), v.std() + 1e-9
    z = (v - mu) / sd
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = _sigmoid(a * z + b)
        err = p - y
        a -= lr * (err * z).mean()
        b -= lr * err.mean()
    # undo the standardisation: a*(v-mu)/sd + b  ==  (a/sd)*v + (b - a*mu/sd)
    return a / sd, b - a * mu / sd


def collect_calibration(model, kwargs, episodes, opponent, force_disrupt, seed0=90000):
    """Play `episodes` greedy games with the torch model, returning (values, wins).

    One row per DECISION but one outcome per GAME, so the effective sample size is `episodes`.
    Uses the torch model rather than the baked one: the quantized critic differs slightly, but by
    far less than the sampling noise of any affordable number of games."""
    from railroad_env import RailroadGymEnv
    from training.simulator import GameSimulator
    from training.ppo import _decode_action

    torch.set_num_threads(1)
    values, wins = [], []
    for i in range(episodes):
        env = RailroadGymEnv(max_turns=100, opponent_strategy=opponent, seed=seed0 + i)
        env.reset()
        sim = GameSimulator.from_env(
            env, allow_skip_disrupt=kwargs.get("policy_channels", 3) == 3,
            force_disrupt=force_disrupt,
        )
        game_values = []
        while not sim.is_game_over():
            mask = sim.get_action_mask()
            with torch.no_grad():
                logits, value = model(
                    torch.from_numpy(sim.get_encoded_state()).unsqueeze(0),
                    torch.from_numpy(mask).unsqueeze(0),
                )
            game_values.append(float(value.item()))
            sim.apply(_decode_action(int(torch.argmax(logits.flatten(1), dim=1).item()),
                                     mask.shape, sim.pad_offsets))
        gs = sim.game_state
        won = 1.0 if gs.scores[0] > gs.scores[1] else 0.0
        values.extend(game_values)
        wins.extend([won] * len(game_values))
        print(f"  calibration game {i + 1}/{episodes}: {gs.scores[0]}:{gs.scores[1]} "
              f"{'win' if won else 'loss/draw'}", flush=True)
    return np.array(values), np.array(wins)


# --- runtime patches, applied to bake_agent.RUNTIME ---------------------------------------
# Every replacement asserts it matched exactly once. If bake_agent.py's runtime is edited in a
# way that breaks one of these, this script fails loudly at bake time rather than quietly baking
# a debug agent that no longer mirrors the real one.

_VALUE_FNS = '''
def _dbg_apool(x,ph,pw):
    """torch AdaptiveAvgPool2d: window i is [floor(i*n/out), ceil((i+1)*n/out)), so for
    (5,7)->(2,3) the windows OVERLAP. Getting this wrong shifts the value silently."""
    c,h,w=x.shape
    o=np.empty((c,ph,pw),np.float32)
    for i in range(ph):
        a=(i*h)//ph; b=-((-(i+1)*h)//ph)
        for j in range(pw):
            u=(j*w)//pw; v=-((-(j+1)*w)//pw)
            o[:,i,j]=x[:,a:b,u:v].mean(axis=(1,2))
    return o

def _dbg_value(bo):
    """AdaptiveAvgPool -> flatten -> Linear -> ReLU -> (Dropout, identity in eval) -> Linear."""
    f=_dbg_apool(bo,_DBG_VPH,_DBG_VPW).reshape(-1)
    w1,b1,_,_=P["v1"]; w2,b2,_,_=P["v2"]
    return float((w2@np.maximum(w1@f+b1,0.0)+b2)[0])

'''

_MAIN_FWD_OLD = '''    logits=_fwd(observe(track,inst,inked_z,active,my_score-foe_score,turn))'''
_MAIN_FWD_NEW = '''    logits,_dbg_val=_fwd(observe(track,inst,inked_z,active,my_score-foe_score,turn))'''

_PRINT_OLD = '''    print(";".join(acts) if acts else "WAIT", flush=True)
    turn+=1'''
_PRINT_NEW = '''    _dbg_cmd=";".join(acts) if acts else "WAIT"
    print(_dbg_cmd, flush=True)

    # DEBUG ONLY, and on stderr so it cannot corrupt the referee's stdout protocol. Keep this
    # line short: a referee may leave stderr on an undrained pipe for the whole game, so a
    # verbose dump here deadlocks the agent instead of failing cleanly. See this file's header.
    if turn==0 and not _DBG_CAL:
        print("!! P(win) is UNCALIBRATED (baked with --calibrate 0): it is a raw sigmoid of an "
              "unbounded value, not a probability. Ordering is meaningful, the number is not.",
              file=sys.stderr, flush=True)
    _dbg_p=1.0/(1.0+np.exp(-max(-60.0,min(60.0,_DBG_A*_dbg_val+_DBG_B))))
    print("T%-3d %6d-%-6d (%+d) | V %+9.4f | P(win) me %.3f foe %.3f | %s"
          %(turn,my_score,foe_score,my_score-foe_score,_dbg_val,_dbg_p,1.0-_dbg_p,_dbg_cmd),
          file=sys.stderr, flush=True)
    turn+=1'''


def build_debug_source(blob, shapes, quant, policy_channels, passive_income, allow_skip,
                       vph, vpw, cal_a, cal_b, calibrated):
    src = BA.build_source(blob, shapes, quant, policy_channels, passive_income,
                          allow_skip=allow_skip)

    # Guard against the failure that cost a debug cycle: the runtime already defines short
    # underscore names (_PB, _CB, _UC, _Q, _SH, _W, P...), and an injected constant that collides
    # with one is silently REBOUND by the runtime's own assignment later in the file. _CB as a
    # calibration coefficient became the im2col buffer cache, which surfaced only as a TypeError
    # mid-game. Every name this script injects is checked against the base runtime first.
    for _name in ("_DBG_VPH", "_DBG_VPW", "_DBG_A", "_DBG_B", "_DBG_CAL",
                  "_dbg_apool", "_dbg_value", "_dbg_val", "_dbg_cmd", "_dbg_p"):
        if re.search(r"(?<![A-Za-z0-9_])" + _name + r"(?![A-Za-z0-9_])", src):
            raise SystemExit(
                f"bake_debug_agent: {_name} already exists in bake_agent.py's runtime. Pick "
                f"another name - a collision here is rebound silently and fails mid-game."
            )

    def sub(s, old, new, what):
        if s.count(old) != 1:
            raise SystemExit(
                f"bake_debug_agent: could not patch the runtime ({what}): expected exactly one "
                f"match, found {s.count(old)}. bake_agent.py's RUNTIME has changed - update the "
                f"patches in this file rather than shipping a debug agent that no longer mirrors "
                f"the real bake."
            )
        return s.replace(old, new)

    src = sub(src, "def _fwd(g):", _VALUE_FNS + "def _fwd(g):", "insert value head")
    src = sub(src, "    return _conv(d1,P[\"head\"],1)",
              "    return _conv(d1,P[\"head\"],1),_dbg_value(bo)", "_fwd returns the value")
    src = sub(src, _MAIN_FWD_OLD, _MAIN_FWD_NEW, "unpack the value in the main loop")
    src = sub(src, _PRINT_OLD, _PRINT_NEW, "stderr debug line")
    # Anchored on the numpy import, not on "_Q=" - that substring also appears in every
    # dequantization branch (_Q=="fp16" and friends), so it is not unique.
    header = (f"_DBG_VPH={vph}\n_DBG_VPW={vpw}\n_DBG_A={cal_a!r}\n_DBG_B={cal_b!r}\n"
              f"_DBG_CAL={calibrated!r}\n")
    src = sub(src, "import numpy as np\n\n_Q=", "import numpy as np\n\n" + header + "_Q=",
              "calibration constants")
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out", default="debug_submission.py")
    ap.add_argument("--quant", choices=("int8", "int4", "fp16"), default="fp16",
                    help="fp16 (default) keeps the CRITIC faithful (0.2%% max error vs torch), "
                         "which is the point of this tool. int4 is what ships and is the only "
                         "mode whose POLICY matches submission.py exactly, but it distorts the "
                         "value by up to 43%% - use it to debug moves, not the value")
    ap.add_argument("--calibrate", type=int, default=0, metavar="N",
                    help="play N greedy games to fit P(win) = sigmoid(a*V + b). 0 (default) bakes "
                         "an uncalibrated sigmoid and says so at runtime. Effective sample size "
                         "is N, not N*decisions, so N=25 is a sketch")
    ap.add_argument("--calibrate-opponent", default="level2Silver",
                    help="opponent the calibration games are played against (default: the frozen "
                         "rank-288 bake). The fit is only valid for this opponent's kind of game")
    ap.add_argument("--skip-disrupt", dest="skip_disrupt", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="as bake_agent.py: defaults to what the checkpoint recorded")
    args = ap.parse_args()

    blob_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kwargs = blob_ckpt["model_kwargs"]
    sd = blob_ckpt["model_state_dict"]
    print(f"{args.checkpoint}: {kwargs}")

    if args.skip_disrupt is None:
        allow_skip = not blob_ckpt.get("force_disrupt", False)
        source = ("checkpoint" if "force_disrupt" in blob_ckpt
                  else "default (checkpoint predates the field)")
    else:
        allow_skip, source = args.skip_disrupt, "--skip-disrupt flag"
    print(f"  SKIP_DISRUPT in the baked agent: {allow_skip}  [{source}]")

    cal_a, cal_b, calibrated = 1.0, 0.0, False
    if args.calibrate > 0:
        from training.agent_model import RailroadUNet
        model = RailroadUNet(**kwargs)
        model.load_state_dict(sd)
        model.eval()
        print(f"\ncalibrating on {args.calibrate} greedy games vs {args.calibrate_opponent} "
              f"(effective sample size is the GAME count, not the decision count)")
        values, wins = collect_calibration(
            model, kwargs, args.calibrate, args.calibrate_opponent,
            force_disrupt=bool(blob_ckpt.get("force_disrupt", False)))
        cal_a, cal_b = fit_calibration(values, wins)
        calibrated = True
        p = _sigmoid(cal_a * values + cal_b)
        base = wins.mean()
        print(f"\n  P(win) = sigmoid({cal_a:+.6f} * V {cal_b:+.6f})")
        print(f"  games {args.calibrate}, decisions {len(values):,}, base win rate {base:.3f}")
        print(f"  Brier {np.mean((p - wins) ** 2):.4f}  "
              f"(predicting the base rate every time scores {base * (1 - base):.4f})")
        print(f"  accuracy at 0.5: {np.mean((p > 0.5) == (wins > 0.5)):.3f}")
        brier, base_brier = float(np.mean((p - wins) ** 2)), base * (1 - base)
        if len(np.unique(wins)) < 2:
            print("\n  !! every calibration game had the SAME outcome, so the fit is meaningless. "
                  "Raise --calibrate, or pick an opponent this checkpoint does not sweep.")
        elif base > 0.85 or base < 0.15:
            print(f"\n  !! the calibration games were {base:.0%} one-sided, so there are barely "
                  f"any examples of the rarer outcome to fit. P(win) will be pulled toward "
                  f"{base:.2f} almost everywhere and will look confident when it is not. Use an "
                  f"opponent that actually contests this checkpoint, and more games.")
        elif brier > 0.9 * base_brier:
            print(f"\n  !! Brier {brier:.4f} barely beats the {base_brier:.4f} you get by "
                  f"predicting the base rate every time, so the value is carrying little outcome "
                  f"information here. Read the ORDERING, not the number.")

    baked = BA.fold_batchnorm(sd) + value_params(sd)
    blob, shapes = BA.quantize(baked, args.quant)
    from training.agent_model import VALUE_POOL_HW
    src = build_debug_source(blob, shapes, args.quant, kwargs["policy_channels"], 3,
                             allow_skip, VALUE_POOL_HW[0], VALUE_POOL_HW[1],
                             cal_a, cal_b, calibrated)
    Path(args.out).write_text(src)
    n_policy = sum(w.size + b.size for _, w, b in BA.fold_batchnorm(sd))
    n_value = sum(w.size + b.size for _, w, b in value_params(sd))
    print(f"\npolicy params {n_policy:,} + critic params {n_value:,} = {n_policy + n_value:,}")
    print(f"wrote {args.out}: {len(src):,} chars  [{args.quant}]")
    print("NOT submittable and not meant to be - the critic is dead weight in the arena, and "
          "this file exists to be read, not sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

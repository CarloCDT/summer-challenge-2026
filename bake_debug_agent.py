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

--probe N INSTEAD OF --calibrate N, when the file has to fit CodinGame's 100,000 character cap.
The value head is 27,905 parameters and pushes a student bake to ~108k, over the cap and therefore
unusable in the arena - which is the one place worth debugging, since our eval demonstrably does
not predict rank. `--probe` drops the value head entirely and fits a 217-parameter logistic probe
on the SAME pooled bottleneck, against actual game outcomes. That lands at ~91k, and the policy
stays byte-identical to `submission.py` (verified: 184 of 184 turns over two full games).

The probe's features are [216 pooled NN dims | score margin | turn], and WHICH SUBSET to use is
selected on held-out games alongside the penalty, because the answer is not what you would guess.
Measured on `144318-distill_iter50`, 100 games vs level2Silver:

    NN features only     held-out Brier 0.2481   (loses to the base rate; selection rejects it)
    margin + turn only   held-out Brier 0.1472   <- selected, accuracy 0.774
    base rate only       held-out Brier 0.2481

So the network contributes NOTHING to predicting the winner here, and the working readout is score
margin and turn. That is consistent with everything else about this checkpoint: it was distilled on
policy logits alone and its value head never received a gradient, so nothing ever asked its
features to encode who is winning. Fitting all 218 columns under one penalty is worse than fitting
two of them - 216 uninformative dimensions bury the pair that work, and the combined fit collapsed
to the base rate - which is exactly why the column subset is a selected hyperparameter.

That is a property of THAT checkpoint, not of the method. `distill.py` now has a
`value_loss_weight` that regresses the teacher's value alongside the policy KL, so a student
distilled with it has a trunk that was actually asked to encode who is winning. Re-run the
selection on such a student before assuming the NN columns are useless - if they start winning it,
that is the cleanest evidence the value term did its job. The same goes for `--calibrate`, which
was only ever a fiction on a random critic.

Four guards exist because the first four attempts each produced a confident wrong number instead:
the split is by GAME and stratified by outcome (a random split gave a 0.49 train against 0.67
held-out base rate, which swamped the score); "predict the base rate" is an explicit candidate, so
the probe is never worse than a constant; diverged fits are discarded rather than winning on a
nonsense Brier; and the column subset is selected rather than assumed.

STDERR IS NOT FREE. A referee spawns this with stderr on a pipe and may not drain it until the
process exits (`railroad_env/opponent.py` only reads it on failure). At ~110 bytes a turn over 100
turns that is ~11 KB against a typical 64 KB pipe buffer, so it fits - but do not add a per-cell
dump to this line without checking that arithmetic again, or the agent will deadlock mid-game
rather than fail cleanly.

    python3 bake_debug_agent.py checkpoints/<run>_iterN.pt -o debug_submission.py --calibrate 25
    python3 test_submission.py debug_submission.py          # stderr shows the critic's read

    # under CodinGame's 100k cap, so it can be pasted into the ARENA where stderr is visible
    # in the replay viewer - the only place the agent meets opponents we do not own
    python3 bake_debug_agent.py checkpoints/<student>.pt -o debug_submission.py \
        --quant int4 --probe 90

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
import base64
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


def _opponent_kwargs(opponent_path):
    """A BakedSubmissionOpponent normally plays the frozen bake under baselines/. Pointing it at
    another file is the only way to calibrate against an opponent that actually CONTESTS the
    checkpoint - and once a student beats the frozen bar ~0.9, a probability fit has almost no
    losses to learn from and collapses to the base rate no matter how many games you play."""
    return {"submission_path": opponent_path} if opponent_path else None


def collect_calibration(model, kwargs, episodes, opponent, force_disrupt, seed0=90000,
                        opponent_path=None):
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
        env = RailroadGymEnv(max_turns=100, opponent_strategy=opponent, seed=seed0 + i,
                             opponent_kwargs=_opponent_kwargs(opponent_path))
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


def collect_probe_data(model, kwargs, episodes, opponent, force_disrupt, seed0=95000,
                       opponent_path=None):
    """Per decision: the 216-dim pooled bottleneck the value head sees, plus who won the game.

    Returns (X, y, game_id) where X is [216 pooled NN dims | score margin | turn]. `game_id`
    matters - the validation split must be by GAME, because every decision in one game shares one
    outcome. Splitting by row would put near-duplicate states on both sides and report a fantasy
    score."""
    import torch.nn.functional as F
    from railroad_env import RailroadGymEnv
    from training.agent_model import VALUE_POOL_HW
    from training.simulator import GameSimulator
    from training.ppo import _decode_action

    torch.set_num_threads(1)
    grab = {}
    h = model.bottleneck.register_forward_hook(lambda m, i, o: grab.__setitem__("b", o))
    X, y, gid = [], [], []
    try:
        for i in range(episodes):
            env = RailroadGymEnv(max_turns=100, opponent_strategy=opponent, seed=seed0 + i,
                                 opponent_kwargs=_opponent_kwargs(opponent_path))
            env.reset()
            sim = GameSimulator.from_env(
                env, allow_skip_disrupt=kwargs.get("policy_channels", 3) == 3,
                force_disrupt=force_disrupt)
            rows = []
            while not sim.is_game_over():
                mask = sim.get_action_mask()
                with torch.no_grad():
                    logits, _ = model(torch.from_numpy(sim.get_encoded_state()).unsqueeze(0),
                                      torch.from_numpy(mask).unsqueeze(0))
                    f = F.adaptive_avg_pool2d(grab["b"], VALUE_POOL_HW).flatten()
                # The last two columns are the WIRE features - current score margin and turn -
                # scaled the way channels 26/27 scale them. They cost nothing at runtime and they
                # are the obvious predictors of who wins; leaving them out was why the first probe
                # found nothing and printed a constant.
                g_ = sim.game_state
                rows.append(np.concatenate([
                    f.numpy(),
                    [(g_.scores[0] - g_.scores[1]) / 10000.0, g_.turn / 100.0],
                ]))
                sim.apply(_decode_action(int(torch.argmax(logits.flatten(1), dim=1).item()),
                                         mask.shape, sim.pad_offsets))
            gs = sim.game_state
            won = 1.0 if gs.scores[0] > gs.scores[1] else 0.0
            X.extend(rows); y.extend([won] * len(rows)); gid.extend([i] * len(rows))
            print(f"  probe game {i + 1}/{episodes}: {gs.scores[0]}:{gs.scores[1]} "
                  f"{'win' if won else 'loss/draw'}", flush=True)
    finally:
        h.remove()
    return np.array(X, dtype=np.float64), np.array(y), np.array(gid)


def fit_probe(X, y, gid, l2_grid=(1.0, 1e2, 1e4, 1e6), iters=800, stride=5, lr=0.5):
    """L2 logistic regression on the pooled bottleneck, selected on HELD-OUT GAMES.

    216 features against one independent outcome per game, so this overfits trivially without a
    penalty, a group-wise split, and a floor. Three things here are not decoration:

    STRATIFIED BY OUTCOME. Splitting games at random gave a 0.49 train base rate against a 0.67
    validation base rate, which makes the held-out Brier unreadable - most of it was the shifted
    base rate, not the model. Wins and losses are now split separately.

    AN INTERCEPT-ONLY CANDIDATE. Large lambda does NOT drive w to zero under gradient descent: the
    shrink and the gradient reach a fixed point at |w| ~ 22, which scored WORSE than the base rate.
    So "predict the training base rate" is an explicit candidate, and the selected probe can never
    be worse than it on held-out games.

    ROW SUBSAMPLING. Consecutive decisions inside a turn are near-identical states, so every 5th
    row loses almost nothing and makes the fit fast enough to re-run from the cached data."""
    games = np.unique(gid)
    won = np.array([y[gid == g][0] for g in games])
    rng = np.random.RandomState(0)
    val_games = set()
    for outcome in (0.0, 1.0):
        grp = games[won == outcome]
        rng.shuffle(grp)
        val_games.update(grp[:max(1, int(0.25 * len(grp)))].tolist())
    va = np.array([g in val_games for g in gid])
    tr = ~va
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, ytr = ((X[tr] - mu) / sd)[::stride], y[tr][::stride]
    Xva, yva = (X[va] - mu) / sd, y[va]

    def brier(w, b):
        return float(np.mean((_sigmoid(Xva @ w + b) - yva) ** 2))

    # candidate 0: intercept only. The floor - a probe that cannot beat this is not a probe.
    base = float(np.clip(ytr.mean(), 1e-6, 1 - 1e-6))
    zero = np.zeros(X.shape[1])
    best = (brier(zero, np.log(base / (1 - base))), "base rate only",
            zero, np.log(base / (1 - base)))

    # WHICH COLUMNS is a hyperparameter, not a given. The last two are the wire features (score
    # margin, turn); the rest are the pooled bottleneck. Fitting all of them under one global
    # penalty let 216 uninformative NN dimensions bury the two that work: measured, margin+turn
    # alone scored 0.1472 against a 0.2481 base rate while the combined fit fell back to the base
    # rate entirely. Selecting the subset on held-out games fixes that and costs nothing.
    n = X.shape[1]
    groups = (("all features", np.arange(n)),
              ("NN features", np.arange(n - 2)),
              ("margin+turn", np.arange(n - 2, n)))
    for gname, cols in groups:
        if len(cols) == 0:
            continue
        Xg, Xvg = Xtr[:, cols], Xva[:, cols]
        for lam in l2_grid:
            w, b = np.zeros(len(cols)), 0.0
            decay = min(0.99, lr * lam / len(ytr))
            for _ in range(iters):
                e = _sigmoid(Xg @ w + b) - ytr
                w = w * (1.0 - decay) - lr * (Xg.T @ e / len(ytr))
                b -= lr * e.mean()
            if not (np.all(np.isfinite(w)) and np.isfinite(b)):
                continue
            sc = float(np.mean((_sigmoid(Xvg @ w + b) - yva) ** 2))
            if sc < best[0]:
                full = np.zeros(n)
                full[cols] = w
                best = (sc, f"{gname}, L2 {lam:g}", full, b)

    sc, chosen, w, b = best
    base_va = float(np.mean((ytr.mean() - yva) ** 2))
    report = {
        "chosen": chosen, "val_brier": sc, "val_base_brier": base_va,
        "val_games": len(val_games), "train_games": len(games) - len(val_games),
        "train_base": float(ytr.mean()), "val_base": float(yva.mean()),
        "val_acc": float(np.mean((_sigmoid(Xva @ w + b) > 0.5) == (yva > 0.5))),
    }
    return w / sd, float(b - (w * mu / sd).sum()), report


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

_PROBE_FNS = '''
def _dbg_apool(x,ph,pw):
    """torch AdaptiveAvgPool2d: window i is [floor(i*n/out), ceil((i+1)*n/out)), so for
    (5,7)->(2,3) the windows OVERLAP. Getting this wrong shifts the readout silently."""
    c,h,w=x.shape
    o=np.empty((c,ph,pw),np.float32)
    for i in range(ph):
        a=(i*h)//ph; b=-((-(i+1)*h)//ph)
        for j in range(pw):
            u=(j*w)//pw; v=-((-(j+1)*w)//pw)
            o[:,i,j]=x[:,a:b,u:v].mean(axis=(1,2))
    return o

def _dbg_value(bo):
    """The pooled bottleneck the value head would read. The probe's logit is built in the main
    loop instead of here, because its last two features are the score margin and the turn, which
    live on the wire rather than in the network."""
    return _dbg_apool(bo,_DBG_VPH,_DBG_VPW).reshape(-1).astype(np.float64)

'''

_MAIN_FWD_OLD = '''    logits=_fwd(observe(track,inst,inked_z,active,conn,pairs,my_score-foe_score,turn))'''
_MAIN_FWD_NEW = '''    logits,_dbg_raw=_fwd(observe(track,inst,inked_z,active,conn,pairs,my_score-foe_score,turn))'''

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
    # In probe mode _dbg_raw is the pooled feature vector and the margin/turn columns are
    # appended here; otherwise it is the value head's scalar and _DBG_A/_DBG_B rescale it.
    if _DBG_LBL=="logit":
        _dbg_val=float(_DBG_PW@np.concatenate([_dbg_raw,
            [(my_score-foe_score)/10000.0,turn/100.0]])+_DBG_PB)
    else:
        _dbg_val=_dbg_raw
    _dbg_p=1.0/(1.0+np.exp(-max(-60.0,min(60.0,_DBG_A*_dbg_val+_DBG_B))))
    # The parentheses are load-bearing: % binds tighter than +, so without them the format
    # applies to the last literal alone and the line dies with "not all arguments converted".
    print(("T%-3d %6d-%-6d (%+d) | "+_DBG_LBL+" %+9.4f | P(win) me %.3f foe %.3f | %s")
          %(turn,my_score,foe_score,my_score-foe_score,_dbg_val,_dbg_p,1.0-_dbg_p,_dbg_cmd),
          file=sys.stderr, flush=True)
    turn+=1'''


def build_debug_source(blob, shapes, quant, policy_channels, passive_income, allow_skip,
                       vph, vpw, cal_a, cal_b, calibrated, probe=None, in_channels=28):
    src = BA.build_source(blob, shapes, quant, policy_channels, passive_income,
                          allow_skip=allow_skip, in_channels=in_channels)

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

    src = sub(src, "def _fwd(g):",
              (_PROBE_FNS if probe is not None else _VALUE_FNS) + "def _fwd(g):",
              "insert the critic readout")
    src = sub(src, "    return _conv(d1,P[\"head\"],1)",
              "    return _conv(d1,P[\"head\"],1),_dbg_value(bo)", "_fwd returns the value")
    src = sub(src, _MAIN_FWD_OLD, _MAIN_FWD_NEW, "unpack the value in the main loop")
    src = sub(src, _PRINT_OLD, _PRINT_NEW, "stderr debug line")
    # Anchored on the numpy import, not on "_Q=" - that substring also appears in every
    # dequantization branch (_Q=="fp16" and friends), so it is not unique.
    header = (f"_DBG_VPH={vph}\n_DBG_VPW={vpw}\n_DBG_A={cal_a!r}\n_DBG_B={cal_b!r}\n"
              f"_DBG_CAL={calibrated!r}\n")
    if probe is not None:
        # float32 through base85, NOT through the quantized blob: 217 numbers cost ~1.1k chars at
        # full precision, so there is nothing to gain by rounding them and a lot to lose - this
        # IS the probability.
        w, b = probe
        header += (f'_DBG_PW=np.frombuffer(base64.b85decode('
                   f'{base64.b85encode(np.asarray(w, np.float32).tobytes())!r}),np.float32)\n'
                   f"_DBG_PB={float(b)!r}\n_DBG_LBL='logit'\n")
    else:
        header += "_DBG_LBL='V'\n"
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
    ap.add_argument("--calibrate-opponent-path", default=None, metavar="BAKE.py",
                    help="point level2Silver at THIS baked .py instead of the frozen one under "
                         "baselines/. Use it when the checkpoint has outgrown the frozen bar: a "
                         "fit needs losses, and at a 0.9+ win rate there are almost none, so both "
                         "--calibrate and --probe collapse toward the base rate however many "
                         "games you play. Bake the previous student and calibrate against that")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="Instead of baking the 27,905-parameter value head, play N games and fit "
                         "a 217-parameter logistic probe on the same pooled bottleneck, against "
                         "actual game OUTCOMES. ~17k characters smaller, which is what gets a "
                         "debug agent under CodinGame's 100k cap, and better calibrated than a "
                         "sigmoid fitted on top of an untrained head. Validated on held-out GAMES")
    ap.add_argument("--probe-cache", default=None, metavar="PATH",
                    help="Save the probe's (features, outcomes) to this .npz and reuse it if it "
                         "already exists. Collecting the data is the expensive part and refitting "
                         "is instant, so cache it once and re-bake freely. Defaults to "
                         "<output>.probe.npz")
    ap.add_argument("--skip-disrupt", dest="skip_disrupt", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="as bake_agent.py: defaults to what the checkpoint recorded")
    args = ap.parse_args()

    blob_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kwargs = blob_ckpt["model_kwargs"]
    sd = blob_ckpt["model_state_dict"]
    print(f"{args.checkpoint}: {kwargs}")
    if "critic_state_dict" in blob_ckpt:
        raise SystemExit("bake_debug_agent: this checkpoint trained a SEPARATE critic, so the "
                         "policy net's own value head is stale and would bake a meaningless P(win). "
                         "Debug-bake its distilled student instead - distill.py labels the student's "
                         "value head from that critic.")

    if args.skip_disrupt is None:
        allow_skip = not blob_ckpt.get("force_disrupt", False)
        source = ("checkpoint" if "force_disrupt" in blob_ckpt
                  else "default (checkpoint predates the field)")
    else:
        allow_skip, source = args.skip_disrupt, "--skip-disrupt flag"
    print(f"  SKIP_DISRUPT in the baked agent: {allow_skip}  [{source}]")

    if args.probe > 0 and args.calibrate > 0:
        raise SystemExit("bake_debug_agent: --probe and --calibrate are alternatives. --probe "
                         "fits the probability directly on outcomes; --calibrate fits a sigmoid "
                         "on top of the value head. Pick one.")

    probe = None
    if args.probe > 0:
        from training.agent_model import RailroadUNet
        model = RailroadUNet(**kwargs)
        model.load_state_dict(sd)
        model.eval()
        cache = Path(args.probe_cache or (args.out + ".probe.npz"))
        if cache.exists():
            d = np.load(cache)
            X, y, gid = d["X"], d["y"], d["gid"]
            print(f"\nreusing {cache} ({len(np.unique(gid))} games, {len(y):,} decisions). "
                  f"Delete it to re-collect.")
        else:
            print(f"\nfitting a win-probability probe on {args.probe} greedy games vs "
                  f"{args.calibrate_opponent}"
                  + (f" [{args.calibrate_opponent_path}]" if args.calibrate_opponent_path else ""))
            X, y, gid = collect_probe_data(
                model, kwargs, args.probe, args.calibrate_opponent,
                force_disrupt=bool(blob_ckpt.get("force_disrupt", False)),
                opponent_path=args.calibrate_opponent_path)
            np.savez_compressed(cache, X=X, y=y, gid=gid)
            print(f"  cached to {cache}")
        if len(np.unique(y)) < 2:
            raise SystemExit("\nevery probe game had the same outcome - there is nothing to fit. "
                             "Raise --probe, or pick an opponent that contests this checkpoint.")
        # Which half is doing the work? NN features alone, wire features alone, or both. This is
        # free once the data is collected and it is the only way to know whether the network is
        # contributing anything at all.
        w, b, rep = fit_probe(X, y, gid)
        probe = (w, b)
        print(f"\n  probe: {X.shape[1] + 1} params, selected {rep['chosen']}, "
              f"{rep['train_games']} train / {rep['val_games']} held-out games "
              f"(base rate {rep['train_base']:.2f} train / {rep['val_base']:.2f} held-out)")
        print(f"  HELD-OUT Brier {rep['val_brier']:.4f}  "
              f"(base rate scores {rep['val_base_brier']:.4f})")
        print(f"  HELD-OUT accuracy at 0.5: {rep['val_acc']:.3f}")
        if rep["val_brier"] > 0.9 * rep["val_base_brier"]:
            print("\n  !! the probe barely beats the base rate on held-out games, so these "
                  "features carry little outcome information at this sample size. Read the "
                  "ORDERING, not the number, and re-fit with more games.")

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
            force_disrupt=bool(blob_ckpt.get("force_disrupt", False)),
            opponent_path=args.calibrate_opponent_path)
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

    # In probe mode the value head is not baked at all - that is the whole size saving.
    baked = BA.fold_batchnorm(sd) + ([] if probe is not None else value_params(sd))
    blob, shapes = BA.quantize(baked, args.quant)
    from training.agent_model import VALUE_POOL_HW
    src = build_debug_source(blob, shapes, args.quant, kwargs["policy_channels"], 3,
                             allow_skip, VALUE_POOL_HW[0], VALUE_POOL_HW[1],
                             cal_a, cal_b, calibrated or probe is not None, probe,
                             in_channels=kwargs["in_channels"])
    Path(args.out).write_text(src)
    n_policy = sum(w.size + b.size for _, w, b in BA.fold_batchnorm(sd))
    n_value = (len(probe[0]) + 1) if probe is not None else sum(
        w.size + b.size for _, w, b in value_params(sd))
    kind = "probe" if probe is not None else "value head"
    print(f"\npolicy params {n_policy:,} + {kind} {n_value:,} = {n_policy + n_value:,}")
    print(f"wrote {args.out}: {len(src):,} chars  [{args.quant}]")
    if len(src) <= BA.CG_SOURCE_LIMIT:
        print(f"  fits CodinGame's {BA.CG_SOURCE_LIMIT:,} char cap "
              f"({100 * len(src) / BA.CG_SOURCE_LIMIT:.0f}% used) - this one CAN be pasted into "
              f"the arena, where stderr shows up in the replay viewer")
    else:
        print(f"  !! {len(src):,} chars is over CodinGame's {BA.CG_SOURCE_LIMIT:,} cap, so this "
              f"file cannot be run in the arena. Use --probe N instead of --calibrate N: it "
              f"replaces the 27,905-param value head with a 217-param probe and saves ~17k chars")
    print("NOT submittable and not meant to be - the critic is dead weight in the arena, and "
          "this file exists to be read, not sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

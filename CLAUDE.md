# CLAUDE.md

Working notes for the CodinGame **Summer Challenge 2026** railway agent. Kept short on purpose.
Claims marked *(notes)* come from earlier sessions and were not re-checked on 2026-09-13 — verify
before relying on them. Everything else was measured or checked in the repo.

## Ground rules

- **Carlo launches real training runs.** Smoke-test a config for 2–3 iterations against a scratch
  `checkpoint_dir`/`log_dir`, stop it, and hand back the command. Baking, evaluating and
  `test_submission.py` on existing artifacts are fine; stop them when done.
- **Pick checkpoints from the eval table, never the newest.** Runs here peak early and decay.
- **Ask before deleting** checkpoints or anything in `baselines/`.
- Repo: private, `github.com/CarloCDT/summer-challenge-2026`, branch `main`. `checkpoints/` is
  untracked, so the checkpoints listed under Housekeeping exist only on this machine.

## Rules that are easy to get wrong

- 100 turns. Each turn: 3 paint points (plains 1 / river 2 / mountain 3) and 1 disruption point.
  Neither carries over.
- A connection pays **1 point per turn per own track on its path** while active. Neutral tracks
  and towns pay nobody.
- The scoring path is the one with the **fewest cells** (BFS, N→E→S→W tie-break). Terrain cost only
  affects what you pay. `AUTOPLACE` uses a cost-weighted search instead.
- Instability 4 inks a region and destroys **both** players' track. Town regions can't be inked.
- Both players placing on the same cell makes it neutral, and both pay.
- The game ends early once no desired connection is reachable.
- Source of truth: the Java referee in `SummerChallenge2026/`. `verify_rules.py` (62 checks) must
  stay green.

## Live pipeline

| step | file |
|---|---|
| environment (faithful port) | `railroad_env/` |
| board → 20×30×38 canvas + action masks | `training/encoding.py`, `railroad_env/features.py` |
| bake runtime vs `encode_state`, every channel, both seats | `verify_bake_parity.py` |
| model | `training/agent_model.py` (`RailroadUNet`) |
| PPO | `training/train_ppo.py`, `training/ppo.py` |
| teacher → student, int4 QAT | `training/distill.py`, `training/quantization.py` |
| checkpoint → submission (≤100,000 chars) | `bake_agent.py` → `submission.py` |
| debug bake that prints P(win) — never submit | `bake_debug_agent.py` |
| play a bake over the wire protocol | `test_submission.py` |
| greedy boss table | `evaluate_checkpoints.py` |
| head-to-head, both seats | `compare_agents.py` |
| behaviour read-out | `diagnose_agent.py` |

```
python3 -u -m training.train_ppo --config training/configs/<config>.yaml
python3 -u -m training.distill --config training/configs/distill.yaml
python3 bake_agent.py checkpoints/<student>.pt -o submission.py --quant int4
python3 test_submission.py submission.py --opponent level2Silver4 --episodes 50
python3 compare_agents.py a=<file.pt|file.py> b=<file.pt|file.py> --episodes 25 --workers 8
```

Legacy, still importable, not used: the AlphaZero/MCTS stack (`training/train.py`, `mcts.py`, …)
and the lite game (`railroad_lite_env.py`, `training/lite*.py`). Don't extend them without asking.

## Observation — 38 channels

| ch | contents | ch | contents |
|---|---|---|---|
| 0–2 | terrain one-hot | 22 | instability / 4 |
| 3, 4 | enemy / own track | 23 | region inked (padding counts as inked) |
| 5, 6 | enemy / own tracks in region, count / 10 | 24 | cell on an active connection |
| 7 | region size / board cells | 25 | town here |
| 8, 9 | enemy / own tracks in region on an active connection, count / 10 | 26 | (own − enemy score) / 10000, flat |
| 10–21 | route guides: −1 this town, −0.5 wanted town, +1 between | 27 | turn / 100, flat |

Channels 28–37 are derived features from `railroad_env/features.py` (added 2026-09-14):

| ch | contents | ch | contents |
|---|---|---|---|
| 28 | active connections through the cell / 10 | 33 | region contains a town (can't be inked) |
| 29, 30 | own / enemy income at stake in the region / 20 | 34 | not-yet-active connections whose cheapest completion crosses the cell / 5 |
| 31 | tanh(score diff / 300), flat | 35 | 1 / cheapest remaining completion cost there |
| 32 | tanh(income-rate diff / 30), flat | 36 | share of desired connections still reachable, flat |
| | | 37 | instability the opponent added to the region / 4 |

Channels are only ever **appended**. `RailroadUNet` slices its input to its own `in_channels`, so
28-channel checkpoints still load and play, and the frozen `level2Silver*` bakes carry their own
28-channel runtime. `train_ppo.py` warm-starts a wider net by giving the new stem inputs zero
weights, which leaves its function unchanged.

**Changing the layout means edits in lockstep:** `railroad_env/features.py` (or
`game_state.py::get_observation`), its copy in `bake_agent.py`'s runtime, `CHANNEL_NAMES` in
`debug_full_agent.ipynb`, and `verify_rules.py::test_derived_features` if a meaning changes. Then run
`python3 verify_bake_parity.py`, which compares the bake's `observe()` with `encode_state` from both
seats.

## Opponents

- `level1` / `passive` waits. `level2` / `boss` autoplaces between two random towns. These are the
  shipped bosses.
- `level1Pro` skips placing half the time and disrupts half the time. `level2Pro` builds the
  cheapest unfinished connection and never disrupts. `level2ProMax` is `level2Pro` plus inking where
  enemy track leads by more than one.
- `level2Silver` … `level2Silver4` are frozen bakes in `baselines/`, run as subprocesses, ~1 s per
  game. Rungs are added, never repointed, so old numbers keep their meaning. They're stateful per
  game and can't be deep-copied mid-game. See `baselines/README.md`.

## What we've learned

Measured in this project, most recent first:

- **2026-09-14 diagnosis of the plateau**, all on `065817_iter200` (the scripts weren't kept):
  - The value loss steered the shared encoder. Against `level2Silver4` its gradient norm was ~32
    against ~7 for the policy loss, near-orthogonal, and `max_grad_norm` 0.5 clipped every
    minibatch. The entropy term's gradient was ~0.01, which is why changing `entropy_coef` did
    nothing. Arms B–D target this (`value_scale`, a separate critic).
  - The critic can't call a contested game early: corr(V(turn 0), outcome) +0.06, mid-game +0.66.
    The per-game sum of margin rewards has sd 0.62, against the ±5 win bonus. Arm D's shaping
    targets this.
  - Against a near-mirror (`level2Silver4`), 24% of our placements collide into neutral track (3%
    against `level2Pro`), 42% of our own track gets inked, and games end at turn ~63 with ~480
    points each.
  - Channel 26 reads ~0.005 in contested games (margins ±100 over a 10,000 divisor), hence channel 31.
  - A disrupt is spread over its region's cells, which adds ~0.64 nats of meaningless entropy to
    disrupt decisions. Read `train/entropy` with that in mind.
  - CodinGame's 100k limit counts UTF-16 code units, so a CJK-range payload carries 15 bits per
    character against base85's 6.4. The current student would be ~44k chars at int4 or ~82k at
    int8. Not built.

- **Long PPO runs from the current lineage decay.** `065817` (200 iterations, 2 epochs, fast lr
  anneal) is the only recent run that improved. The 500-iteration, 3-epoch attempts (`130019`,
  `164104`) pushed entropy past 2.0 and lost ground on every eval column. Halving the entropy bonus
  did not slow the climb.
- **Healthy optimiser metrics don't mean learning.** `074038`, `222648` and `130019` all kept
  clip fraction, KL and explained variance in range while specialising, stalling and decaying.
  Only eval tables show progress, so keep a held-out rung in `eval_bosses`.
- **Against contested opponents the reward is roughly one bit per game.** The per-turn margin is
  about 1 point (0.001 after `score_norm` 1000), against ~160 per turn versus `level2Pro`. The ±5.0
  win bonus dominates the return. This is the likeliest cause of slow, noisy learning. Candidate
  fix, not built: potential-based shaping on projected remaining income.
- **γ 0.998 / λ 0.99 hurt** (`222648`): advantage std up 46%, explained variance 0.87 → 0.77. Keep
  0.997 / 0.98.
- **Don't anneal lr to zero.** Use `lr_end` ≈ 20% of peak. 2.5e-5 peak was too cold.
- **Self-play is worth its compute under `win_bonus`.** Mean |advantage| per decision was 1.196 for
  `self`, second only to `level2Pro`. `self` mirrors the live weights, so it never ratchets.
- **Distillation did not lose the contested edge** (2026-09-14, 128 paired seeds against
  `level2Silver3`): teacher fp32 0.62, its int4 bake 0.63, an int8 re-bake of the same student 0.66,
  all within one SE (±0.043). The earlier 0.73 vs 0.62 was noise, and so is the old "int4 costs
  ~0.07" note below as far as this matchup shows. `distill.yaml` still runs only 50 iterations.
- **Value distillation works.** `value_loss_weight` 0.25 with `value_huber` 2.0 gives the student a
  critic correlating ~0.975 with the teacher, at no policy cost and no submission size.
- **Win-probability fits need losses.** Against an opponent won ≥0.9, the fit collapses to the base
  rate. Calibrate against a near-even rung.
- **`action_selection: onpolicy` rollouts aren't reproducible from a seed** (global torch RNG). Use
  `epsilon_greedy` with `epsilon: 0` for A/B tests.
- **Two identical agents score 0:0** (every contested cell goes neutral). Use it to check a rung
  points at the right file.

*(notes)*, not re-checked:

- Our eval table doesn't predict arena rank. Only the `114307_iter200` lineage (rank ~288) has
  arena evidence.
- BatchNorm must stay in eval mode during the PPO update. The pin is in `ppo.py::ppo_update`
  (checked); the original finding, that train-mode BatchNorm broke the PPO ratio, is from notes.
- Masked-off actions get zero gradient. `force_disrupt` is recorded in checkpoints and honoured by
  eval and bake.
- int4 costs ~0.07 win rate against fp32 head-to-head; `quant_aware: int4` targets that. Judge
  quantization by match results, not move agreement.
- The bake must answer in 50 ms per turn and runs ~3 ms median. Keep its stderr small, since the
  referee may not drain the pipe.
- Rollout is ~93% of iteration time, mostly `GreedyAutoplaceOpponent` re-pricing (no cache yet).

## Current state — 2026-09-14

- **Best agent:** teacher `checkpoints/20260913-065817_iter200.pt`. Its student
  `20260913-113208-distill_iter50.pt` is baked as `submission.py` (87,420 chars) and frozen as
  `level2Silver4`. The arena bot is close to Gold (Carlo, 2026-09-14).
- **`ppo_28ch_vs_silver3_1k.yaml` ran** as `20260913-181145`. Entropy stayed down (1.6 → 1.45) and
  nothing improved: ~0.53 against `level2Silver4`, ~0.65 against `level2Silver3`. The plateau isn't
  the schedule; see the 2026-09-14 diagnosis under "What we've learned".
- **Ready, not launched:** four 38-channel arms warm-started from `065817_iter200`,
  `training/configs/ppo_38ch_{a_control,b_valuefix,c_critic,d_critic_shaping}.yaml`. They share the
  opponent pool (the 065817 recipe, one rung up) and differ only in the block marked ARM.
- The contest ends **2026-09-21**. Arena results aren't recorded in the repo; add them here when known.

## Next steps

1. Run arm D (separate critic + shaping) and arm A (control) first, then B and C to attribute a gain.
   Compare on the 200-episode eval table, never on `self_play/win_rate`.
2. Distil the best arm (`distill.py` reads value labels from a separate critic automatically), bake,
   `verify_bake_parity.py` on the student, `test_submission.py`, submit.
3. If no arm beats the control: a pending-placements channel (one forward per placement), or a
   snapshot league of the run's own past checkpoints.

## Housekeeping

- **Keep:** `checkpoints/20260910-224257_iter200.pt`, `20260911-114307_iter200.pt`,
  `20260912-155239_iter500.pt`, `20260913-065817_iter200.pt`, and everything in `baselines/`.
- `checkpoints/` holds 184 files, 3.4 GB. 88 are 25-channel and won't load; check
  `model_kwargs["in_channels"]` first.
- `ppo_finetune_student.yaml` has a `REPLACE_ME` path and targets a 25-channel student. Unusable.
- `/home/carlo/claude_env` is a standalone copy of the environment for others to train against.

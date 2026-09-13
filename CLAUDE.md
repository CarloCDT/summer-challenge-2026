# CLAUDE.md

Context for the CodinGame **Summer Challenge 2026** railway agent. Read this before touching
anything; several things here are counterintuitive and were established by measurement.

## Ground rules

**Carlo launches all real training runs.** Verify a config with a handful of iterations against a
scratch `checkpoint_dir`/`log_dir`, kill it, and hand back the command. Never leave a full run
going. Baking, evaluating existing checkpoints and `test_submission.py` are analysis of artifacts
that already exist and are fine to run — but say so, and stop them when done.

**Pick checkpoints from the eval table, never by recency.** Every run here oscillates hard. The
`090433` PPO run scored +1889 on level2 at iteration 2000 and collapsed to −59 at iteration 2100,
its final checkpoint. Read `eval/<boss>/margin` out of `runs/<timestamp>/` before choosing.

## The game

Two players, 100 turns, maps 14–20 rows with width at a 1.5 aspect ratio (so up to 20×30). Each
turn: **3 paint points** (track costs 1 plains / 2 river / 3 mountain) and **1 disruption point**.

- A connection pays **1 point per turn per own track on the path**, for as long as it is active.
- The scoring path is the one with the **fewest cells** (`TrainBFS`, plain breadth-first, north-first
  tie-break). Terrain cost only affects what you *pay*, never which route the trains take.
- `AUTOPLACE` uses a *cost-weighted* search instead. The two optimise different things.
- Disrupting adds 1 instability; at **4** the region is inked, destroying **both** players' track
  in it permanently. Regions containing a town can **never** be inked.
- Towns are passable like track but owned by nobody, so they score for nobody, and you cannot
  build on them.

`SummerChallenge2026/` holds the original Java referee and the Python bosses. It is the source of
truth — check it rather than guessing. `verify_rules.py` asserts our port against it; keep it green.

## Layout

| path | role |
|---|---|
| `railroad_env/` | the environment, a faithful port of the Java referee |
| `training/encoding.py` | board → fixed `(28, 20, 30)` canvas + action masks |
| `training/agent_model.py` | `RailroadUNet`, the live model |
| `training/train_ppo.py`, `training/ppo.py` | the live training path |
| `training/distill.py` | teacher → small student, for the size cap |
| `training/quantization.py` | fake-quant for `quant_aware` — train the student as an int4 net |
| `training/self_play_opponent.py` | drives a torch model as player 1 (the `self` pool entry) |
| `railroad_env/wire.py` | the referee's stdin/stdout protocol, shared by `test_submission.py` and the `level2Silver` boss |
| `bake_agent.py` | checkpoint → self-contained `submission.py` |
| `baselines/` | bakes we actually submitted, frozen byte-for-byte — the `level2Silver` boss reads one; see its README |
| `bake_debug_agent.py` | `bake_agent.py` **plus the value head** — prints V and a fitted P(win) to stderr each turn. Debugging only, never submitted |
| `test_submission.py` | runs a baked file over the referee's wire protocol |
| `evaluate_checkpoints.py` | greedy boss table on fixed seeds |
| `compare_agents.py` | head-to-head round robin, `.pt` or baked `.py`, both seats — **the instrument that separates candidates the boss table cannot** |
| `diagnose_agent.py` | behavioural read-out: disrupt-vs-skip rate, peak-vs-end connections |

This tree is a **private git repo**: `github.com/CarloCDT/summer-challenge-2026`, branch `main`.
Tracked: the code, the configs, the notebooks, `submission.py`, `baselines/`, and `runs/` (the
tensorboard histories every measurement below cites). **Untracked: `checkpoints/`** — 2.0 GB, and
88 of the 123 are stale 25-channel weights. So the two irreplaceable checkpoints named at the
bottom of this file exist on this machine only and are not backed up by the repo.

**Legacy but working, not on the live path:** the AlphaZero/MCTS stack (`training/train.py`,
`mcts.py`, `self_play.py`, `replay_buffer.py`, `model.py`, configs `default/debug/strong/
mcts_random/connect_towns_topk`) and the lite game (`railroad_lite_env.py`, `training/lite*.py`,
`ppo_lite.yaml`). All import cleanly. Don't extend them without asking.

## Observation: 28 channels

| ch | contents |
|---|---|
| 0–2 | terrain one-hot |
| 3, 4 | enemy / own track present |
| 5, 6 | enemy / own tracks in region, **count / 10** |
| 7 | region size / board cells |
| 8, 9 | enemy / own tracks in region **on an active connection**, count / 10 |
| 10–21 | route guides: `-1` this town, `-0.5` a town it wants, `+1` cells between |
| 22 | instability / 4 |
| 23 | region inked (off-board padding is marked inked) |
| 24 | cell is on an active connection |
| 25 | a town stands here |
| 26 | (own − enemy score) / 10000, flat plane |
| 27 | turn / 100, flat plane |

Channels 5/6/8/9 are **counts, not densities**, and this was a deliberate reversal. A disruptor
thresholds on an *absolute* lead (enemy − own ≥ 2), so a density hid the deciding quantity behind
a product of three channels. Channels 26/27 are flat because a convolution is local.

**If you change the observation you MUST update three places in lockstep:**
`railroad_env/game_state.py` (`get_observation`), the `observe()` in `bake_agent.py`'s `RUNTIME`,
and `CHANNEL_NAMES` in `debug_full_agent.ipynb`. Then prove parity: bake a file, and compare
`encode_state` against the runtime's `observe` fed through `test_submission.init_lines`/
`frame_lines` over a few hundred observations. A silent mismatch means the submitted agent plays
a different game than it trained on.

## Bosses

`BOSS_TIERS = level1, level1Pro, level2, level2Pro, level2ProMax, level2Silver, level2Silver2,
level2Silver3, level2Silver4`

- `level1` — always waits (the shipped League 1 boss).
- `level2` — AUTOPLACE between **two random towns** every turn (the shipped League 2 boss).
- `level2Pro` — prices every unfinished connection, commits to the cheapest. Never disrupts.
- `level2ProMax` — ours. `level2Pro` plus: inks the region where enemy track most outnumbers its
  own, when that lead is **> 1**. Ties break to the lowest zone id, so it is deterministic.
- `level2Silver` — **a bake we actually submitted**, run as a subprocess over the referee's wire
  protocol (`railroad_env/wire.py`, shared with `test_submission.py`). Deterministic, ~1.0 s per
  100-turn game — on par with level2Pro. It reads a **frozen** copy under `baselines/`, currently
  the one that reached CodinGame rank ~288, and **not** the live `submission.py`.

- `level2Silver2` — **the second frozen rung, added 2026-09-12.** Same machinery, different frozen
  file: `baselines/20260912-124725-distill_iter50.py`, the bake of the first student distilled with
  `value_loss_weight` off the `win_bonus` teacher `20260912-074038_iter350`. Added as a new tier
  rather than by repointing `level2Silver`, which would have retired the rank-288 bake and silently
  changed the meaning of every number ever measured against that tier. **It carries no arena
  evidence** — it was never submitted, so it is a harder local bar, not a validated one. Error
  messages now read `TIER_NAME` off the class, so a failure names the rung that failed.

  Measured against each other, 10 games per seat on the same seeds: the 2026-09-12 bake beats the
  rank-288 bake **+859, win 0.90**, and the reverse seat gives exactly **−859, win 0.10**. Note
  both rungs descend from the lineage everything here now trains on, so this is a fixed bar and
  **not** a held-out generalisation measure.

- `level2Silver3` — **the third rung and the current top of the league, added 2026-09-12.**
  `baselines/20260912-210537-distill_iter50.py`, the bake of the student distilled from teacher
  `20260912-155239_iter500` with `value_loss_weight` (critic correlation 0.975). Submitted; no
  arena result yet.

  **It is the most contested opponent this repo owns**, and that is its real value. 10 games each:

  | seat 0 | opponent | margin | win |
  |---|---|---|---|
  | rung 3 | `level2Silver2` | **+103** | **0.70** |
  | rung 3 | `level2Silver` | +750 | 0.90 |
  | rung 2 | `level2Silver` | +859 | 0.90 |

  Everything else in the league saturates at 0.90+, so rung 3 against rung 2 is what to use
  wherever losses are needed — the win-probability fits above especially. Note the ladder is **not
  a rating**: rung 3 beats rung 2, yet beats rung 1 by *less* than rung 2 does. Non-transitivity,
  as warned at the top. Note also the scale — those games score 960 to 857, against ~23,000 versus
  a scripted builder. Two strong disruptors ink each other out, and that mutual-destruction regime
  is most of what the league measures.

- `level2Silver4` — **the fourth rung, added 2026-09-13, the current best candidate.**
  `baselines/20260913-113208-distill_iter50.py`, the bake of the student distilled from teacher
  `20260913-065817_iter200`. That teacher is the warm start of `ppo_28ch_vs_silver3_1k.yaml`, which
  now trains against **this rung alone** — no `opponent_pool`, no self-play (decided 2026-09-13).
  Consequences worth knowing: the run plays a frozen near-mirror of its own starting weights; the
  `self_play/*` scalars and their early warning are gone; and with one training opponent,
  specialisation is the likeliest failure, so every other rung is held out in `eval_bosses`.

`level2Silver` is deliberately the *file*, not the checkpoint behind it. Loading that checkpoint
into torch would be a different player: the bake is int4 with BatchNorm folded and reproduces only
~62% of the torch model's argmax moves. Verified exact — the submission scores identically playing
seat 0 through `test_submission.py` and seat 1 as `level2Silver` on the same board (5 seeds,
23331/20845/21921/11472/23157 both ways).

**It used to read `submission.py`, and that was a mistake worth understanding (changed 2026-09-11).**
The boss was live state: re-baking silently changed the training opponent *and* the eval baseline of
anything then running, so `eval/level2Silver/margin` was not comparable across a re-bake and every
number had to be annotated with which bake produced it. `BakedSubmissionOpponent.DEFAULT_PATH` now
points at `baselines/20260911-144318-distill_iter50_rank288.py`, a byte-for-byte copy of that
submission, which nothing overwrites. Verified by removing `submission.py` from the tree entirely
and playing a full game against `level2Silver`: it plays, unchanged. Re-baking cannot move this
tier any more.

Consequences of the frozen choice. The bar **no longer rises on its own** — beating `level2Silver`
once means beating it forever, and raising it is now a deliberate one-line edit (copy the new bake
into `baselines/`, repoint `DEFAULT_PATH`), which is the right amount of friction because it moves
every number this tier has ever produced. Record which baseline a table was measured against. The
old live behaviour is still available per instance for a one-off "am I beating what we would send
right now" check: `opponent_kwargs={'submission_path': 'submission.py'}`.

It remains **stateful per game** (its own turn counter and route cache), so it is spawned lazily,
closed on the turn that reaches `max_turns`, and refuses to be deep-copied mid-game — it cannot be
used with `GameSimulator.clone()`/MCTS. Having it in `BOSS_TIERS` also lengthens any eval that does
not pin `eval_bosses` by roughly one second per episode.

`level2ProMax` finishes building around turn 25, then spends 75 turns purely inking. Only ~14%
of the board (buildable cells in town regions) is permanently safe. It was long treated as a
cliff — an early teacher scored **+8998 on level2Pro and −5508 on level2ProMax** (win 0.05,
connections completed falling from 100% to 42%) — but **that cliff has since been climbed**; see
"Beating level2ProMax" below.

## Results and hard-won facts

**READ THIS FIRST: our eval table does not predict CodinGame rank.** Measured 2026-09-11. A
student that finished **last** in our own 250-game round robin (`prev_shipped`, 0.290 overall)
ranks **higher** in the arena than the successor that beats it 0.68–0.72 head-to-head. Every
number below is a number against *our* opponents, and the arena is not made of them.

Three things are going on, and all three are worth carrying:

- **Non-transitivity.** A beats B beats C beats A is normal in this game. Beating your own
  predecessor is not evidence of ranking better against a field of strangers — and when
  `level2Silver` is 50% of the training pool, "beat the predecessor" is literally the objective
  being optimised.
- **The eval set and the training pool are the same opponents**, so overfitting to the pool moves
  every eval number *up*. The table structurally cannot detect the failure. Opponents held out of
  training are the only honest generalisation measure, and we do not currently hold any out.
- **We optimise margin; the arena ranks on wins.** These come apart — in the round robin, 3 of 10
  pairings had margin and win rate pointing in opposite directions (`student_full` loses the
  teacher by −238 on score while winning 0.52 of matches).

**Rollout games are not reproducible from a seed under `action_selection: onpolicy`.** The
per-game seed fixes the map and the opponent, but the action itself comes from
`dist.sample()`, which draws on the **global torch RNG** — so replaying one seed twice gives
different games and different lengths. Found 2026-09-11 by an A/B test that silently compared two
different games. Use `action_selection: epsilon_greedy` with `epsilon: 0.0` (pure argmax) when you
need two rollouts to be comparable.

**Head-to-head is the sharper instrument.** `scratchpad/roundrobin.py` plays every agent against
every other, both seats, and separates checkpoints that a fixed-boss table shows as identical.
Two agents with the same weights played against each other score 0:0 on every seed — they pick
the same cell each turn and every track resolves to NEUTRAL, which scores for nobody. That is a
useful sanity check, not a bug.

**Beating level2ProMax (2026-09-11).** `runs/20260910-224257` (`ppo_28ch_level2ProMax.yaml`, warm
started from the level2Pro run's `_iter400`) peaked at **iteration 200** and beat its own starting
point on all five bosses at once:

| checkpoint | level1 | level1Pro | level2 | level2Pro | level2ProMax |
|---|---|---|---|---|---|
| warm start `184004_iter400` | +12899 | +9736 | +10174 | +9781 | −2205 |
| **`224257_iter200`** | +15994 | +13380 | +13581 | +11945 | **+2367** (win 0.94) |
| `224257_iter800` | +6063 | +4895 | +6243 | +6271 | −544 |

**The peak is at iteration 200 of 1000, and the run destroyed it over the following 600.** This is
the sharpest instance of the oscillation the ground rules warn about, so it is worth knowing how
the decay looked: own score stayed flat while the *opponent's doubled* (level2 opp 6547 → 13051,
level2Pro 8223 → 13902) as `conn` climbed 0.51 → ~0.99. The policy drifted into completing every
connection available, including routes whose path is mostly the opponent's track — which pays them
as much as you. `reward_mode: margin` is supposed to punish exactly that and is correctly wired
(`ppo.py:93` → `simulator.py:63`); it did not prevent the drift.

Two diagnostics called it while the run was still going, and both are worth checking early:

- `self_play/score_margin` went **+416 (iter80) → −1185 (iter800)** and `self_play/win_rate`
  **0.53 → 0.08**. PPO was making its *own training objective* worse — not overfitting to the
  opponent, simply not optimising. When the self-play margin turns over, stop the run.
- `train/clip_frac` sat at **0.42–0.55 for all 800 iterations** (healthy PPO is 0.05–0.2), as it
  did at 0.40–0.59 on the run before it. **Root cause found and fixed — see "BatchNorm broke
  every PPO run" below.** It was never a step-size problem.

**BatchNorm broke every PPO run (found 2026-09-11).** Rollout runs under `model.eval()`, so
`old_log_probs` come from BatchNorm's *running* statistics, but `ppo_update` called
`model.train()`, which switches BatchNorm to *per-minibatch* statistics. BatchNorm sits in the
conv trunk (`agent_model.py` `ConvBlock`, `stem`), so this changes the policy logits directly.
Measured on `224257_iter200`, identical weights, **zero gradient steps**:

| update mode | median ratio | outside the 0.8–1.2 clip range |
|---|---|---|
| `model.eval()` | 1.0000 | **0%** |
| `model.train()` | 0.431 | **86.3%** |

PPO's premise — that `π_old` generated the data — was violated *before the first optimiser step*,
so the surrogate was mostly clipping noise and the runs random-walked. This is the mechanism
behind the oscillation the ground rules warn about. Two diagnostics that identify it: `clip_frac`
high **and insensitive to `lr`** (cutting 1e-4 → 1e-5 barely moved it), and a ratio that is not
1.0 when recomputed on the rollout batch with unchanged weights.

Fixed in `ppo.py::ppo_update` by pinning every `_BatchNorm` module to eval during the update.
`clip_frac` then falls from 0.43 to a **0.17–0.22 plateau** over ~10 iterations and `approx_kl`
from 0.081 to ~0.017. (Measured on the real run at constant lr. Do not trust a short smoke test
for this: `decay_progress = iteration / (num_iterations - 1)`, so a 5-iteration run anneals lr to
exactly 0 on its last iteration and prints a flattering `clip_frac` of 0.00.) Dropout is left
alive deliberately: it sits only in the value head and never touches the policy logits. Note this
also matches deployment — `bake_agent.py` folds BatchNorm out of `running_mean`/`running_var`, so
the **eval-mode network is the submitted network**, and training in train mode was optimising a
function that was neither the one collecting the data nor the one that ships.

**A masked action channel gets exactly zero gradient — so whatever still offers it runs on frozen
weights.** `force_disrupt` keeps the 3-channel head (so a 3-channel checkpoint still warm-starts)
but withholds `SKIP_DISRUPT` from every mask. `masked_fill` blocks the gradient completely
(measured: head channel 2 grad norm **0.000** vs 15.7 and 10.1 for PLACE/DISRUPT), so those
weights stay frozen at the warm start while the features feeding them drift for the whole run.
Anything downstream that still offers the action is then deciding on stale logits — which is the
exact failure `force_disrupt` exists to remove, since a declined disrupt destroys a resource that
is 1/turn and never carries over. Both offenders are now wired to the flag: checkpoints record
`force_disrupt`, `evaluate.py::play_eval_game` takes it (otherwise the table you *pick checkpoints
from* scores untrained behaviour), and `bake_agent.py` reads it to drop the skip branch, with
`--no-skip-disrupt` for checkpoints written before the field existed. **Any future masked-off
action needs the same three-way treatment.**

**GAE's real-reward horizon is `1/(1-γλ)`, not `1/(1-γ)`.** At γ=0.997/λ=0.95 that is ~19
decisions ≈ 6 turns (the old γ=0.99 gave ~17 ≈ 5.7), so raising γ bought **0.7 turns** of direct
credit — everything beyond that is delegated to the critic. Measured on `224257_iter200` the
critic's explained variance is **0.44** pooled (0.36–0.54 per opponent, so the opponent mixture is
not the cause; healthy PPO is 0.8–0.95, and some of the shortfall is irreducible given a
stochastic policy and opponent). `train/explained_variance` is now logged — it bounds how much
long-horizon credit assignment is actually working, which is exactly what disruption depends on.

Contributing factor: `max_turns: 50` against `eval_max_turns: 100`. level2ProMax inks from ~turn
25, so a 50-turn episode contains ~25 turns of inking and the 100-turn eval contains ~75. Training
under-prices a large inkable footprint by ~3x while the payoff for completing a connection is
immediate — building more is rational in the trained game and wrong in the scored one.

**Distillation held the result.** `runs/20260911-063919-distill` distilled `224257_iter200` into
the `(34,34,36)` student and kept most of it, and — unlike the PPO run — it is *stable* across
every eval: level2ProMax **+1642 / +2010 / +1440 / +1804** (win 0.82–0.86) at iters 25/50/75/100,
with level1 +16608, level1Pro +13724, level2 +13498, level2Pro +12170 at iter100. Note the student
agrees with the teacher on only **65%** of moves — which, per the int4 finding below, is the
expected and unalarming shape. `iter100` is the best row on 4 of 5 bosses; `submission.py` was
baked from `iter25`, so a re-bake from `iter100` is the obvious cheap win.

**Distillation is no longer the bottleneck, and the student is NOT capacity-limited.** Measured
2026-09-11 in the same round robin: the `(34,34,36)` student at full precision scores **0.570**
against its 64-channel teacher's **0.600**, and beats it **0.52** head-to-head. An earlier note
here guessed the student might be too small — it is not, so a bigger net buys little and the
12.9k characters of headroom under the cap are not the lever they look like. The remaining loss
between "what we trained" and "what we ship" is **quantization**, not distillation and not
parameter count.

**The opponent is the syllabus.** Distillation only ever teaches the student on states it
actually visits, so `distill.py`'s rollout opponent decides the curriculum. Rolling out against
`level2` alone drills the student on positions from games won 1.00 by +12,819 and never shows it
the inked endgames that decide real matches — the teacher's competence there cannot transfer if
those positions are absent from the data. `distill.yaml` now takes an `opponent_pool` mirroring
the mix the agent will face. Side effect worth compensating for: harder opponents ink the board
out and end games earlier, which cut decisions per game from ~852 to ~273, so
`games_per_iteration` buys ~3x less data than it used to.

**int4 quantization costs no score, despite looking catastrophic.** Over 40 real games the baked
student scored +2212 at int4 versus +2098 at int8, while agreeing with the torch model on only
62% of moves versus 95%. The policy picks one of ~1800 cells and most disagreements are near-ties.
**Judge quantization by `test_submission.py` margin, never by `--verify` agreement.** Group-wise
int4 scales were tried to "fix" the agreement and rejected: no better score, fewer matching moves,
9k more characters. The loss is 4-bit resolution itself, not outlier-skewed scales.

**Refined 2026-09-11: int4 does cost a little, and head-to-head is how you see it.** Scoring
against a fixed boss is too blunt. A 250-game round robin of the *same* student at three
precisions (`scratchpad/roundrobin.py`, seats alternated, 25 games per pairing) gave overall win
rates **fp32 0.570 → int8 0.540 → int4 0.500**, monotone and consistent in every direct pairing.
No single gap clears significance (fp32→int4 is 0.070 against a 0.05 SE); it is the ordering that
carries the weight. Practical conclusion is unchanged — int8 at `(34,34,36)` is 177k chars, 1.8x
over the cap — but the gap is real and is what `quant_aware` below exists to close.

**Shrinking the net to afford int8 is a bad trade, quantified.** Fitting int8 under the cap means
`(23,23,25)` = 66,237 params against int4's `(34,34,36)` = 139,035: a 52% parameter cut to buy
~0.04 win rate. For scale, the *same architecture* trained differently spans 0.29 → 0.50 in the
same round robin. Training quality is worth roughly 5x what quantization is.

**Quantization-aware training (`training/quantization.py`, `quant_aware: int4`).** The student is
graded at full precision but ships at int4, and that round trip was the largest remaining loss in
the pipeline. QAT runs every conv forward on per-output-channel rounded weights while gradients
update the full-precision ones (straight-through), so the student learns weights that survive
rounding. **The non-obvious part: `bake_agent.py` quantizes the BatchNorm-FOLDED weight, but
folding is a per-output-channel scale and this quantizer picks its scale per output channel, so
the two commute** — quantize-then-fold equals fold-then-quantize. Verified against
`fold_batchnorm` on a real checkpoint: worst relative difference 4.4e-8 on BatchNorm layers,
exactly 0.0 on the three without. That is why fake-quantizing the raw conv weight is exact and no
fold plumbing is needed during training. Checkpoints store full-precision weights (bake applies
the identical quantization); rollout and eval use the rounded ones, so both reflect what ships.

**Size.** CodinGame caps source at 100,000 characters. Bake at int4 (the default). Sizes are
measured by baking an *untrained* net, which is the worst case since random weights are
incompressible; trained weights zlib to ~0.75 of that. At 28 channels, `(34,34,36)` bakes to
~101k worst case (marginally over) but a trained one lands near 80k. `(34,34,34)` restores the
guarantee. About 2,000 characters of the runtime are comments, which strip out if you need room.
Measured: the shipped 28-channel `(34,34,36)` student bakes to **87,060 characters**, ~13k under
the cap — the worst-case estimate holds, but this is the layout with the least headroom.

**Reading the critic (`bake_debug_agent.py`, 2026-09-11).** The shipped bake drops the value
head, so `submission.py` can say what it did but never what it thought. This script bakes the
critic back in and prints one stderr line per turn: score, V, a fitted P(win), and the commands.
Three things it established.

**Quantization is a choice between the two heads.** Baked value against torch value on the same
state, `114307_iter300`, 30 turns: fp16 is **0.20% max / 0.03% mean** relative error, int4 is
**42.8% max / 5.5% mean**. 4-bit weights wreck a scalar regression in a way they demonstrably do
not wreck an argmax over ~1800 cells. So this tool defaults to **fp16** (trust V, moves differ
slightly from what ships) and int4 is for debugging the policy (exact shipped moves, V is fiction).

**P(win) is fitted, not read off the net.** The value head is unbounded and regresses discounted
return, so the script plays `--calibrate N` games and fits `sigmoid(a*V + b)`. The effective
sample size is **N, not N × decisions** — one outcome per game. And it is only valid against the
opponent it was fitted on. First fit, 24 games vs `level2Silver`: Brier 0.0788 against a base-rate
0.0829, with a 0.909 base win rate, so it is barely better than guessing the base rate and the
printed probabilities sit near 1.0 even at 0:0. The script now warns on a one-sided or
uninformative fit. **A checkpoint trained with `win_bonus` carries the outcome in its return
directly, so re-fit after that run and compare Brier — that is the cheap test of whether the bonus
did what it was added for.**

**Which checkpoint to point it at.** For reading the critic, a PPO teacher at fp16. For the
student (`144318-distill_iter50`, what `submission.py` was baked from) use `--quant int4`: it is
quantization-aware at int4, so int4 is its native precision and the debug bake then picks
**identical commands to `submission.py`** — verified over all 60 turns of a game — which makes it
an exact stand-in for the shipped agent. Its V is a random readout though (see the distillation
item in the backlog), so calibrate before believing any number: uncalibrated, `sigmoid(V)` sat at
0.50 for a whole game even at +2765. Calibrated on 24 games vs `level2Silver` (base rate 0.72,
Brier 0.184 against 0.202) it at least moves, 0.60 to 0.87 across a game.

**Getting a debug agent into the ARENA (`--probe`, 2026-09-11).** The value head is 27,905
parameters and pushes a student bake to **~108k characters**, over CodinGame's 100,000 cap and so
unusable in the one place worth debugging — the arena is the only opponent pool we do not own, and
our eval table demonstrably does not predict rank there. `--probe N` drops the value head and fits
a **217-parameter** logistic probe on the same pooled bottleneck against real game outcomes:
**90,861 characters**, and the policy stays byte-identical to `submission.py` (184 of 184 turns
over two full games). stderr shows up in CodinGame's replay viewer.

**The student's NN features predict the winner no better than a coin, and the score margin does
it well.** The probe's columns are [216 pooled NN dims | score margin | turn] and the subset is
selected on held-out games. 100 games vs `level2Silver`:

| features | held-out Brier | note |
|---|---|---|
| base rate only | 0.2481 | the floor |
| NN features only | 0.2481 | loses to the floor, rejected |
| **margin + turn** | **0.1472** | selected, accuracy 0.774 |

So the printed probability is driven by the score margin and the turn, **not by the network**. That
is consistent with everything else about this checkpoint: distilled on policy logits alone, value
head never given a gradient. Note also that fitting all 218 columns together is *worse* than
fitting two of them — 216 uninformative dimensions bury the pair that work and the combined fit
collapsed to the base rate — which is why the column subset is a selected hyperparameter and not an
assumption. Four guards were added because the first four attempts each produced a confident wrong
number: the split is **by game and stratified by outcome** (a random split gave a 0.49 train
against 0.67 held-out base rate, which swamped the score), **"predict the base rate" is an explicit
candidate**, **diverged fits are discarded**, and **the column subset is selected**. Features cache
to `<out>.probe.npz`, so re-fitting is free.

**The cheap test of whether `win_bonus` worked** is to re-fit this probe on a checkpoint from the
1k run. That reward puts the match outcome directly into the return, so if its features still
cannot beat the base rate, the bonus did not teach the network anything about winning.

**That test is currently BLOCKED, and the blocker is the frozen bar, not the fit (2026-09-12).**
Ran on `20260912-124725-distill_iter50` — the first student distilled with `value_loss_weight`,
off the win_bonus teacher `20260912-074038_iter350`. Its critic is real (value correlation with
the teacher **0.973**, argmax agreement 0.527 against the shipped student's 0.494), but **both**
readouts came back starved because that student beats `level2Silver` **0.91**:

| fit | games | base rate | Brier | base-rate Brier |
|---|---|---|---|---|
| `--calibrate` on the value head | 60 | 0.913 | 0.0727 | 0.0796 |
| `--probe`, selected margin+turn | 100 | 0.96 held-out | 0.0396 | 0.0433 |

A probability fit learns from LOSSES, and at a 0.9+ win rate there are almost none — the probe's
24 held-out games contained about one. So the probe again picking margin+turn over the NN columns
is **not** evidence the features are uninformative this time; the selection had no power to tell.
Both runs printed their one-sided warning, which is the guard working.

Nothing in the pool contests this student any more (`level2ProMax` 0.99, `level2Silver` 0.91), so
this is the "the bar no longer rises on its own" consequence of freezing `level2Silver`, arriving
in practice. `bake_debug_agent.py --calibrate-opponent-path BAKE.py` now points `level2Silver` at
any baked file for the fit only, without touching `baselines/` or moving the tier for anything
else. The obvious contested opponent is a bake of the **teacher**, which is stronger than its own
student. Redo the win_bonus test that way before concluding anything about the features.

**Do not enlarge that stderr line.** A referee leaves stderr on a pipe it may not drain until the
process exits (`railroad_env/opponent.py` reads it only on failure). Measured: 7,149 bytes over a
60-turn game, so a full 100-turn game stays well under a typical 64 KB pipe buffer. Verified by
running a game with stderr deliberately undrained. A per-cell dump would deadlock the agent
mid-game rather than fail cleanly.

**Speed.** The baked agent must answer in **50 ms per turn**, 1000 ms on the first. The forward
pass was 95% of the turn and is now 7.3x faster: one matmul per convolution against a
preallocated im2col buffer, plus BLAS pinned to one thread before numpy imports. Median went
20.75 ms → 2.85 ms, worst case 37.87 ms → 12.01 ms. Note: the textbook `as_strided` reshape is
**90x slower** here (numpy materialises it through a scattered copy); nine slice copies win.
`print(..., flush=True)` is mandatory — `test_submission.py` spawns with `-u`, which would hide a
buffering bug locally.

**Training cost is dominated by the opponent, not the network.** Rollout is ~93% of an iteration.
Per 100-turn game: level2 0.16 s, level2Pro 0.92 s, level2ProMax 0.86 s. Inside that, `autobuild`
is ~88%, because `GreedyAutoplaceOpponent` re-prices every unfinished connection from scratch
every turn. Caching it (invalidate on track landing on a route, or its region inking) is the
single biggest available speedup and is **not implemented**.

`games_per_iteration` should be a **multiple of** `num_workers`. Equal values mean one game per
worker, so the iteration waits for the slowest game; fewer games than workers leaves workers idle.
Each job also pickles the full ~8.8 MB state dict to its worker, so very large values trade
straggler smoothing for IPC.

## Current state — START HERE

*Last checked 2026-09-13. Everything below is verifiable from the repo — re-check it rather than
trusting it, this section has gone stale before.*

**What is shipped right now.** `submission.py` (87,348 chars) is the int4 bake of
`20260912-210537-distill_iter50.pt`, distilled from teacher `20260912-155239_iter500.pt`. It is the
first **value-distilled** student — `distill.py` now regresses the teacher's value alongside the
policy KL, so its critic ended at **0.975** correlation with the teacher instead of keeping its
random initialisation, and argmax agreement finished at **0.531** (the previous shipped student
managed 0.494, so the critic cost the policy nothing). Verified byte-identical to a re-bake of that
checkpoint. Submitted 2026-09-12; **the arena result is not in yet.**

Until it is, `114307_iter200` remains the only lineage in this repo validated from outside our own
eval table. Nothing below changes that.

**The league is now three rungs** (`level2Silver`, `level2Silver2`, `level2Silver3` — see the boss
list). Every shipped bake gets frozen under `baselines/` and added rather than swapped in, so no
past number changes meaning. Measured 10 games each:

| seat 0 | opponent | margin | win |
|---|---|---|---|
| rung 3 | `level2Silver2` | +103 | **0.70** |
| rung 3 | `level2Silver` | +750 | 0.90 |
| rung 2 | `level2Silver` | +859 | 0.90 |

Rung 3 against rung 2 is **the only non-saturated matchup this repo owns**, so it is what to point
any fit that needs losses at — the win-probability work above especially. The ladder is not a
rating: rung 3 beats rung 2 yet beats rung 1 by *less* than rung 2 does.

**Three PPO runs on 2026-09-12, and what each one taught.**

| run | config | iters | verdict |
|---|---|---|---|
| `20260912-074038` | `ppo_28ch_vs_silver_1k.yaml` | 552 | mechanically healthy, **specialised** |
| `20260912-155239` | — | 500 | produced the shipped teacher; ended with `clip_frac` 0.001 |
| `20260912-222648` | `ppo_28ch_vs_silver3.yaml` (first version) | 914 | **inert — no progress at all** |

`074038` is the clean demonstration of pool overfitting. Over 500 iterations it gained **+5.0%**
against `level2Silver` (its 0.40-weight training opponent) and lost **−14.8%** against
`level2ProMax`, with `level1`/`level1Pro`/`level2` — none of them in the pool — flat to −1.6%. Every
optimiser metric looked fine throughout (clip_frac 0.22 → 0.10, EV 0.91). **Health metrics cannot
see specialisation; only a held-out column can.**

`222648` is the more expensive lesson: **a run can be perfectly stable and still do nothing.** 914
iterations moved every eval column by less than its noise band — against `level2Silver3` it went
0.61 → 0.65 win with a dip to 0.46, at SE ≈ 0.05. Three causes, in order of size:

- **`lr` too cold, and annealed to zero.** 2.5e-5 was derived to continue a *different* run's tail
  over 500 iterations; reused on 1000 from another checkpoint it is simply small. `clip_frac` ran
  0.162 → **0.040** and `approx_kl` → 0.003 (healthy 0.05–0.2), i.e. the policy barely moved. With
  `lr_end: 0` the back third was at 2.5e-6 and could not have moved. **Set `lr_end` to ~20% of
  peak** — a run still training at the end costs nothing when checkpoints are picked from a table.
- **γ 0.998 with λ 0.99 made the gradient noisier.** γ 0.998 is a 500-decision discount horizon on
  a ~225-decision game, i.e. effectively undiscounted, so the critic must call the whole game from
  the opening. Isolated on identical games: advantage std **1.048 → 1.527 (+46%)** for +89% horizon,
  and EV fell 0.87 → 0.77 with value loss doubling and still climbing at iteration 900. Noisier
  gradients plus a smaller step is the one combination guaranteed to go nowhere.
- **No held-out column.** `eval_bosses` was the three rungs, all three in the training pool, so the
  table structurally could not tell "improving" from "specialising".

**`ppo_28ch_vs_silver3.yaml` is the fixed retry and has NOT been launched.** `lr` 5e-5, `lr_end`
1e-5, γ 0.997, λ 0.98, and — the part worth copying — **`level2Silver2` is held out of the pool
while staying in `eval_bosses`.** That gets a genuine generalisation column without adding a
trivial boss: it sits near win 0.85, so it has range, and it is never trained against. Read it
against `level2Silver3`: both rising is the run working, `level2Silver3` rising while
`level2Silver2` is flat is `074038` happening again.

`ppo_28ch_vs_silver2.yaml` was written, validated and never launched; it is the same shape one rung
down and is kept for its measurement notes.

**`win_bonus` changed which opponents matter, and the old numbers are wrong.** Advantage magnitude
across the pool now spreads **2.3x**, not 12.9x, and self-play went from 6.7% of the gradient to
**35.9% at 0.30 weight** — a mirror game is the one matchup whose outcome is in doubt, so the ±5.0
terminal term is most of its return. Self-play is a real use of compute now; see the
per-opponent-normalisation backlog item for the table.

**Suggested next steps, in order.**
1. Launch `ppo_28ch_vs_silver3.yaml`. Watch `clip_frac` stays in 0.05–0.2 and read
   `eval/level2Silver2` (held out) against `eval/level2Silver3` (contested) at every checkpoint.
2. Get the arena result for the current submission. Two of the last three runs produced agents our
   table liked; only the ladder can say whether any of it transferred.
3. **Snapshot league** — sample opponents from the run's *own* past checkpoints, not just the live
   weights and three hand-frozen bakes. It is the standard answer to "no opponent is strong enough"
   and it is what turns self-play from a treadmill into a ladder. Needs a `SelfPlayOpponent` variant
   holding a frozen state dict plus a sampler; small, but code.
4. Re-fit the win-probability probe on a `win_bonus` teacher against **rung 3 vs rung 2** — the
   contested matchup finally makes that test possible (see the debug-agent section).

**Backlog, deferred by decision, roughly in priority order:**
- **League**: sample training opponents from a *population* of past bakes rather than only the
  latest, so the agent cannot specialise against one style. `BakedSubmissionOpponent` already takes
  a `submission_path`, so `level2Silver` is a league of size one that we keep overwriting.
  Half done as of 2026-09-11: `baselines/` is where the population lives and `level2Silver` reads a
  frozen member of it rather than whatever was baked last, so the pool is at least *stable*.
  **A second and third rung landed 2026-09-12** (`level2Silver2`, `level2Silver3`, see the boss
  list above), so the league is now size three and the mechanism is proven — each further rung is a
  copied file plus a subclass. Rung 3 vs rung 2 is contested at 0.70, the first matchup here that
  is not saturated.
  What remains is sampling training opponents from the population rather than naming one, which is
  an `opponent_pool` edit, not new machinery. Note none of it is held out: every rung descends from
  the ancestor of everything we now train, so it measures progress against a fixed bar, not
  generalisation. Two rungs also mean any eval that does not pin `eval_bosses` now costs roughly
  two extra seconds per episode.
- ~~**Distil the value head too**~~ — BUILT 2026-09-12. `distill.py` now takes
  `value_loss_weight` (default 0, so nothing existing changes) and regresses the teacher's value
  alongside the policy KL, once per **unique state** rather than once per decision — the value does
  not move between a turn's sub-decisions, so weighting by decision count would only over-sample
  early turns. `distill.yaml` sets **0.25**. It is **free in submission size**: `bake_agent.py`
  drops the critic, and baking the same net at `value_hidden` 128 and 32 gives 101,233 against
  101,213 characters, a 20-character difference that is compression noise on the conv blob.

  **Use Huber, not MSE — the target's scale swings between iterations.** The opponent is sampled
  per game, so `distill/teacher_value_var` came out **3.6 / 6.3 / 23.6** over three consecutive
  iterations. Under plain MSE the unlucky iteration hands the *shared trunk* a ~6x larger gradient
  and drags the policy with it: at weight 0.25 the policy KL spiked to **4.60** against the **1.75**
  a `value_loss_weight: 0` baseline showed on the same seed and iteration. `value_huber: 2.0` fixes
  it. `distill/value_mse` is logged as a true MSE either way, so it stays comparable across runs.

  Paired 6-iteration A/B, same seed and therefore the same states (`value_loss_weight` 0.25 with
  Huber against 0):

  | iter | agree on/off | value MSE on/off | value corr on/off |
  |---|---|---|---|
  | 0 | 0.278 / 0.296 | 0.52 / 7.6 | +0.990 / −0.392 |
  | 2 | 0.387 / 0.398 | 1.84 / 56.2 | +0.997 / +0.653 |
  | 4 | 0.503 / 0.522 | 0.57 / 16.7 | +0.989 / +0.259 |
  | 5 | 0.303 / 0.301 | 1.31 / 7.3 | +0.907 / −0.148 |

  So the critic goes from noise to a **+0.99** correlation with the teacher for about **0.017** of
  argmax agreement. Iteration 5 collapsed to 0.30 in **both** arms, which is the read to copy: a
  wobble that appears in the baseline on the same seed is the run, not the value term. `distill/argmax_agreement` is the guard — the policy is what ships, so if a
  real run comes in below the **0.494** the shipped run reached, the trunk is being pulled and the
  weight is too high. `distill/value_corr` is the payoff. Two things downstream should change and
  are worth checking: `bake_debug_agent.py --calibrate` stops being a fiction, and the `--probe`
  column selection may finally pick the NN features over margin-and-turn — which would be the
  cleanest evidence the term did its job. A PPO fine-tune from these weights also stops spending
  its first iterations re-fitting a random critic.
- **Held-out eval**: keep 2–3 opponents *out* of the training pool. Right now `eval_bosses` and the
  training pool are nearly the same set, so the table structurally cannot detect pool overfitting.
- ~~**Terminal win bonus**~~ — BUILT 2026-09-11. `win_bonus` (raw score units, `+win_bonus` on a
  win, `-win_bonus` on a loss, 0 on a draw) is added to the final decision's reward in
  `ppo.py::_play_one_game`, then divided by `score_norm` like everything else. Default 0, so every
  existing config is unchanged; `ppo_28ch_vs_silver_1k.yaml` sets 5000. Two things to know. It
  interacts with `gae_lambda`, because GAE credits a terminal reward directly over only
  `1/(1-γλ)` decisions — measured on a 277-decision game, the last 19 decisions absorb **64%** of
  the bonus at λ=0.95 but only **36%** at λ=0.98, so raising λ is what spreads it back over the
  game. And it lands unevenly across the pool: ±5.0 against mean |return| per decision of 4.34
  (level2Pro) but 0.18 (level2Silver) and 0.08 (self-play). Expect the critic to take it badly at
  first — on a scratch run, value loss went 0.09 → 1.4 and explained variance 0.97 → ~0.8.
- **Per-opponent reward normalisation** — the clean fix for the spread in return scale across the
  pool, and **measured 2026-09-11 to be worse than the 12.9x previously filed**. Advantages are
  normalised ONCE globally over the mixed batch (`ppo.py:331`), so an opponent's influence is set
  by its return magnitude, not by its pool weight. On `114307_iter300`, 6 games each:

  | opponent | pool weight | mean abs return | share of the batch's advantage mass |
  |---|---|---|---|
  | level2Pro | 0.10 | 4.34 | 34.3% |
  | level2ProMax | 0.20 | 1.91 | 35.4% |
  | level2Silver | 0.50 | 0.18 | 23.6% |
  | self | 0.20 | 0.08 | 6.7% |

  The pool is weighted 50% to `level2Silver` *because* it is the only contested opponent, and it
  contributes less gradient than `level2Pro` at a fifth of the weight. Self-play gets 6.7% for 20%
  of the compute. The weights do not do what the config comment says they do. A log-ratio reward
  was analysed and **rejected**: the smoothing constant is a dial between "shut-outs explode"
  (spread 35x) and "literally margin/c", with a best case of 11.0x against margin's 12.9x.

  **`win_bonus` largely fixed this, and the self-play row above is now obsolete (measured
  2026-09-12).** Re-run on `074038_iter500` under γ=0.997, λ=0.98, `win_bonus` 5000, `margin`,
  4 games per opponent:

  | opponent | mean abs advantage / decision | decisions/game |
  |---|---|---|
  | level2Pro | 1.667 | 320 |
  | **self** | **1.196** | 265 |
  | level2Silver2 | 0.911 | 250 |
  | level2Silver | 0.769 | 261 |
  | level2ProMax | 0.730 | 289 |

  The spread is **2.3x**, not 12.9x. A mirror game is the one matchup whose *outcome* is genuinely
  in doubt, so the ±5.0 terminal term is most of its return rather than a rounding error on a
  blowout — self-play went from 6.7% of the advantage mass to **35.9% at 0.30 weight**. Two
  consequences: self-play is now a real use of compute rather than a documented waste, and pool
  weights approximately mean what they say under this reward. Per-opponent normalisation is
  correspondingly less urgent than the 12.9x figure implies.
- `gae_lambda` 0.95 → 0.98 (roughly triples the real-reward horizon; costs variance).
- Potential-based shaping on the scoring-rate derivative — policy-invariant so it is safe, but its
  upside shrank once EV reached 0.98.
- Value-function clipping, KL early-stopping, collecting the opponent's transitions in self-play
  games (~2x data from that share), value-head dropout consistency.
- `GreedyAutoplaceOpponent` pricing cache — the documented biggest speedup, and a prerequisite if
  MCTS is ever revived.
- **MCTS**: discussed and parked. Inference-time search is impossible (50 ms budget vs 2.85 ms per
  forward pass), so it could only be a training-time policy-improvement operator. The legacy stack
  is staler than it looks: `mcts.py` has a 2-channel action space and `SCORE_NORM = 200` calibrated
  for "300-700 point" games, against today's 20,000+ — its value target would saturate at ±1 on
  every game. Reviving it is a port, not a config change.

**Known-stale, left in place deliberately:** `ppo_finetune_student.yaml` (`REPLACE_ME` path,
targets a 25-channel student — unusable as-is), and the legacy AlphaZero/MCTS stack. 88 of the 123
checkpoints are 25-channel and invalid against the current environment (~1.4 GB); they were left
alone because CLAUDE.md's own rule is to ask first.

Eval always runs the production **100-turn** horizon. Training episodes are deliberately shorter
in the `ppo_28ch_*` configs, which means the turn channel spans only 0.00–`max_turns/100` during
training against 1.00 at eval; see the level2ProMax post-mortem above for what that cost.

**Checkpoints that matter:** `checkpoints/20260910-224257_iter200.pt` — the original +2367-on-
level2ProMax peak that every later lineage descends from — `20260911-114307_iter200.pt`, the
arena-validated teacher, and `20260912-155239_iter500.pt`, the teacher behind what is shipped now.
Do not let any of them get swept up in a cleanup.

The same goes for all three files in `baselines/`, which are not checkpoints but **are** the
`level2Silver` / `level2Silver2` / `level2Silver3` bosses — deleting one breaks every eval that
names that tier, and unlike a checkpoint it cannot be re-baked into the same bytes unless the
matching `.pt` still exists.

**Ladder of configs**, each warm-starting from the last:
`ppo_28ch_level1Pro.yaml` (cold start) → `ppo_28ch_level2Pro.yaml` → `ppo_28ch_level2ProMax.yaml`
→ `ppo_28ch_mixed_selfplay.yaml` (opponent pool + self-play) → `ppo_28ch_vs_silver.yaml` (adds
`level2Silver`) → `ppo_28ch_vs_silver_1k.yaml` (1000 iterations, `gae_lambda` 0.98, `win_bonus`
5000) → `ppo_28ch_vs_silver2.yaml` → `distill.yaml` (`quant_aware: int4`, its own opponent pool).

`ppo_28ch_vs_silver2.yaml` (added 2026-09-12) forks `runs/20260912-074038` at its `_iter500` and
swaps the pool's 0.40 slot from `level2Silver` to `level2Silver2`, because the old rung is solved:
the warm start beats it **0.92** (12 games) against **0.83** for the new one, and the parent run's
own 100-game column reached +1168 win 0.90. It also continues rather than restarts the anneal —
`lr` 2.5e-5 over 500 iterations is exactly the parent's tail, since `decay_progress` runs over the
full count. Both rungs stay in `eval_bosses`: the old one keeps every earlier table comparable, the
new one is the column to pick a checkpoint on. Nothing here is held out — `level2Silver2` is the
distillation of this run's own teacher `074038_iter350`, so it is a fixed bar, not generalisation.

**`submission.py`**: baked 2026-09-12 from `20260912-210537-distill_iter50.pt` — the first
**value-distilled** student — at 87,348 characters, int4, quantization-aware. `force_disrupt` is
recorded in that checkpoint, so the bake drops the SKIP_DISRUPT branch automatically. It is frozen
byte-for-byte at `baselines/20260912-210537-distill_iter50.py` and is what `level2Silver3` plays
as, so re-baking `submission.py` does not move that tier.

**Re-baking `submission.py` is now safe for training and eval**, which it was not before
2026-09-11. `level2Silver` reads a frozen copy under `baselines/`, so overwriting `submission.py`
no longer changes the training opponent or the eval baseline of a running job. Two habits still
matter: before submitting a new bake, copy it into `baselines/` so the agent it replaces stays
measurable, and remember that `submission.py` is what `test_submission.py` runs, so a bad bake is
still visible there immediately.

The runtime is **verified seat-invariant** — same board fed as player 0 and as player 1 with track
owners swapped yields byte-identical commands over 150 decisions. The only absolute field on the
wire is `tile.track`, remapped by `foe = 1 - my_id`; everything else (board, towns, scores) arrives
own-first, the referee does no mirroring, turns are simultaneous, and a same-cell collision
resolves to neutral for both. Note `test_submission.py` hardcodes `my_id = 0`, so nothing in the
repo exercises the player-1 path by default.

**`checkpoints/` holds 168 files, 3.0 GB as of 2026-09-13, and is mixed** — 88 are 25-channel and therefore invalid
against the current environment (also semantically stale on channels 5/6/8/9 and 10–21); 34 are
28-channel, and one legacy file declares neither. Check `model_kwargs["in_channels"]` before
loading anything. Ask before deleting.

`ppo_finetune_student.yaml` still has a `REPLACE_ME` teacher path and targets a distilled
25-channel student, so it cannot be used as-is.

## Shared package

`/home/carlo/claude_env` is a standalone repo for others to train against: the environment,
`encoding.py`, a README and `play_vs_bosses.ipynb` (agent example, boss ladder, 28-channel
heatmap). pygame is optional there via a lazy import. Its notebook currently has text output but
no figures, because it was executed with a headless matplotlib backend.

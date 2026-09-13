# Frozen baselines

Bakes we have actually submitted, kept byte-for-byte. **The `level2Silver`, `level2Silver2` and
`level2Silver3` bosses read these, not the live `submission.py`** — that is the entire point of the
directory.

It used to read `submission.py`, which made it live state: re-baking silently changed the training
opponent and the eval baseline of anything then running, and `eval/level2Silver/margin` stopped
being comparable across the change. Pointed at a file nothing overwrites, the column means the same
thing across re-bakes and across runs.

| file | baked from | note |
|---|---|---|
| `20260911-144318-distill_iter50_rank288.py` | `checkpoints/20260911-144318-distill_iter50.pt`, teacher `20260911-114307_iter200.pt` | 87,679 chars, int4, quantization-aware. Reached **CodinGame rank ~288** — the only agent here validated from outside our own eval table. **This is what `level2Silver` plays as.** |
| `20260912-210537-distill_iter50.py` | `checkpoints/20260912-210537-distill_iter50.pt`, teacher `20260912-155239_iter500.pt` | 87,348 chars, int4, quantization-aware, value-distilled (critic correlation 0.975 with its teacher). **This is what `level2Silver3` plays as**, and the current top of the league. Submitted 2026-09-12; arena result not yet in. |
| `20260912-124725-distill_iter50.py` | `checkpoints/20260912-124725-distill_iter50.pt`, teacher `20260912-074038_iter350.pt` | 87,104 chars, int4, quantization-aware. First student distilled with `value_loss_weight`, off the `win_bonus` teacher. **This is what `level2Silver2` plays as.** Not yet submitted, so it carries **no arena evidence** — it is a stronger local bar, not a validated one. |

Added 2026-09-12, verified byte-identical to the `submission.py` baked from that checkpoint (a
re-bake of the checkpoint reproduces the file exactly, `bake_agent.py ... --quant int4`).

**The rungs measured against each other**, 10 games each over the same seeds:

| seat 0 | opponent | score | opp | margin | win |
|---|---|---|---|---|---|
| rung 3 (`210537`) | `level2Silver2` | 960 | 857 | **+103** | **0.70** |
| rung 3 (`210537`) | `level2Silver` | 2879 | 2129 | +750 | 0.90 |
| rung 2 (`124725`) | `level2Silver` | 3214 | 2354 | +859 | 0.90 |
| rung 1 (`144318`) | `level2Silver2` | 2354 | 3214 | −859 | 0.10 |

The last two are the same pairing from both seats and mirror exactly, which doubles as the
seat-invariance check.

Two things to read out of this. **Rung 3 against rung 2 is the most contested matchup this repo
owns** — 0.70, where everything else saturates at 0.90+ — so it is the right opponent for anything
that needs losses to learn from, calibration fits especially. And **the ladder is not a scale**:
rung 3 beats rung 2, but beats rung 1 by *less* than rung 2 does (0.90 at +750 against 0.90 at
+859). Non-transitivity is normal here; a rung is a fixed opponent, not a rating.

Note also how small the scores are in the rung-3-vs-rung-2 games: 960 to 857, against ~23,000
against a scripted builder. Two strong disruptors ink the board out and destroy each other's track.
That regime is most of what the league measures, and it may not be what the arena plays. A useful third reading: playing
`submission.py` against `level2Silver2` while the two are the same bytes scores **0:0 on every
seed** — identical agents pick the same cell each turn and every track resolves to NEUTRAL. That
is the cheapest confirmation that a rung is pointed at the file you think it is, not a bug.

Nothing in here is ever edited or re-baked. To raise the bar, copy the new bake in under a name that
records the checkpoint it came from, then point `BakedSubmissionOpponent.DEFAULT_PATH` at it. That
is deliberately a code change rather than a file copy, because it moves every number the tier has
ever produced — record which baseline a table was measured against.

A second and third frozen rung were added that way on 2026-09-12 (`Level2Silver2Opponent` and
`Level2Silver3Opponent`, each a subclass pinning its own `DEFAULT_PATH`, plus entries in
`BOSS_TIERS` and `OPPONENT_STRATEGIES`). Adding a rung is
preferred over repointing `DEFAULT_PATH`: repointing would retire the rank-288 bake and silently
change the meaning of every number ever measured against that tier, while a new rung keeps both
columns. Error messages read the tier name off `TIER_NAME` so a failure names the rung that
actually failed rather than always saying `level2Silver`.

No rung is **held out** — all three descend from the lineage everything here now trains on, so they
measure progress against a fixed bar, not generalisation. That remains open in CLAUDE.md.

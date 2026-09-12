# Frozen baselines

Bakes we have actually submitted, kept byte-for-byte. **The `level2Silver` boss reads one of these,
not the live `submission.py`** — that is the entire point of the directory.

It used to read `submission.py`, which made it live state: re-baking silently changed the training
opponent and the eval baseline of anything then running, and `eval/level2Silver/margin` stopped
being comparable across the change. Pointed at a file nothing overwrites, the column means the same
thing across re-bakes and across runs.

| file | baked from | note |
|---|---|---|
| `20260911-144318-distill_iter50_rank288.py` | `checkpoints/20260911-144318-distill_iter50.pt`, teacher `20260911-114307_iter200.pt` | 87,679 chars, int4, quantization-aware. Reached **CodinGame rank ~288** — the only agent here validated from outside our own eval table. **This is what `level2Silver` currently plays as.** |

Nothing in here is ever edited or re-baked. To raise the bar, copy the new bake in under a name that
records the checkpoint it came from, then point `BakedSubmissionOpponent.DEFAULT_PATH` at it. That
is deliberately a code change rather than a file copy, because it moves every number the tier has
ever produced — record which baseline a table was measured against.

A second frozen rung can also be added as its own tier (a subclass pinning its own `DEFAULT_PATH`,
plus entries in `BOSS_TIERS` and `OPPONENT_STRATEGIES`). A population of past selves rather than one
opponent is the "league" in CLAUDE.md's backlog, and it is the reason these files are kept rather
than deleted once superseded.

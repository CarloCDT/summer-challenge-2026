# Frozen baselines

Bakes kept byte-for-byte. The `level2Silver*` bosses play these files, not the live
`submission.py`, so re-baking never changes a training opponent or an eval column.

| boss | file | student checkpoint | teacher | note |
|---|---|---|---|---|
| `level2Silver` | `20260911-144318-distill_iter50_rank288.py` | `20260911-144318-distill_iter50` | `20260911-114307_iter200` | reached CodinGame rank ~288 |
| `level2Silver2` | `20260912-124725-distill_iter50.py` | `20260912-124725-distill_iter50` | `20260912-074038_iter350` | |
| `level2Silver3` | `20260912-210537-distill_iter50.py` | `20260912-210537-distill_iter50` | `20260912-155239_iter500` | |
| `level2Silver4` | `20260913-113208-distill_iter50.py` | `20260913-113208-distill_iter50` | `20260913-065817_iter200` | current best candidate |

All are int4, quantization-aware bakes. Each was verified byte-identical to a re-bake of its
checkpoint.

Head-to-head, 10 games each: rung 3 beat rung 2 +103 (win 0.70), rung 3 beat rung 1 +750 (0.90),
and rung 2 beat rung 1 +859 (0.90). Results aren't transitive, so a rung is a fixed opponent, not a
rating.

## Rules

- **Never edit or re-bake a file here.** Add a new rung instead: copy the bake in, add a
  `BakedSubmissionOpponent` subclass with its own `DEFAULT_PATH` and `TIER_NAME`, and register it in
  `BOSS_TIERS` and `OPPONENT_STRATEGIES` in `railroad_env/opponent.py`.
- **Check a new rung** by playing the live `submission.py` against it while they're the same bytes.
  It should score 0:0 on every seed.
- None of these are held out from our training lineage. They measure progress against fixed bars,
  not generalisation.

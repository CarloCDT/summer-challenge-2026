# Frozen baselines

Bakes we have actually submitted, kept byte-for-byte so a margin against one means the same thing
next month as it does today. Nothing in here is ever edited or re-baked: `submission.py` at the
repo root is live state that the `level2Silver` boss reads at run time, and re-baking it silently
moves that boss and invalidates every `eval/level2Silver/*` number. These files are the fixed
rungs that survive a re-bake.

| file | boss tier | baked from | note |
|---|---|---|---|
| `20260911-144318-distill_iter50_rank288.py` | `level2SilverPro` | `checkpoints/20260911-144318-distill_iter50.pt`, teacher `20260911-114307_iter200.pt` | 87,679 chars, int4, quantization-aware. Reached **CodinGame rank ~288** — the only agent here validated from outside our own eval table. |

To add a rung: copy the bake in under a name that records the checkpoint it came from, then give it
a tier in `railroad_env/opponent.py` (a `FrozenSubmissionOpponent` subclass with its own
`DEFAULT_PATH`, plus entries in `BOSS_TIERS` and `OPPONENT_STRATEGIES`). That is the whole of the
"league" idea in CLAUDE.md's backlog: a population of past selves rather than only the latest.

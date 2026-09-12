# Project Summary

Engineering notes for the Railroad Tycoon RL project: what the code is, why it's shaped this
way, and which decisions were deliberate. For the game rules and usage, see [README.md](README.md).

## What this is

A faithful Python port of the CodinGame Summer Challenge 2026 referee (Java source vendored in
`SummerChallenge2026/`), wrapped as a Gymnasium environment, plus two training stacks: PPO
(primary) and an AlphaZero-style MCTS variant.

The port was written by reading the Java line by line; `verify_rules.py` encodes the result as 57
executable checks, each citing the Java method it mirrors. If either side changes, that file is
the diff.

---

## Architecture

### `railroad_env/` — the game

Split to mirror the Java packages, so the two can be compared file by file:

| File | Mirrors | Notes |
|---|---|---|
| `constants.py` | `Game.java`'s constant block | Single source of truth for budgets, costs, thresholds, map sizing |
| `grid.py` | `grid/Tile,Zone,Town,Grid.java` | Coordinates are `(x, y)` tuples; numpy arrays stay `[y, x]` |
| `grid_maker.py` | `grid/GridMaker.java` | The most intricate port — see below |
| `pathfinding.py` | `grid/pathfinding/*` | `train_bfs`, `terrain_path_exists`, `autobuild` |
| `game_state.py` | `Game.java` | Turn resolution, scoring, disruption, plus the RL observation |
| `environment.py` | `Referee.java` (loosely) | Gymnasium wrapper; actions use the game's own vocabulary |
| `opponent.py` | `config/Boss.py`, `config/level2/Boss.py` | `passive` and `boss` are the real shipped AIs |

**Turn order** (`GameState.resolve_turn`) follows `Game.performGameUpdate` exactly: expand
AUTOPLACE → all PLACE_TRACKS (both players, simultaneously, against the turn-start board) → all
DISRUPT → ink unstable regions → recompute connections and score → update tile states. Getting
this order wrong changes outcomes: a track placed on the same turn its region inks is built,
then destroyed, and scores nothing.

**Two sources of truth, kept in lockstep.** `Grid` holds `Tile` objects (the authoritative game
state), while `GameState` maintains numpy mirrors (`tracks`, `terrain`, `regions`) for cheap
array access in the observation encoder and renderer. All track mutations funnel through
`_set_track` so the two cannot drift.

### Map generation quirks worth knowing

`grid_maker.py` reproduces `GridMaker.java` including behaviour that looks accidental:

- **Rivers** flow from map edges with a directional weighting (1.75 preferred / 1.0 lateral /
  0.25 reverse), split with probability 0.055, refuse to run adjacent to existing water
  (8-neighbourhood, ignoring their own last two cells), and are reverted to grass if shorter than
  3 cells. The first river of every map always splits.
- **Zones** grow round-robin: each zone in turn claims one random unassigned neighbour, until
  every zone is boxed in. This yields near-equal blobs, not BFS floods.
- **Town placement** has a fallback pass that skips the terrain and edge checks the first pass
  applies. It only fires when the zone-based pass runs short, and the town's tile is forced to
  plains afterwards regardless — so the outcome is still legal, just placed differently.
- **Desired connections** are filtered for reciprocity *sequentially*, so later towns are
  filtered against the already-filtered lists of earlier ones. Order matters, and the port
  preserves it.

### `training/` — the learning stack

`GameSimulator` is the bridge between the game's turn-at-a-time interface and an agent that wants
to make one decision at a time. It holds an independent copy of the state, accumulates pending
placements while recomputing legality against *board + pending*, cascades PLACE → DISRUPT → turn
resolution as budgets run out, and only then submits the whole turn (which is when the opponent
moves and scoring happens).

`TrainUNet` is a dual-head U-Net, ~4.0M params: two encoder stages, a bottleneck feeding the
value head, and a decoder with skip connections producing the per-cell policy heatmap. Two 2×2
pools take 20×30 → 10×15 → 5×7; because 15 doesn't halve evenly the decoder upsamples to
explicit target sizes rather than a fixed `scale_factor`.

---

## Decisions and their reasons

**Padding to a fixed 20×30 canvas.** The real game randomizes board size every match. Rather
than restrict training to one size, every board is centered in the largest possible canvas and
the padding is marked as inked — the same signal a destroyed region carries, so "unbuildable"
is one concept the network learns once. This is why board coordinates and canvas coordinates
differ, and why anything indexing the policy heatmap must go through `pad_offsets`.

**The value head is bounded only for AlphaZero.** `TrainUNet(bounded_value=True)` puts a `Tanh`
on the value head, which is right when the target is a normalized win margin. PPO regresses onto
discounted returns whose magnitude isn't bounded, so it passes `False`. An earlier version had
PPO fitting a `Tanh` head to raw score deltas hundreds of times larger than anything it could
output; the value loss exploded into the tens of thousands.

**PPO rewards are scaled by `SCORE_NORM = 100`.** Scoring compounds — a good agent earns
hundreds of points per turn late in a 100-turn game — so raw per-turn deltas make for badly
conditioned returns. This is a conditioning choice, not a range constraint.

**PPO discounts (`gamma=0.99`, `gae_lambda=0.95`).** Earlier simplified experiments used
undiscounted returns, justified by disruption being switched off: a completed connection's future
payout was then *guaranteed*. In the faithful game a region can ink out and destroy your track,
so that premise no longer holds.

**Epsilon-greedy on top of PPO.** A purely stochastic policy essentially never assembles the
contiguous corridor a connection requires — measured at 0/20 games scoring anything, against
13/20 for pure argmax with the *same untrained network*. A random-init CNN's output is spatially
correlated, so argmax naturally clusters picks near each other and near existing track; sampling
destroys that structure. Epsilon decays 0.5 → 0.05 over a run. The recorded `log_prob` is always
the policy's own probability for whichever action was taken, which is the standard practical
approximation rather than a correct off-policy correction.

**Parallel rollouts pin each worker to one thread.** PyTorch defaults every process to
intra-op-parallelizing across all cores, so N workers each grabbing 32 threads on a 32-core box
thrashes badly — measured at ~28× CPU-time-to-wall-time oversubscription, turning four tiny
forward passes into a 30-second stall. Workers call `torch.set_num_threads(1)` and run on CPU;
the parallelism comes from processes, not threads. Pools use `spawn`, since forking a process
that already holds a CUDA context is unsafe.

---

## Deliberate deviations from the Java

Both are invisible to gameplay and documented in `verify_rules.py`:

1. **Random streams.** Java's `Random` and numpy produce different sequences, so a given seed
   yields a different — but distributionally identical — map. Every range, probability and
   weighting matches.
2. **`autobuild`'s heuristic.** `AutobuildAStar.heuristic` is `cursor.manhattanTo(cursor)`,
   i.e. always 0, making the original a uniform-cost search rather than a guided A*. The port
   keeps that: an admissible heuristic would generate different paths than the real game does.

## The lite game

`railroad_lite_env.py` is a `gym.Wrapper` that simplifies the faithful env for training:
plains-only terrain, no disruption, 25 turns, a half-idle random opponent, and 16 observation
channels instead of 28. It is a wrapper rather than a set of env flags deliberately — the
earlier version of this project had those simplifications baked into the env itself, which made
it impossible to tell at a glance whether a given run was playing the real game.

Implementation notes:

- **Terrain is flattened after generation**, not during it. Towns therefore sit exactly where
  the real generator put them (it already placed them on plains), and only the *costs* change.
- **`LiteGameSimulator` overrides `_compute_legal` to return nothing for the disrupt phase**, so
  the base class's phase cascade resolves the turn the moment the paint budget runs out. That
  gives exactly 3 decisions per turn with no special-casing.
- **The lite inked channel is really a padding indicator.** With no disruption nothing ever
  inks, so channel 14 is 1.0 exactly on the off-board padding. It is kept for that reason.
- **`GameSimulator.clone()` uses `type(self)`**, not a hardcoded class — otherwise cloning a
  `LiteGameSimulator` (as MCTS would) silently degrades it to full-game behaviour. That was a
  latent bug found while building the lite path.
- **`LITE_SCORE_NORM = 50`** rather than the full game's 100: 25 turns at 3 placements score in
  the hundreds, not the tens of thousands.

## Known gaps

- The AlphaZero/MCTS stack (`train.py` and its configs) works against the faithful env but has
  had far less mileage than PPO since the port. Its search-free variants (`no_mcts`,
  `unet_only`, `connect_towns_topk`) are pipeline checks, not serious training strategies —
  their policy targets are self-imitation with no advantage signal, which reinforces whatever
  the network already does regardless of whether it worked.
- No wrapper layer yet for curriculum simplifications (shorter games, disabled disruption,
  all-plains terrain). Those existed before the port and were dropped to keep the env
  unambiguously faithful; they belong in a `gym.Wrapper` rather than in the env itself.
- The observation has no explicit turn counter or score channel. A finite-horizon value function
  would likely benefit from knowing how many turns remain.

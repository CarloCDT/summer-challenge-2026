# Architecture Overview

## Files and Components

### Core Game Engine
- **`game_state.py`** - GameState class that manages all game logic
  - Map representation (terrain, regions, towns, tracks)
  - Action execution (placing tracks, disrupting regions)
  - Connection finding (pathfinding between towns)
  - Scoring system (active connections)
  - Observation generation (HxWx25 tensor)

### Gymnasium Integration
- **`environment.py`** - RailroadGymEnv class
  - Gymnasium.Env subclass for standard RL training
  - Map generation (procedural with random seed)
  - Action space: `Dict({"actions": Sequence(...)})`
  - Observation space: `Box(low=0, high=1, shape=(height, width, 25), dtype=float32)`
  - Step logic: resolves both players' PLACE actions together (handles same-cell collisions),
    then both players' DISRUPT actions, then scoring
  - Reset/close lifecycle management

### AI Opponents
- **`opponent.py`** - Multiple opponent implementations
  - `RandomOpponent`: Random valid actions each turn
  - `GreedyOpponent`: Priority-based placement and disruption
  - `FixedBehaviorOpponent`: Custom user-defined behavior

### Rendering
- **`renderer.py`** - GameRenderer class using Pygame
  - Headless and visual modes
  - Grid rendering with terrain coloring
  - Track visualization by owner (red/blue/gray)
  - Town visualization
  - UI panel showing scores and resources

## Data Structures

### Observation Space (HxWx28)
Each cell in the H×W grid has 28 feature channels. Most are in `[0, 1]`; channels 10-21 and 26 are
**signed**:

```
[0:3]   - Terrain (one-hot: plains, river, mountain)
[3]     - Enemy track presence (0 or 1)
[4]     - Own track presence (0 or 1)
[5]     - Enemy tracks in this cell's region, COUNT / 10
[6]     - Own tracks in this cell's region, COUNT / 10
[7]     - This region's size, cells/total_map_cells
[8]     - Enemy tracks in region on an active connection, COUNT / 10
[9]     - Own tracks in region on an active connection, COUNT / 10
[10:22] - One channel per town id slot (0-11), SIGNED route guide - see below
[22]    - Region instability (0-1, normalized by max 4)
[23]    - Region inked out (0 or 1)
[24]    - Part of an active (scoring) connection (0 or 1)
[25]    - A town stands here (ownerless, never buildable)
[26]    - (own - enemy score) / 10000, flat across every cell (SIGNED)
[27]    - turn / 100, flat across every cell
```

Channels 5/6/8/9 hold **raw counts, not densities** - a deliberate reversal. Whether a region gets
inked turns on an *absolute* lead (enemy - own), so dividing by region size hid the deciding
quantity behind a product of three channels. `TRACK_COUNT_SCALE = 10` is a fixed divisor, not a
normalizer; mean region size is ~8 cells, so these land in roughly 0..1 but may exceed it.
Channels 26/27 are flat planes because a convolution is local.

Changing any of this means updating `railroad_env/game_state.py` (`get_observation`), the
`observe()` inside `bake_agent.py`'s `RUNTIME`, and `CHANNEL_NAMES` in `debug_full_agent.ipynb`
**in lockstep**, then proving parity between the two implementations.

Channels 5-9 are per-region scalars, computed once per region and broadcast to every cell that
belongs to it (same pattern as the old region-id channel, just richer). Neutral (owner 2) tracks
count toward *both* enemy and own totals in every one of these channels, per design - a contested
cell should read as "presence" for both sides, not neither.

**Channels 10-21** (`GameState.TOWN_CHANNEL_START`, `NUM_TOWN_CHANNELS`): channel `10 + town_id`
marks the shortest *open-terrain* route from that town to each of its `desired_connections`
targets (unioned over all targets if more than one). The values are **signed**
(`ROUTE_ORIGIN_VALUE = -1.0` at the channel's own town, `ROUTE_DEST_VALUE = -0.5` at a town it
wants, `ROUTE_PATH_VALUE = +1.0` on the cells between) so that a single 3x3 filter can separate
endpoints from corridor. A plot with `vmin=0` will hide both town values against the background.
"Open-terrain" is deliberately more permissive than `find_connection` (the active-
connection checker): any non-inked cell is passable regardless of terrain type or whether a track
exists there yet, and existing tracks (either owner's, or neutral) don't divert or block the route
- they're just normal ground. This makes it a "build along this line" guide that's meaningful even
before any tracks exist, unlike channel 24 which only lights up once a connection is actually
completed. A town id ≥ the map's actual town count leaves its channel all zero.

`info['towns']` (from `reset()`/`step()`) carries the full `{id, x, y, desired_connections}` list -
channels 10-21 tell you *where* a route goes, not which town id owns which channel or what its
targets are; that mapping has to come from `info['towns']`.

`info` also carries `player_actions`/`opponent_actions` (`{"placed": [...], "disrupted": [...]}`, what
each side actually got applied that turn) and `window_open` (`False` once a `render_mode="human"`
window has been closed - a driving loop should stop stepping when this goes `False`).

### Action Format
Actions are a dictionary with a list of fixed-shape `(action_type, (param1, param2))` tuples, matching
`action_space` exactly (`Discrete(3)` + a 2-element `Box`) so `action_space.sample()` is always valid:
```python
{
    "actions": [
        (0, (0, 0)),          # WAIT (params ignored)
        (1, (x, y)),          # PLACE at (x, y)
        (2, (region_id, 0)),  # DISRUPT region (2nd param unused)
    ]
}
```

Actions are executed in order until resources are exhausted. Invalid actions are skipped. Note: the
Opponent classes (`opponent.py`) use a separate, private tuple format (`('PLACE', x, y)`,
`('DISRUPT', region_id)`, `('WAIT',)`) - this is an internal contract between `environment.py` and
`opponent.py`, never exposed through `action_space`.

### Game State
Key properties of GameState:
- `tracks[H][W]`: -1=empty, 0=player0, 1=player1, 2=neutral
- `region_instability`: dict mapping region_id → instability (0-4)
- `inked_regions`: set of region_ids that are destroyed
- `paint_points[2]`: remaining paint for each player
- `disruption_points[2]`: remaining disruption for each player
- `scores[2]`: current scores

## Key Algorithms

### Pathfinding (shared BFS core)

Both pathfinding methods delegate to `_bfs_shortest_path(start, goal, is_passable)`, which differ
only in their `is_passable(x, y)` predicate:

```
find_connection(from_town_id, to_town_id):        # "is this pair already connected?"
  - passable if the cell has a track (any owner) OR is any town (not just source/destination -
    a route can legitimately pass through a third town's cell)
  - inked regions never passable
  - used for scoring (calculate_active_connections) and observation channel 24

find_shortest_route(from_town_id, to_town_id):    # "where would a connection go?"
  - passable if the cell is simply not inked - terrain type and existing track ownership
    (or lack of any track at all) don't matter
  - used for observation channels 10-21 (per-town route guide)
```

Shortest path and the N>E>S>W tie-break (per spec, "prioritize N first, then E, then S, then W when
moving from the requesting town to the desired connected town") both fall out of how the shared BFS
is implemented: neighbors are always tried in fixed N/E/S/W order, and `visited` is marked at
*enqueue* time rather than dequeue time. That combination guarantees the first path to reach any
cell (including the goal) is the lexicographically-smallest-by-direction shortest path from the
source - exactly the spec's tie-break rule, not just "a" shortest path. Verified directly: on a
fully-tracked grid with multiple equal-length routes, the BFS consistently prefers N/E over S/W at
every branch point when either choice remains optimal.

### Active Connections Calculation
```
For each town with desired connections:
  - Find shortest path to desired town
  - Mark all cells in path as active
  - Track which town pairs use each cell
```

### Scoring
```
For each cell with a track:
  - If cell is part of any active connection
  - Award 1 point to track owner (skipped for neutral/owner-2 tracks) per active connection through it
  - Points awarded at end of turn, after all PLACE and DISRUPT actions (both players) resolve
```

### Map Generation
```
1. Generate random terrain (plains 70%, river 15%, mountain 15%)
2. Grow regions via randomized BFS (8-25 cells each) so the map has many small,
   separately-colored regions rather than one connected blob
3. Place towns on random plain cells: at most 1 per region, and never in a region
   adjacent to another region that already has a town
4. Assign each town 0-3 desired connections to any other town (not just earlier ones);
   a post-pass guarantees every town is the target of at least one other town's
   desired_connections, per spec
```

## Resource Management

### Paint Points
- **Budget**: 3 per turn, reset each turn
- **Costs**: Plains=1, River=2, Mountain=3
- **Execution**: Actions are executed in order, skipped if insufficient paint

### Disruption Points
- **Budget**: 1 per turn, reset each turn
- **Effect**: Increase region instability by 1
- **Inked Out**: Region destroyed at instability ≥ 4

### Action Execution Order
1. Both players' PLACE actions are collected against the turn-start board state and resolved
   together (`collect_placements` + `resolve_placements`): a cell claimed by only one player
   becomes that player's track; a cell claimed by both becomes a neutral (owner 2) track, and
   both players are charged paint for it
2. Both players' DISRUPT actions are applied (always after all placements, per spec)
3. Scoring phase (`score_round`)
4. Resources reset, turn incremented

## Training Considerations

### Observation Design
- Channels 0-2 (terrain) are constant after reset
- Channels 3-4 (per-cell track presence) change based on player/opponent actions
- Channel 7 (region size) is constant; channels 5-6/8-9 (region track/active-connection
  aggregates) change dynamically as tracks are placed and connections form/break
- Channels 10-21 (per-town open-terrain routes) are constant after reset - they only depend on
  `desired_connections` and inked regions, and no region gets inked until turn 4 at the earliest
  (4 disruption points needed); a route only ever changes if a region along it gets inked out
  mid-game, in which case that route's channel updates to route around it (or goes empty if no
  route remains)
- Channels 22-24 (instability, inked, active connection) change dynamically

### Action Space Design
- Discrete action types: WAIT (0), PLACE (1), DISRUPT (2)
- Each action is a fixed-shape `(type, (param1, param2))` tuple, so `action_space.sample()`
  always produces something `step()` accepts
- Variable number of actions per turn (0 to N) via a Gymnasium `Sequence` space
- Natural for autoregressive policies or masking approaches

### Reward Structure
- Single scalar reward per step, equal to player 0's actual score delta that turn
  (verified to sum exactly to the final `scores[0]` over an episode)
- Aligned with game goal (maximize points) by construction, not by approximation
- Relatively sparse early in episode but denser once connections form
- Can be augmented with auxiliary rewards (e.g., regional control)

### Episode Structure
- Fixed length: 100 turns
- Deterministic given seed
- Both players take actions simultaneously each turn
- No hidden information (full game state visible in observation)

## Known Limitations & Future Improvements

### Current Limitations
1. **Impossible actions are silently skipped, not reported**: matches the spec's "commands that are
   impossible actions are skipped" rule, but there's no feedback channel telling an agent why an
   action didn't do anything - it must infer this from before/after observation diffs
2. **Opponent doesn't use AUTOPLACE**: Simplifies logic but limits opponent strength
3. **Random opponent uses raw probability**: Could be improved with weighted strategies
4. **No early-win check for "all desired connections impossible"**: the spec's second victory
   condition isn't implemented - episodes always run the full 100 turns. This is a deliberate choice
   (fixed-length episodes), not an oversight
5. **Terrain probability fixed**: No way to generate harder/easier maps

### Potential Improvements
1. **Enhanced opponents**
   - Rule-based bot using game analysis
   - Self-play training
   - Heuristic evaluation of board state

2. **Curriculum learning**
   - Procedural difficulty (map size, town count)
   - Opponent progression
   - Resource constraints

3. **Observation variants**
   - Partial observability (fog of war)
   - Player-specific observations
   - Flattened vs image-based formats

4. **Reward shaping**
   - Auxiliary rewards (track placement, connection formation)
   - Relative scoring (vs opponent)
   - Exploration bonuses

5. **Multi-agent variant**
   - Both players as learning agents
   - Competitive and cooperative modes
   - Communication protocols

## Testing

No automated test suite is committed (no `pytest`/`unittest` files). `docs/example_usage.py` is
runnable example code, not assertions - see `PROJECT_SUMMARY.md`'s Testing Status section for what
was actually verified during development and wasn't persisted as tests. It demonstrates:
- Environment creation and reset
- Single and multi-action execution
- Resource constraints
- Opponent AI
- Custom opponent behavior
- Edge cases (out of bounds, invalid actions)
- Observation shape and channels
- Scoring and connection logic

Run from anywhere: `python3 docs/example_usage.py` (it inserts the project root onto `sys.path` itself, so no `PYTHONPATH`/cwd juggling is needed)

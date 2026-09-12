# Railroad Tycoon - Gymnasium RL Environment

A gymnasium-compatible RL environment for the CodinGame Summer Challenge 2026. Two players place train tracks on a grid to connect towns and score points while sabotaging each other's efforts.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from railroad_env import RailroadGymEnv

env = RailroadGymEnv()  # keep render_mode=None unless running in a plain script (see Rendering section)
obs, info = env.reset()

for step in range(100):
    actions = {"actions": [
        (1, (5, 10)),  # PLACE track at (5, 10)
        (2, (3, 0)),   # DISRUPT region 3
    ]}
    obs, reward, terminated, truncated, info = env.step(actions)
    
    if terminated:
        break

env.close()
```

## Environment Overview

### Action Space

Actions are provided as a dictionary with key `"actions"` containing a list of `(action_type, (param1, param2))`
tuples. This fixed shape matches `env.action_space` exactly (`Discrete(3)` paired with a 2-element `Box`), so
`env.action_space.sample()` always produces something `env.step()` accepts.

**Action Types:**
- `(0, (0, 0))` - WAIT (do nothing, params ignored)
- `(1, (x, y))` - PLACE track at position (x, y)
  - Cost: 1-3 paint points (depends on terrain)
  - Terrain 0 (plains): 1 point
  - Terrain 1 (river): 2 points
  - Terrain 2 (mountain): 3 points
- `(2, (region_id, 0))` - DISRUPT region (2nd param unused)
  - Cost: 1 disruption point per turn
  - Increases region instability by 1
  - Region is inked out (destroyed) when instability ≥ 4

**Multi-action Execution:**
Actions are executed in order. The environment tries to execute each action with available resources:
- 3 paint points per turn
- 1 disruption point per turn
- Impossible actions are skipped
- Once resources are exhausted, remaining actions are skipped

### Observation Space

The observation is a numpy array of shape `(height, width, 28)`. Most channels are in `[0, 1]`,
but **channels 10-21 and 26 are signed** — see the table. Channels 5-9 are per-region scalars
broadcast to every cell belonging to that region. Neutral (owner 2) tracks count toward *both*
enemy and own in those channels.

| Channel | Description | Values |
|---------|-------------|--------|
| 0-2 | Terrain one-hot (plains, river, mountain) | 0.0 or 1.0 |
| 3 | Enemy track on this cell | 0.0 or 1.0 |
| 4 | Own track on this cell | 0.0 or 1.0 |
| 5 | Enemy tracks in this cell's region, **count / 10** | [0, ~1] |
| 6 | Own tracks in this cell's region, **count / 10** | [0, ~1] |
| 7 | This region's size, cells/total_map_cells | [0, 1] |
| 8 | Enemy tracks in region on an active connection, **count / 10** | [0, ~1] |
| 9 | Own tracks in region on an active connection, **count / 10** | [0, ~1] |
| 10-21 | One channel per town id slot (0-11), route guide | **-1.0, -0.5, 0.0 or +1.0** |
| 22 | Region instability / 4 | [0, 1] |
| 23 | Region inked out | 0.0 or 1.0 |
| 24 | Cell is part of an active (scoring) connection | 0.0 or 1.0 |
| 25 | A town stands here (ownerless, never buildable) | 0.0 or 1.0 |
| 26 | (own − enemy score) / 10000, same value in every cell | **signed** |
| 27 | turn / 100, same value in every cell | [0, 1] |

Channels 5/6/8/9 are **counts, not densities** — a deliberate reversal. What decides whether a
region gets inked is an *absolute* lead (enemy − own), so dividing by region size destroyed the
deciding quantity. The divisor 10 is a fixed scale, not a normalizer. Channels 26/27 are flat
planes because a convolution is local and has no other way to see a board-wide scalar.

**Channels 10-21** are the interesting one: channel `10 + town_id` marks the shortest
*open-terrain* route from that town to each of its `desired_connections` targets (union of all its
targets' routes, if it has more than one). The route is **signed**: `-1.0` at the town the channel
belongs to, `-0.5` at a town it wants to reach, and `+1.0` on the cells between. Towns being
negative and the corridor positive lets a single 3x3 filter tell endpoints from path. Beware when
plotting these — an `imshow` with `vmin=0` clamps both town values onto the background and makes
the towns invisible. "Open-terrain" means the route ignores
track/build state entirely - it's computed the same way whether zero tracks exist or the towns are
already fully connected, only inked regions block it, and any track already sitting on the route
(either owner's, or neutral) doesn't divert or block it either. This is a "build along this line"
guide, not a check of what's already built - contrast with channel 24, which only lights up once a
connection is actually completed. A town id ≥ the map's actual town count leaves its channel all
zero (e.g. an 8-town map never touches channels 18-21).

Channels 10-21 tell you *where* a route goes, but not *which* town owns each route or who its
targets are - `info['towns']` (returned by `reset()`/`step()`) gives the full
`{id, x, y, desired_connections}` list for that.

### Reward

The reward equals the actual points scored by player 0 that turn (i.e. `scores[0]` delta) - 1 point for
each track cell player 0 owns for every active connection passing through it, matching the game's real
scoring rule exactly (a cell shared by two active connections scores twice).

### `info` dict

Returned by both `reset()` and `step()`:
- `scores`, `turn`, `paint_points`, `disruption_points` - straightforward state
- `towns` - full `{id, x, y, desired_connections}` list (see the channels 10-21 note above)
- `player_actions` / `opponent_actions` - `{"placed": [(x, y), ...], "disrupted": [region_id, ...]}`,
  what each side actually got applied that turn (a "placed" cell can still end up neutral if both
  players claimed it - see the collision rule under Placing Tracks below)
- `window_open` - `True` unless rendering and the pygame window has been closed; a driving loop
  should stop stepping once this goes `False`

### Termination

Episodes terminate after 100 turns.

## Configuration

### Opponent Strategies

**Random Opponent** (default)
```python
env = RailroadGymEnv(opponent_strategy="random", opponent_seed=42)
```

**Greedy Opponent**
```python
env = RailroadGymEnv(opponent_strategy="greedy")
```

**Custom Opponent**
```python
def my_strategy(game_state):
    """
    Returns a list of actions for player 1 (opponent).
    
    Args:
        game_state: GameState object with current game state
    
    Returns:
        List of actions: [("PLACE", x, y), ("DISRUPT", region_id), ("WAIT",), ...]
    """
    return [("WAIT",)]

env = RailroadGymEnv()
obs, info = env.reset()
env.set_opponent_behavior(my_strategy)
```

### Rendering

**Headless Mode (default)**
```python
env = RailroadGymEnv()
```

**Visual Mode with Pygame**
```python
env = RailroadGymEnv(render_mode="human", cell_size=20)
env.reset()
for step in range(100):
    actions = {"actions": [(0, (0, 0))]}
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated:
        break
env.close()
```

### Map Configuration

```python
env = RailroadGymEnv(
    width=25,          # Grid width
    height=17,         # Grid height
    num_towns=8,       # Number of towns
    seed=42,           # Random seed for map generation
)
```

## Game Rules Summary

### Towns and Regions
- A region is a contiguous group of cells (any mix of terrain), used only for disruption - it has
  no cost effect on its own
- Each region can have **at most 1 town**, and never in a region adjacent to another region that
  already has a town - towns are naturally spread apart on the map
- Every town is guaranteed to be the target of at least one other town's desired connection

### Placing Tracks
- Connect towns with train tracks to score points
- Each town has a list of desired connections (unilateral)
- Tracks must connect via shortest path (with direction priority: N > E > S > W)
- Score 1 point per track piece you own, per active connection passing through it
- If both players place at the same empty cell in the same turn, it becomes a neutral (owner 2)
  track and both are charged for it - this is resolved by collecting both players' PLACE actions
  against the turn-start board before applying either

### Disruption
- Increase region instability by 1 per disruption point
- At instability ≥ 4, region is inked out (all tracks destroyed)
- Blocks future placements in that region
- All PLACE actions (both players) resolve before any DISRUPT action, every turn

### Victory Conditions
- Have the most points after 100 turns
- Be in the lead if all desired connections become impossible (not implemented as early
  termination in this environment - episodes always run the full 100 turns)

## API Reference

### RailroadGymEnv

#### `__init__`
```python
RailroadGymEnv(
    width: int = 25,
    height: int = 17,
    num_towns: int = 8,
    opponent_strategy: str = "random",
    opponent_seed: int = None,
    seed: int = None,
    render_mode: Optional[str] = None,
    cell_size: int = 20,
)
```

#### `reset(seed=None, options=None) -> (obs, info)`
Reset the environment and return initial observation and info.

#### `step(action) -> (obs, reward, terminated, truncated, info)`
Execute one step in the environment.

#### `render() -> bool`
Draws one frame (only if render_mode="human"). Returns `False` once the window has been closed -
prefer checking `info["window_open"]` from `step()` in a driving loop instead of calling this directly.

#### `close()`
Close the environment and clean up resources.

#### `set_opponent_behavior(behavior_fn)`
Set a custom opponent strategy function.

## Game State API

The `GameState` class provides access to the current game state:

```python
game_state = env.game_state

# Properties
game_state.width              # Map width
game_state.height             # Map height
game_state.turn               # Current turn (0-100)
game_state.scores             # [player0_score, player1_score]
game_state.tracks             # (height, width) array of track owners
game_state.paint_points       # [player0_points, player1_points]
game_state.disruption_points  # [player0_points, player1_points]
game_state.inked_regions      # Set of region IDs that are inked out

# Methods
game_state.get_terrain(x, y)           # Get terrain type at (x, y)
game_state.get_region(x, y)            # Get region ID at (x, y)
game_state.get_observation()           # Get current observation tensor
game_state.calculate_active_connections()  # Get active connections
game_state.place_track(x, y, player_id)    # Try to place a track
game_state.disrupt_region(region_id, player_id)  # Try to disrupt region
```

## Examples

- `example_usage.py` - importable snippets (custom opponent, action format, etc.)
- `play_with_rendering.py` - standalone terminal script that watches a full game with pygame

## Notes

- The environment is deterministic for a given seed
- The opponent's random seed is separate from the map generation seed
- Action execution stops when resources are exhausted; invalid actions are silently skipped
- The observation includes both players' states; the agent must learn to distinguish them from the channels
- Region generation (8-25 cells per region, grown via randomized BFS) is a synthetic generator built
  for this environment, not the real CodinGame map generator - see `ARCHITECTURE.md` for how it works


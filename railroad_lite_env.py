"""RailroadLiteEnv - a deliberately simplified view of the faithful game, as a Gym wrapper.

The full env (`railroad_env`) is a 1:1 port of the original referee and stays untouched. This
wrapper strips it down to the part that matters for learning to connect towns:

  * **Plains only.** Every cell's terrain is flattened after generation, so every track costs 1
    paint point. With the unchanged 3-point budget that is exactly 3 placements per turn.
  * **No disruption.** DISRUPT is removed from both players' action sets, so no region ever
    destabilises and nothing is ever destroyed.
  * **25 turns** instead of 100.
  * **Opponent: half-idle random.** Each turn it flips a coin - WAIT, or spend its whole budget
    on random legal placements. It never disrupts.
  * **16 observation channels** instead of 28 - the ones that still carry information here.

Map generation, town placement, desired connections, connection pathfinding and scoring are all
still the real thing.
"""
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from railroad_env import constants as C
from railroad_env.environment import RailroadGymEnv
from railroad_env.game_state import GameState
from railroad_env.grid import get_rail_cost
from railroad_env.opponent import OpponentStrategy

LITE_MAX_TURNS = 25

# Which of GameState's 28 channels survive. Dropped: the three terrain one-hots (everything is
# plains now), every region-aggregate channel (regions no longer matter with disruption gone),
# and instability (always 0). `region inked` is kept only because the network canvas marks its
# off-board padding as inked - it is the padding indicator.
LITE_CHANNELS = (
    3,                                  # enemy track present
    4,                                  # own track present
    *range(GameState.TOWN_CHANNEL_START,
           GameState.TOWN_CHANNEL_START + GameState.NUM_TOWN_CHANNELS),  # 10-21 town route guides
    GameState.INKED_CHANNEL,            # 23 - in practice: off-board padding
    GameState.ACTIVE_CONNECTION_CHANNEL,  # 24
)
NUM_LITE_CHANNELS = len(LITE_CHANNELS)  # 16

LITE_CHANNEL_NAMES = (
    "enemy track present",
    "own track present",
    *(f"town {i} route guide" for i in range(GameState.NUM_TOWN_CHANNELS)),
    "off-board / inked",
    "active connection",
)
assert len(LITE_CHANNEL_NAMES) == NUM_LITE_CHANNELS


class LiteHalfIdleOpponent(OpponentStrategy):
    """Flips a coin each turn: WAIT, or spend the whole paint budget on random legal placements.
    Never disrupts (the lite game has no disruption at all)."""

    def __init__(self, seed: int = None, wait_probability: float = 0.5):
        super().__init__(seed)
        self.wait_probability = wait_probability

    def get_actions(self, state: GameState, player_id: int) -> List[tuple]:
        if self.rng.random_sample() < self.wait_probability:
            return [("WAIT",)]

        actions = []
        budget = state.paint_points[player_id]
        placements = [
            ((x, y), get_rail_cost(state.grid.cells[(x, y)]))
            for y in range(state.height)
            for x in range(state.width)
            if state._is_placeable(x, y)
        ]

        while placements:
            affordable = [(c, cost) for c, cost in placements if cost <= budget]
            if not affordable:
                break
            coord, cost = affordable[self.rng.randint(len(affordable))]
            actions.append(("PLACE", coord[0], coord[1]))
            budget -= cost
            placements = [(c, k) for c, k in placements if c != coord]

        return actions or [("WAIT",)]


def flatten_terrain(game_state: GameState) -> None:
    """Turn every cell into plains, so all track costs 1. Applied after generation, which is why
    towns still sit where the real generator put them (it placed them on plains anyway)."""
    for tile in game_state.grid.cells.values():
        tile.type = C.TYPE_GRASS
    game_state.terrain[:] = C.TYPE_GRASS


def lite_observation(obs: np.ndarray) -> np.ndarray:
    """(h, w, 25) full observation -> (h, w, 16) lite observation."""
    return np.ascontiguousarray(obs[:, :, list(LITE_CHANNELS)])


class RailroadLiteEnv(gym.Wrapper):
    def __init__(
        self,
        grid_height: Optional[int] = None,
        max_turns: int = LITE_MAX_TURNS,
        opponent_wait_probability: float = 0.5,
        opponent_seed: Optional[int] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        cell_size: int = 20,
    ):
        env = RailroadGymEnv(
            grid_height=grid_height,
            max_turns=max_turns,
            # Replaced with LiteHalfIdleOpponent on every reset; "passive" just keeps the inner
            # env from building a full-game opponent we would throw away.
            opponent_strategy="passive",
            seed=seed,
            render_mode=render_mode,
            cell_size=cell_size,
        )
        super().__init__(env)

        self.opponent_wait_probability = opponent_wait_probability
        self.opponent_seed = opponent_seed
        self._opponent_rng = np.random.RandomState(seed)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(C.MAX_GRID_HEIGHT, C.MAX_GRID_WIDTH, NUM_LITE_CHANNELS),
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        flatten_terrain(self.env.game_state)

        opponent_seed = (
            self.opponent_seed
            if self.opponent_seed is not None
            else int(self._opponent_rng.randint(0, 2**31 - 1))
        )
        self.env.opponent = LiteHalfIdleOpponent(
            seed=opponent_seed, wait_probability=self.opponent_wait_probability
        )

        return lite_observation(self.env.game_state.get_observation()), info

    def step(self, action: Dict[str, Any]):
        actions = [a for a in action.get("actions", []) if not _is_disrupt(a)]
        obs, reward, terminated, truncated, info = self.env.step({"actions": actions})
        return lite_observation(obs), reward, terminated, truncated, info

    # Explicit rather than relying on gym.Wrapper's attribute forwarding, since the training
    # stack reaches for these constantly.
    @property
    def game_state(self) -> GameState:
        return self.env.game_state

    @property
    def opponent(self) -> OpponentStrategy:
        return self.env.opponent

    def legal_placements(self):
        return self.env.legal_placements()


def _is_disrupt(action) -> bool:
    """DISRUPT is silently dropped in the lite game - matching how the real referee skips
    impossible actions rather than erroring."""
    if not action:
        return False
    head = action[0]
    if isinstance(head, str):
        return head in ("DISRUPT", "DISRUPT_AT")
    return head == 2

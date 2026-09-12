"""GameSimulator: the sub-turn state machine RL agents act through.

A real turn is several PLACE_TRACKS (limited by the 3-point paint budget) followed by at most
one DISRUPT, all submitted together. A spatial policy head instead wants one atomic decision at
a time, so this accumulates a turn's actions one by one - recomputing legality against the board
*plus* whatever is already pending this turn - and only submits the whole turn to the underlying
GameState once the budget is exhausted.

Phase cascading matches the real resolution order: PLACE until nothing affordable remains, then
DISRUPT, then the turn resolves (which is when the opponent moves and scoring happens).
"""
import copy
from typing import List, Optional, Tuple

import numpy as np

from railroad_env import constants as C
from railroad_env.game_state import GameState
from railroad_env.opponent import OpponentStrategy

from .encoding import BOARD_HEIGHT, BOARD_WIDTH, compute_pad_offsets, encode_state

# ("PLACE", x, y), ("DISRUPT", x, y) or ("SKIP_DISRUPT", x, y) - all in real board coordinates.
# DISRUPT is addressed by cell (the game's own DISRUPT x y form) so it fits the same spatial
# policy head as PLACE.
Action = Tuple[str, int, int]

PLACE_CHANNEL = 0
DISRUPT_CHANNEL = 1
SKIP_DISRUPT_CHANNEL = 2
ACTION_KIND_CHANNEL = {
    "PLACE": PLACE_CHANNEL,
    "DISRUPT": DISRUPT_CHANNEL,
    "SKIP_DISRUPT": SKIP_DISRUPT_CHANNEL,
}

# The skip action needs somewhere to live in a spatial policy head. It gets a whole channel, and
# the mask marks EVERY on-board cell of it as legal - not a single slot. That matters: DISRUPT
# spans every cell of every disruptable region (typically a few hundred), so a lone skip cell
# would carry ~1/400th of the probability mass and an untrained policy would essentially never
# decline. Spreading it across the plane makes "skip" and "disrupt" comparable a priori; every
# cell of the channel decodes to the same action, so the action's probability is just the sum
# over its cells.
SKIP_DISRUPT_ACTION: Action = ("SKIP_DISRUPT", 0, 0)


class GameSimulator:
    def __init__(
        self,
        game_state: GameState,
        opponent: OpponentStrategy,
        allow_skip_disrupt: bool = False,
        reward_mode: str = "own",
        force_disrupt: bool = False,
    ):
        """`allow_skip_disrupt` adds a third policy channel holding a single "don't disrupt this
        turn" action, offered whenever a real disruption target exists. Without it the agent is
        forced to spend its disruption point every turn it legally can. Placement is always
        compulsory either way - the phase only ends once the paint budget can't buy anything,
        so the agent cannot decline to build.

        `reward_mode` picks what `last_turn_reward` reports:
          "own"    - points this player gained this turn
          "margin" - own gain minus the opponent's. Both players score on the SAME shared
                     connection, so this correctly discourages completing a route whose path is
                     mostly the opponent's track (which pays them more than it pays you).

        `force_disrupt` withholds SKIP_DISRUPT even though the action space still contains it,
        so the head keeps its third channel and a 3-channel checkpoint still loads. Setting
        `allow_skip_disrupt=False` instead would narrow the head to 2 and break that warm start.
        Why withhold it at all: disruption points are 1/turn and never carry over, so a declined
        disrupt destroys the resource outright, while its payoff (instability 1->4, then the ink)
        lands 4+ turns later and is invisible to a myopic discount. Measured, a good policy
        declines ~4% of the time and a collapsing one 48%, so the option costs far more than it
        earns."""
        if reward_mode not in ("own", "margin"):
            raise ValueError(f"reward_mode must be 'own' or 'margin', got {reward_mode!r}")
        # Independent from construction onward: nothing here mutates the env it came from.
        self.allow_skip_disrupt = allow_skip_disrupt
        self.force_disrupt = force_disrupt
        self.reward_mode = reward_mode
        self.game_state, self.opponent = copy.deepcopy((game_state, opponent))
        self._pending_place: List[Tuple[int, int]] = []
        self._pending_disrupt: List[int] = []
        self._paint_left = self.game_state.paint_points[0]
        self._disrupt_left = self.game_state.disruption_points[0]
        self._phase = "place"
        self._done = self.game_state.is_done()
        self.last_turn_reward: Optional[float] = None
        self._legal_actions_cache: List[Action] = []
        if not self._done:
            self._maybe_advance()

    @classmethod
    def from_env(cls, env, **kwargs) -> "GameSimulator":
        return cls(env.game_state, env.opponent, **kwargs)

    def clone(self) -> "GameSimulator":
        # type(self), not GameSimulator, so subclasses (e.g. LiteGameSimulator) clone as
        # themselves rather than silently degrading to the base behaviour.
        new_sim = type(self).__new__(type(self))
        new_sim.allow_skip_disrupt = self.allow_skip_disrupt
        new_sim.force_disrupt = self.force_disrupt
        new_sim.reward_mode = self.reward_mode
        new_sim.game_state, new_sim.opponent = copy.deepcopy((self.game_state, self.opponent))
        new_sim._pending_place = list(self._pending_place)
        new_sim._pending_disrupt = list(self._pending_disrupt)
        new_sim._paint_left = self._paint_left
        new_sim._disrupt_left = self._disrupt_left
        new_sim._phase = self._phase
        new_sim._done = self._done
        new_sim.last_turn_reward = None
        # Safe to share: the list is always reassigned, never mutated in place.
        new_sim._legal_actions_cache = self._legal_actions_cache
        return new_sim

    # ------------------------------------------------------------------ queries

    @property
    def policy_channels(self) -> int:
        return 3 if self.allow_skip_disrupt else 2

    @property
    def pad_offsets(self) -> Tuple[int, int]:
        return compute_pad_offsets(self.game_state.height, self.game_state.width)

    def is_game_over(self) -> bool:
        return self._done

    def get_encoded_state(self) -> np.ndarray:
        return encode_state(self.game_state)

    def get_legal_actions(self) -> List[Action]:
        return self._legal_actions_cache

    def get_action_mask(self) -> np.ndarray:
        """(policy_channels, BOARD_HEIGHT, BOARD_WIDTH): channel 0 = legal PLACE cells, channel 1
        = legal DISRUPT cells, channel 2 (only when allow_skip_disrupt) = the single "skip
        disrupting" slot. Only the channel(s) for the current phase are ever populated, since
        PLACE must fully resolve before DISRUPT within a turn."""
        mask = np.zeros((self.policy_channels, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        if self._done:
            return mask

        pad_top, pad_left = self.pad_offsets
        for kind, x, y in self._legal_actions_cache:
            if kind == "SKIP_DISRUPT":
                # Whole on-board plane - see SKIP_DISRUPT_ACTION's comment.
                mask[
                    SKIP_DISRUPT_CHANNEL,
                    pad_top:pad_top + self.game_state.height,
                    pad_left:pad_left + self.game_state.width,
                ] = 1.0
            else:
                mask[ACTION_KIND_CHANNEL[kind], pad_top + y, pad_left + x] = 1.0
        return mask

    def _compute_legal(self, phase: str) -> List[Action]:
        if self._done:
            return []

        gs = self.game_state
        actions: List[Action] = []

        if phase == "place":
            if self._paint_left <= 0:
                return []
            pending = set(self._pending_place)
            for y in range(gs.height):
                for x in range(gs.width):
                    if (x, y) in pending:
                        continue
                    if gs._is_placeable(x, y) and gs.get_track_cost(x, y) <= self._paint_left:
                        actions.append(("PLACE", x, y))

        elif phase == "disrupt":
            if self._disrupt_left <= 0:
                return []
            for zone in gs.zones:
                if not gs.can_disrupt(zone.id):
                    continue
                for (x, y) in zone.coords:
                    actions.append(("DISRUPT", x, y))
            # Only worth offering when there is something to decline.
            if actions and self.allow_skip_disrupt and not self.force_disrupt:
                actions.append(SKIP_DISRUPT_ACTION)

        return actions

    # ------------------------------------------------------------------ mutation

    def apply(self, action: Action) -> None:
        if self._done:
            raise RuntimeError("apply() called after the game has ended")

        kind, x, y = action
        gs = self.game_state

        if self._phase == "place":
            if kind != "PLACE":
                raise ValueError(f"expected PLACE action during place phase, got {action}")
            if (x, y) in self._pending_place or not gs._is_placeable(x, y):
                raise ValueError(f"illegal PLACE action: {action}")
            cost = gs.get_track_cost(x, y)
            if cost > self._paint_left:
                raise ValueError(f"unaffordable PLACE action: {action}")
            self._pending_place.append((x, y))
            self._paint_left -= cost

        elif self._phase == "disrupt":
            if kind == "SKIP_DISRUPT":
                if not self.allow_skip_disrupt:
                    raise ValueError("SKIP_DISRUPT used but allow_skip_disrupt is False")
                self._disrupt_left = 0  # ends the phase, resolving the turn without disrupting
                self._maybe_advance()
                return
            if kind != "DISRUPT":
                raise ValueError(f"expected DISRUPT action during disrupt phase, got {action}")
            zone_id = gs.get_region(x, y)
            if not gs.can_disrupt(zone_id):
                raise ValueError(f"illegal DISRUPT action: {action}")
            self._pending_disrupt.append(zone_id)
            self._disrupt_left -= 1

        else:
            raise RuntimeError("apply() called with no active phase")

        self._maybe_advance()

    def _maybe_advance(self) -> None:
        """Skip a phase once it has no legal moves left, cascading all the way to real turn
        resolution if both are exhausted. Always ends by repopulating the legal-action cache."""
        legal = self._compute_legal(self._phase)
        if self._phase == "place" and not legal:
            self._phase = "disrupt"
            legal = self._compute_legal("disrupt")
        if self._phase == "disrupt" and not legal:
            self._resolve_turn()
            return
        self._legal_actions_cache = legal

    def _resolve_turn(self) -> None:
        """Submits the accumulated turn to the real engine - which is where the opponent moves,
        zones ink out, and scoring happens."""
        gs = self.game_state
        actions = [("PLACE", x, y) for (x, y) in self._pending_place]
        actions += [("DISRUPT", zid) for zid in self._pending_disrupt]
        if not actions:
            actions = [("WAIT",)]

        opponent_actions = self.opponent.get_actions(gs, 1)
        summary = gs.resolve_turn([actions, opponent_actions])
        delta = summary["score_delta"]
        self.last_turn_reward = (
            float(delta[0] - delta[1]) if self.reward_mode == "margin" else float(delta[0])
        )

        gs.do_income()
        self._pending_place = []
        self._pending_disrupt = []
        self._paint_left = gs.paint_points[0]
        self._disrupt_left = gs.disruption_points[0]
        self._phase = "place"
        self._done = gs.is_done()

        if self._done:
            self._legal_actions_cache = []
        else:
            self._maybe_advance()

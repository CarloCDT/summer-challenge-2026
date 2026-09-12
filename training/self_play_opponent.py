"""A network-driven opponent for player 1, so the agent can train against itself.

Kept out of `railroad_env/` deliberately: that package is torch-free and is shipped standalone
in /home/carlo/claude_env, so the only place a model may be imported is here under `training/`.

The turn is played exactly the way the baked submission plays it - ONE forward pass, then the
affordable PLACE cells taken greedily by logit, then the disrupt decision off the same pass.
That matters: if self-play used a stronger procedure than deployment (re-encoding between
placements, say) the agent would be trained against an opponent it can never actually be.
"""
import copy
from typing import List

import numpy as np
import torch

from railroad_env import constants as C
from railroad_env.game_state import Action, GameState
from railroad_env.opponent import OpponentStrategy

from .encoding import BOARD_HEIGHT, BOARD_WIDTH, compute_pad_offsets, encode_state
from .simulator import ACTION_KIND_CHANNEL, SKIP_DISRUPT_CHANNEL

PLACE_CHANNEL = ACTION_KIND_CHANNEL["PLACE"]
DISRUPT_CHANNEL = ACTION_KIND_CHANNEL["DISRUPT"]


class SelfPlayOpponent(OpponentStrategy):
    """Player 1 driven by `model`, reading the board from player 1's perspective.

    `epsilon` samples from the policy instead of taking its argmax that fraction of the time,
    purely to keep self-play games from collapsing into one deterministic line - it is the
    opponent's exploration, unrelated to the agent's own action selection.
    """

    def __init__(self, model, seed: int = None, epsilon: float = 0.0,
                 allow_skip_disrupt: bool = True, force_disrupt: bool = False):
        super().__init__(seed)
        self.model = model
        self.epsilon = epsilon
        self.allow_skip_disrupt = allow_skip_disrupt
        self.force_disrupt = force_disrupt

    def __deepcopy__(self, memo):
        """GameSimulator deep-copies (game_state, opponent) on construction. The model is only
        ever read during rollout, so share it rather than cloning ~8.8 MB of weights per game."""
        clone = SelfPlayOpponent.__new__(SelfPlayOpponent)
        clone.rng = copy.deepcopy(self.rng, memo)
        clone.model = self.model
        clone.epsilon = self.epsilon
        clone.allow_skip_disrupt = self.allow_skip_disrupt
        clone.force_disrupt = self.force_disrupt
        return clone

    def _static_masks(self, state: GameState):
        """(terrain cost, town mask), rebuilt only when the grid itself changes.

        Keyed on the grid object rather than its id() - holding the reference keeps it alive,
        so identity stays meaningful for as long as the cache does."""
        if getattr(self, "_static_grid", None) is not state.grid:
            h, w = state.height, state.width
            self._static_grid = state.grid
            self._static_cost = np.array(
                [state.get_track_cost(x, y) for y in range(h) for x in range(w)],
                dtype=np.int32,
            ).reshape(h, w)
            towns = np.zeros((h, w), dtype=bool)
            for town in state.towns:
                towns[town.coord[1], town.coord[0]] = True
            self._static_towns = towns
        return self._static_cost, self._static_towns

    def _policy_planes(self, state: GameState, player_id: int) -> np.ndarray:
        """One forward pass -> (3, BOARD_HEIGHT, BOARD_WIDTH) logits, unmasked."""
        obs = encode_state(state, player=player_id)
        # Legality is applied per decision below, so feed an all-ones mask and let the caller
        # restrict; masking here would need the mask rebuilt between placements anyway.
        ones = np.ones((3, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        with torch.no_grad():
            logits, _ = self.model(
                torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(ones).unsqueeze(0)
            )
        return logits[0].cpu().numpy()

    def _pick(self, plane: np.ndarray, legal: np.ndarray) -> int:
        """Argmax over the legal cells of a flattened plane, or a sample with prob `epsilon`."""
        scores = np.where(legal, plane, -np.inf).ravel()
        if self.epsilon > 0.0 and self.rng.random_sample() < self.epsilon:
            finite = np.isfinite(scores)
            p = np.exp(scores[finite] - scores[finite].max())
            return int(np.flatnonzero(finite)[self.rng.choice(len(p), p=p / p.sum())])
        return int(np.argmax(scores))

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        planes = self._policy_planes(state, player_id)
        pad_top, pad_left = compute_pad_offsets(state.height, state.width)
        h, w = state.height, state.width
        actions: List[Action] = []

        # Rollout is ~93% of an iteration, so the legality scan is vectorised rather than a
        # per-cell Python loop. Terrain cost and town positions are fixed for the whole game,
        # so they are built once per grid instead of once per turn - the naive version cost
        # ~60k redundant get_track_cost calls per game. Only the inked set actually changes.
        cost, is_town = self._static_masks(state)
        inked = np.array([z.inked for z in state.zones], dtype=bool)[state.regions]
        buildable = (state.tracks == C.TRACK_NONE) & ~is_town & ~inked

        place = planes[PLACE_CHANNEL, pad_top:pad_top + h, pad_left:pad_left + w]
        paint = state.paint_points[player_id]
        while paint > 0:
            legal = buildable & (cost <= paint)
            if not legal.any():
                break
            y, x = divmod(self._pick(place, legal), w)
            actions.append(("PLACE", int(x), int(y)))
            paint -= int(cost[y, x])
            buildable[y, x] = False

        if state.disruption_points[player_id] > 0:
            disrupt = planes[DISRUPT_CHANNEL, pad_top:pad_top + h, pad_left:pad_left + w]
            disruptable = np.array([state.can_disrupt(z.id) for z in state.zones], dtype=bool)
            legal = disruptable[state.regions]
            if legal.any():
                # The skip plane spans the whole board, so compare its best on-board cell
                # against the best disrupt target - the same contest the baked agent runs.
                skip = planes[SKIP_DISRUPT_CHANNEL, pad_top:pad_top + h, pad_left:pad_left + w]
                may_skip = self.allow_skip_disrupt and not self.force_disrupt
                if not (may_skip and skip.max() > np.where(legal, disrupt, -np.inf).max()):
                    y, x = divmod(self._pick(disrupt, legal), w)
                    actions.append(("DISRUPT", int(state.get_region(int(x), int(y)))))

        return actions or [("WAIT",)]

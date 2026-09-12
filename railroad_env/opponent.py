"""Opponent policies for player 1.

`passive` and `boss` are the two AIs the real challenge actually ships (config/Boss.py and
config/level2/Boss.py respectively); the others are extra baselines for RL training.
"""
import copy
import inspect
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

from . import constants as C
from .game_state import Action, GameState
from .grid import get_rail_cost
from .pathfinding import autobuild
from .wire import frame_lines, init_lines, parse_actions


class OpponentStrategy:
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        raise NotImplementedError


class PassiveOpponent(OpponentStrategy):
    """Always WAITs - the challenge's own default Boss (config/Boss.py) and its League 1 Boss."""

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        return [("WAIT",)]


class BossOpponent(OpponentStrategy):
    """The challenge's League 2 Boss (config/level2/Boss.py): every turn, AUTOPLACE between two
    randomly chosen towns."""

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        towns = state.towns
        if len(towns) < 2:
            return [("WAIT",)]
        town_a = towns[self.rng.randint(len(towns))]
        others = [t for t in towns if t.id != town_a.id]
        town_b = others[self.rng.randint(len(others))]
        return [("AUTOPLACE", town_a.coord[0], town_a.coord[1], town_b.coord[0], town_b.coord[1])]


class RandomOpponent(OpponentStrategy):
    """Spends its paint budget on random legal placements, then optionally disrupts a random
    legal region. Not one of the shipped bosses - a baseline for training.

    Two independent per-turn coin flips: `wait_probability` skips placing entirely that turn,
    and `disrupt_probability` decides whether to spend the disruption point. Defaults reproduce
    the original behaviour (always place, disrupt half the time)."""

    def __init__(
        self,
        seed: int = None,
        wait_probability: float = 0.0,
        disrupt_probability: float = 0.5,
    ):
        super().__init__(seed)
        self.wait_probability = wait_probability
        self.disrupt_probability = disrupt_probability

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        actions: List[Action] = []
        budget = state.paint_points[player_id]

        placements = [] if self.rng.random_sample() < self.wait_probability else _legal_placements(state)
        while placements:
            affordable = [(c, cost) for c, cost in placements if cost <= budget]
            if not affordable:
                break
            coord, cost = affordable[self.rng.randint(len(affordable))]
            actions.append(("PLACE", coord[0], coord[1]))
            budget -= cost
            placements = [(c, k) for c, k in placements if c != coord]

        if state.disruption_points[player_id] > 0 and self.rng.random_sample() < self.disrupt_probability:
            disruptable = [z.id for z in state.zones if state.can_disrupt(z.id)]
            if disruptable:
                actions.append(("DISRUPT", int(disruptable[self.rng.randint(len(disruptable))])))

        return actions or [("WAIT",)]


class GreedyAutoplaceOpponent(OpponentStrategy):
    """Builds toward whichever desired connection is cheapest to finish, using AUTOPLACE.

    A stronger relative of the shipped League 2 Boss: where that one picks two *random* towns
    every turn, this one prices every still-unfinished desired connection with the same
    cheapest-path search AUTOPLACE itself uses, and commits to the cheapest. That means it
    finishes connections instead of scattering half-built routes, and it naturally prefers
    routes over plains to routes over mountains.

    `disrupt_probability` defaults to 0 - it is a pure builder unless asked otherwise."""

    def __init__(self, seed: int = None, disrupt_probability: float = 0.0):
        super().__init__(seed)
        self.disrupt_probability = disrupt_probability

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        actions: List[Action] = []

        best_pair, best_cost = None, None
        for town in state.towns:
            for other in town.desired_connections:
                if other.id in town.paths:
                    continue  # already an active connection - nothing left to build
                chain = autobuild(state.grid, town.coord, other.coord)
                if not chain:
                    continue
                cost = sum(get_rail_cost(state.grid.cells[c]) for c in chain)
                if best_cost is None or cost < best_cost:
                    best_pair, best_cost = (town, other), cost

        if best_pair is not None:
            a, b = best_pair
            actions.append(("AUTOPLACE", a.coord[0], a.coord[1], b.coord[0], b.coord[1]))

        if state.disruption_points[player_id] > 0 and self.rng.random_sample() < self.disrupt_probability:
            enemy = 1 - player_id
            best_zone, best_count = None, 0
            for zone in state.zones:
                if not state.can_disrupt(zone.id):
                    continue
                count = sum(1 for (x, y) in zone.coords if state.tracks[y, x] == enemy)
                if count > best_count:
                    best_zone, best_count = zone.id, count
            if best_zone is not None:
                actions.append(("DISRUPT", int(best_zone)))

        return actions or [("WAIT",)]


class Level2ProMaxOpponent(GreedyAutoplaceOpponent):
    """level2Pro's builder, plus a deterministic disruptor aimed at the enemy's best region.

    Each turn it scores every disruptable region by (enemy tracks - own tracks) and inks the
    highest scorer, but only when that lead is strictly greater than 1. The gap matters: at a
    lead of exactly 1 the region is nearly contested, and inking it destroys the agent's own
    track there too (Game.doActions wipes the whole zone, both colours). Requiring 2 keeps it
    from trading its own rails away for a marginal gain.

    Neutral track (TRACK_NEUTRAL, a cell both players built on) is deliberately not counted:
    it belongs to both sides, so it contributes equally to each term and cancels out.

    Ties go to the lowest zone id, so the policy is fully deterministic given the board - unlike
    the `disrupt_probability` path it inherits, which is left switched off."""

    #: A region must hold at least this many more enemy tracks than own ones to be worth inking.
    DISRUPT_MIN_LEAD = 2

    def __init__(self, seed: int = None):
        # The parent's probabilistic disrupt stays off - this subclass decides deterministically.
        super().__init__(seed, disrupt_probability=0.0)

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        # Reuse the level2Pro builder, dropping its WAIT filler so a disrupt-only turn is not
        # emitted as "WAIT; DISRUPT".
        actions = [a for a in super().get_actions(state, player_id) if a[0] != "WAIT"]

        if state.disruption_points[player_id] > 0:
            enemy = 1 - player_id
            # Seeding best_lead at DISRUPT_MIN_LEAD - 1 makes the threshold and the argmax the
            # same comparison: nothing at or below the gap can ever win.
            best_zone, best_lead = None, self.DISRUPT_MIN_LEAD - 1
            for zone in state.zones:
                if not state.can_disrupt(zone.id):
                    continue
                lead = 0
                for (x, y) in zone.coords:
                    track = state.tracks[y, x]
                    if track == enemy:
                        lead += 1
                    elif track == player_id:
                        lead -= 1
                if lead > best_lead:
                    best_zone, best_lead = zone.id, lead
            if best_zone is not None:
                actions.append(("DISRUPT", int(best_zone)))

        return actions or [("WAIT",)]


class GreedyOpponent(OpponentStrategy):
    """Extends its own track where it already has neighbours, cheapest cells first, and disrupts
    whichever disruptable region holds the most enemy track."""

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        actions: List[Action] = []
        budget = state.paint_points[player_id]

        scored = []
        for coord, cost in _legal_placements(state):
            x, y = coord
            adjacency = 0
            for dx, dy in C.ADJACENCY:
                tile = state.grid.get(x + dx, y + dy)
                if tile is not None and (tile.track == player_id or tile.is_town()):
                    adjacency += 1
            scored.append((-adjacency, cost, coord))
        scored.sort()

        for _, cost, coord in scored:
            if budget >= cost:
                actions.append(("PLACE", coord[0], coord[1]))
                budget -= cost

        if state.disruption_points[player_id] > 0:
            enemy = 1 - player_id
            best_zone, best_count = None, 0
            for zone in state.zones:
                if not state.can_disrupt(zone.id):
                    continue
                count = sum(1 for (x, y) in zone.coords if state.tracks[y, x] == enemy)
                if count > best_count:
                    best_zone, best_count = zone.id, count
            if best_zone is not None:
                actions.append(("DISRUPT", int(best_zone)))

        return actions or [("WAIT",)]


class Level1ProOpponent(RandomOpponent):
    """A step up from the League 1 Boss (which never acts): each turn it flips two independent
    coins - half the time it skips placing entirely, and half the time it disrupts. Not one of
    the challenge's own AIs; this is the opponent training/configs/ppo_full.yaml uses."""

    def __init__(self, seed: int = None, wait_probability: float = 0.5, disrupt_probability: float = 0.5):
        super().__init__(seed, wait_probability=wait_probability, disrupt_probability=disrupt_probability)


def _legal_placements(state: GameState):
    """[( (x, y), cost ), ...] for every cell a track may legally be placed on right now."""
    placements = []
    for y in range(state.height):
        for x in range(state.width):
            if state._is_placeable(x, y):
                placements.append(((x, y), get_rail_cost(state.grid.cells[(x, y)])))
    return placements


# The last two "boss" tiers, and the strongest: our own trained agents played as opponents.
# level1/level2 are the challenge's own shipped AIs; every "Pro"/"Silver" tier is ours, and
# all of them are considerably stronger than anything the real ladder fields.
class BakedSubmissionOpponent(OpponentStrategy):
    """`level2Silver`: the baked submission file itself, played as a boss.

    This drives the real `submission.py` as a subprocess over the referee's wire protocol, and
    that is the whole point - it is the shipped agent, not an approximation of it. Loading the
    checkpoint into torch instead would NOT be the same player: submission.py is int4-quantized
    with BatchNorm folded, and it reproduces only ~62% of the torch model's argmax moves (see
    bake_agent.quantize). Identical score, different moves - so for a baseline that is supposed
    to be "what we are sending", only the file will do.

    Costs about 1.0s per 100-turn game, comparable to level2Pro's 0.92s, because the submission
    answers a turn in single-digit milliseconds by design.

    The process is spawned lazily on the first turn, so a GameSimulator that deep-copies this
    object at construction copies a clean, unstarted opponent. It is stateful across turns (it
    keeps its own turn counter and a route-channel cache), so it must see every frame of one
    game in order and can never be cloned mid-game - `__deepcopy__` refuses that explicitly
    rather than silently desyncing.
    """

    #: Subclasses pin a different file and a different tier name; everything else is shared.
    TIER = "level2Silver"
    DEFAULT_PATH = Path(__file__).resolve().parent.parent / "submission.py"

    def __init__(self, seed: int = None, submission_path: str = None):
        super().__init__(seed)
        self.submission_path = Path(submission_path) if submission_path else self.DEFAULT_PATH
        self._proc = None
        self._started = False

    def __deepcopy__(self, memo):
        if self._started:
            raise RuntimeError(
                f"{self.TIER} cannot be copied once its game has started - the "
                "subprocess holds turn state that cannot be forked. This opponent does not "
                "support GameSimulator.clone()/MCTS search."
            )
        clone = type(self).__new__(type(self))
        clone.rng = copy.deepcopy(self.rng, memo)
        clone.submission_path = self.submission_path
        clone._proc = None
        clone._started = False
        return clone

    def _start(self, state: GameState, player_id: int) -> None:
        if not self.submission_path.exists():
            raise FileNotFoundError(
                f"{self.TIER} needs a baked agent at {self.submission_path}. Bake one with "
                f"`python3 bake_agent.py <checkpoint> -o submission.py`, or pass "
                f"opponent_kwargs={{'submission_path': ...}}."
            )
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(self.submission_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._started = True
        self._write(init_lines(state, player_id))

    def _write(self, lines: List[str]) -> None:
        self._proc.stdin.write("\n".join(lines) + "\n")
        self._proc.stdin.flush()

    def get_actions(self, state: GameState, player_id: int) -> List[Action]:
        if not self._started:
            self._start(state, player_id)
        if self._proc is None:
            raise RuntimeError(
                f"{self.TIER} was asked for a turn after its game ended. The subprocess is "
                "closed on the turn that reaches max_turns and cannot be resumed - a fresh one "
                "would restart its internal turn counter and play a different game. Build a new "
                "opponent per game."
            )

        self._write(frame_lines(state, player_id))
        reply = self._proc.stdout.readline()
        if not reply:
            err = self._proc.stderr.read()
            self.close()
            raise RuntimeError(f"{self.submission_path.name} died mid-game:\n{err[-2000:]}")

        actions = parse_actions(reply)
        # The submission never sees the game end - it just stops being asked - so close on the
        # turn that resolves the last one rather than leaking a process per game.
        if state.turn + 1 >= state.max_turns:
            self.close()
        return actions or [("WAIT",)]

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def __del__(self):
        # Backstop for games that end early (is_done() before max_turns) or raise.
        try:
            self.close()
        except Exception:
            pass


class FrozenSubmissionOpponent(BakedSubmissionOpponent):
    """`level2SilverPro`: a submission we actually sent, kept byte-for-byte and never re-baked.

    Same mechanism as `level2Silver` - the file driven as a subprocess over the referee's wire
    protocol - and deliberately the opposite choice of file. `level2Silver` reads `submission.py`,
    which is live state, so it TRACKS whatever is currently baked and `eval/level2Silver/*` stops
    being comparable the moment you re-bake. This one reads a frozen copy under `baselines/`, so it
    is a fixed rung: a margin against it means the same thing next month as it does today.

    It defaults to the bake that reached **CodinGame rank ~288**, from teacher
    `20260911-114307_iter200.pt`, which makes it the one baseline in this repo with evidence from
    outside our own eval table. Note that is the same file `level2Silver` reads *today*, so the two
    tiers score identically until the next re-bake - verified over 12 seeds, every column equal -
    and after one, the gap between the columns is exactly what the re-bake changed.

    It is the FILE, not the checkpoint behind it, for the reason `level2Silver` is: the bake is
    int4 with BatchNorm folded and reproduces only ~62% of the torch model's argmax moves, so
    loading that checkpoint into torch would be a different player. What we want frozen is what we
    sent.

    See `baselines/README.md` for how to add another rung - a population of past selves is the
    "league" in CLAUDE.md's backlog, and this is its first member.
    """

    TIER = "level2SilverPro"
    DEFAULT_PATH = (
        Path(__file__).resolve().parent.parent
        / "baselines"
        / "20260911-144318-distill_iter50_rank288.py"
    )


BOSS_TIERS = ("level1", "level1Pro", "level2", "level2Pro", "level2ProMax", "level2Silver",
              "level2SilverPro")

OPPONENT_STRATEGIES = {
    # Faithful ports of the challenge's own Boss AIs
    "passive": PassiveOpponent,               # config/Boss.py, config/level1/Boss.py
    "level1": PassiveOpponent,                #   alias, for the boss-tier naming
    "boss": BossOpponent,                     # config/level2/Boss.py
    "level2": BossOpponent,                   #   alias
    # Ours - not part of the challenge
    "level1Pro": Level1ProOpponent,           # random placements half the time, disrupts half
    "level2Pro": GreedyAutoplaceOpponent,     # cheapest-path AUTOPLACE, always building
    "level2ProMax": Level2ProMaxOpponent,     #   the same, plus deterministic disruption
    "level2Silver": BakedSubmissionOpponent,  #   the baked submission.py itself, as a boss
    "level2SilverPro": FrozenSubmissionOpponent,  # the frozen rank-288 bake, never re-baked
    "random": RandomOpponent,
    "greedy": GreedyOpponent,
    "greedy_autoplace": GreedyAutoplaceOpponent,
}


def make_opponent(name: str, seed: int = None, **kwargs) -> OpponentStrategy:
    """Extra kwargs (e.g. wait_probability, disrupt_probability) are passed through to whichever
    strategy accepts them and silently ignored by the ones that don't, so callers can hand over
    one settings dict without knowing which opponent it is building."""
    if name not in OPPONENT_STRATEGIES:
        raise ValueError(
            f"Unknown opponent strategy: {name}. Available: {sorted(OPPONENT_STRATEGIES)}"
        )
    cls = OPPONENT_STRATEGIES[name]
    accepted = set(inspect.signature(cls.__init__).parameters)
    return cls(seed=seed, **{k: v for k, v in kwargs.items() if k in accepted})

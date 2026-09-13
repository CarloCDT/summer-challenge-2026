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


# The last and strongest "boss" tier: one of our own trained agents played as an opponent.
# level1/level2 are the challenge's own shipped AIs; every "Pro"/"Silver" tier is ours, and
# all of them are considerably stronger than anything the real ladder fields.
class BakedSubmissionOpponent(OpponentStrategy):
    """`level2Silver`: a bake we actually submitted, played as a boss.

    This drives a real baked agent as a subprocess over the referee's wire protocol, and that is
    the whole point - it is a shipped agent, not an approximation of one. Loading the checkpoint
    into torch instead would NOT be the same player: a bake is int4-quantized with BatchNorm
    folded and reproduces only ~62% of the torch model's argmax moves (see bake_agent.quantize).
    Identical score, different moves - so for a baseline that is supposed to be "what we sent",
    only the file will do.

    It reads a FROZEN copy under `baselines/`, NOT the live `submission.py`, and that distinction
    is the whole reason `baselines/` exists. This boss used to default to `submission.py`, which
    made it live state: re-baking silently changed the training opponent and the eval baseline of
    anything then running, and `eval/level2Silver/margin` stopped being comparable across the
    change. Pointed at a file nothing overwrites, the column means the same thing across re-bakes
    and across runs.

    The default is the bake that reached CodinGame rank ~288, the only agent in this repo with
    evidence from outside our own eval table. To raise the bar deliberately, copy the new bake into
    `baselines/` and change DEFAULT_PATH - one line, and it should stay a decision, because it moves
    every number this tier has ever produced. `submission_path` still overrides per instance, so
    `opponent_kwargs={'submission_path': 'submission.py'}` gets the old live behaviour back for a
    one-off comparison.

    Costs about 1.0s per 100-turn game, comparable to level2Pro's 0.92s, because a baked agent
    answers a turn in single-digit milliseconds by design.

    The process is spawned lazily on the first turn, so a GameSimulator that deep-copies this
    object at construction copies a clean, unstarted opponent. It is stateful across turns (it
    keeps its own turn counter and a route-channel cache), so it must see every frame of one
    game in order and can never be cloned mid-game - `__deepcopy__` refuses that explicitly
    rather than silently desyncing.
    """

    #: The tier this class is registered as, used in its error messages. Subclasses that pin a
    #: different baseline override it, so a failure names the rung that actually failed.
    TIER_NAME = "level2Silver"

    #: A frozen bake, never the live submission.py - see the class docstring.
    DEFAULT_PATH = (
        Path(__file__).resolve().parent.parent
        / "baselines"
        / "20260911-144318-distill_iter50_rank288.py"
    )

    def __init__(self, seed: int = None, submission_path: str = None):
        super().__init__(seed)
        self.submission_path = Path(submission_path) if submission_path else self.DEFAULT_PATH
        self._proc = None
        self._started = False

    def __deepcopy__(self, memo):
        if self._started:
            raise RuntimeError(
                f"{self.TIER_NAME} cannot be copied once its game has started - the "
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
                f"{self.TIER_NAME} needs a baked agent at {self.submission_path}. That is a frozen "
                f"baseline, so the fix is to restore the file (it is tracked in git), not to "
                f"re-bake it. To play a different agent as this boss, bake one with `python3 "
                f"bake_agent.py <checkpoint> -o <file>` and pass "
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
                f"{self.TIER_NAME} was asked for a turn after its game ended. The subprocess is "
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


class Level2Silver2Opponent(BakedSubmissionOpponent):
    """`level2Silver2`: the SECOND frozen rung, and the reason `baselines/` is a directory.

    Identical machinery to `level2Silver` - the same subprocess over the same wire protocol, the
    same statefulness, the same refusal to be deep-copied mid-game. The only difference is which
    frozen file it plays, and that is the whole idea: two rungs instead of one turn `baselines/`
    from an archive into a small league, so "is this agent better" can be asked against a
    population of past selves rather than against a single opponent the pool is 30% weighted to.

    Why a second tier rather than moving DEFAULT_PATH. Repointing `level2Silver` would retire the
    rank-288 bake, and that is the only agent in this repo with evidence from outside our own eval
    table - every number ever measured against that tier would silently change meaning. Adding a
    rung keeps both columns and costs one class.

    What this rung is NOT: held out. It is a bake of a student descended from the same lineage as
    everything we now train, so it measures progress against a fixed bar, not generalisation. Both
    rungs share that limitation - see CLAUDE.md's held-out-eval backlog item.

    Adding it to `BOSS_TIERS` lengthens any eval that does not pin `eval_bosses` by roughly another
    second per episode, on top of what `level2Silver` already costs.
    """

    TIER_NAME = "level2Silver2"

    #: The bake of `checkpoints/20260912-124725-distill_iter50.pt` - the first student distilled
    #: with `value_loss_weight`, off the win_bonus teacher `20260912-074038_iter350`. Verified
    #: byte-identical to the `submission.py` baked from that checkpoint on 2026-09-12.
    DEFAULT_PATH = (
        Path(__file__).resolve().parent.parent
        / "baselines"
        / "20260912-124725-distill_iter50.py"
    )


class Level2Silver3Opponent(BakedSubmissionOpponent):
    """`level2Silver3`: the THIRD frozen rung, and the current top of the league.

    Same machinery as the two rungs above it; only the frozen file differs. Added 2026-09-12 from
    the `submission.py` baked out of `20260912-210537-distill_iter50.pt` - a student distilled with
    `value_loss_weight`, so unlike the rank-288 bake its critic actually received a gradient.

    Adding rather than repointing, for the third time and for the same reason: each rung is the
    fixed meaning of every number ever measured against it. Retiring one silently rewrites history;
    adding one costs a class and keeps the ladder readable.

    None of the three rungs is HELD OUT - every one descends from the lineage this repo trains on,
    so the league measures progress against a fixed bar, not generalisation. `level2ProMax` is the
    held-out detector (see `ppo_28ch_vs_silver2.yaml`), not anything in `baselines/`.
    """

    TIER_NAME = "level2Silver3"

    #: The bake of `checkpoints/20260912-210537-distill_iter50.pt`. Verified byte-identical to the
    #: `submission.py` baked from that checkpoint on 2026-09-12 (87,348 chars, int4).
    DEFAULT_PATH = (
        Path(__file__).resolve().parent.parent
        / "baselines"
        / "20260912-210537-distill_iter50.py"
    )


class Level2Silver4Opponent(BakedSubmissionOpponent):
    """`level2Silver4`: the FOURTH frozen rung, added 2026-09-13 as the current best candidate.

    The bake of `20260913-113208-distill_iter50.pt`, distilled from teacher
    `20260913-065817_iter200.pt`. Same machinery as the rungs above; only the frozen file differs.

    Note the lineage when training against it: its teacher IS `065817_iter200`, so a PPO run warm
    started from that checkpoint and trained against this rung is playing the int4 distillation of
    its own starting weights - close to a mirror, but a FROZEN one, so unlike `self` it does not
    move with the agent.
    """

    TIER_NAME = "level2Silver4"

    #: Verified byte-identical to the `submission.py` baked from that checkpoint (87,420 chars,
    #: int4) on 2026-09-13.
    DEFAULT_PATH = (
        Path(__file__).resolve().parent.parent
        / "baselines"
        / "20260913-113208-distill_iter50.py"
    )


BOSS_TIERS = ("level1", "level1Pro", "level2", "level2Pro", "level2ProMax", "level2Silver",
              "level2Silver2", "level2Silver3", "level2Silver4")

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
    "level2Silver": BakedSubmissionOpponent,  #   a bake we submitted, frozen under baselines/
    "level2Silver2": Level2Silver2Opponent,   #   a second, stronger frozen rung
    "level2Silver3": Level2Silver3Opponent,   #   a third
    "level2Silver4": Level2Silver4Opponent,   #   a fourth - the current best candidate
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

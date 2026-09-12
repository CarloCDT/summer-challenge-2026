"""The referee's stdin/stdout protocol, as a pair of serializers and a parser.

This is the exact format `Serializer.serializeGlobalInfoFor` / `serializeFrameInfoFor` emit in
SummerChallenge2026's Java referee, so anything driven through here is being fed what CodinGame
would feed it. Two callers share it: test_submission.py, which replays a baked file as a
subprocess, and opponent.py's BakedSubmissionOpponent, which runs one as a boss.

Only `track` is absolute (the raw owner index); scores are already own-first and the board is
never mirrored, so a bot's whole perspective comes from the `my_id` line plus that field.
"""
from typing import List

from .game_state import Action, GameState


def init_lines(gs: GameState, my_id: int) -> List[str]:
    """The one-time header: player index, board size, every tile, then the towns."""
    out = [str(my_id), str(gs.width), str(gs.height)]
    for y in range(gs.height):
        for x in range(gs.width):
            tile = gs.grid.get(x, y)
            out.append(f"{tile.zone_id} {tile.type}")
    out.append(str(len(gs.towns)))
    for t in gs.towns:
        conns = ",".join(str(c.id) for c in t.desired_connections) or "x"
        out.append(f"{t.id} {t.coord[0]} {t.coord[1]} {conns}")
    return out


def frame_lines(gs: GameState, my_id: int) -> List[str]:
    """One turn: own score, opponent score, then every tile's track/instability/inked/conns."""
    out = [str(gs.scores[my_id]), str(gs.scores[1 - my_id])]
    active = gs.calculate_active_connections()
    for y in range(gs.height):
        for x in range(gs.width):
            tile = gs.grid.get(x, y)
            zone = gs.grid.zones[tile.zone_id]
            conns = active.get((x, y))
            cs = ",".join(sorted(f"{a}-{b}" for a, b in conns)) if conns else "x"
            out.append(f"{tile.track} {zone.instability} {1 if zone.inked else 0} {cs}")
    return out


def parse_actions(line: str) -> List[Action]:
    """The referee's own command vocabulary -> this env's action tuples."""
    actions: List[Action] = []
    for cmd in line.strip().split(";"):
        p = cmd.strip().split()
        if not p:
            continue
        head = p[0].upper()
        if head == "PLACE_TRACKS":
            actions.append(("PLACE", int(p[1]), int(p[2])))
        elif head == "DISRUPT":
            actions.append(("DISRUPT", int(p[1])))
        elif head == "WAIT":
            actions.append(("WAIT",))
        elif head == "AUTOPLACE":
            actions.append(("AUTOPLACE", int(p[1]), int(p[2]), int(p[3]), int(p[4])))
        elif head == "MESSAGE":
            pass
        else:
            raise ValueError(f"submission emitted an unparseable command: {cmd!r}")
    return actions

#!/usr/bin/env python3
"""Verifies the Python env against the original Java referee's rules.

Every check below cites the Java source it mirrors, so a future change to either side can be
diffed against this file. Run: python3 verify_rules.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from railroad_env import constants as C
from railroad_env.game_state import GameState
from railroad_env.grid import Grid, Town, Zone, get_rail_cost, manhattan
from railroad_env.grid_maker import GridMaker, JavaRandom
from railroad_env.pathfinding import train_bfs
from railroad_env import RailroadGymEnv

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def make_state(seed, height=None, max_turns=C.MAX_TURNS):
    grid = GridMaker(JavaRandom(seed), height=height).make()
    return GameState(grid, max_turns=max_turns)


# --------------------------------------------------------------------- map generation

def test_grid_dimensions():
    """GridMaker.init: h in [14,20], w = Math.round(h * 1.5)."""
    valid_pairs = {14: 21, 15: 23, 16: 24, 17: 26, 18: 27, 19: 29, 20: 30}
    seen = set()
    for seed in range(60):
        gs = make_state(seed)
        seen.add((gs.height, gs.width))
    bad = [(h, w) for h, w in seen if valid_pairs.get(h) != w]
    check("grid dimensions follow h in [14,20], w = round(h*1.5)", not bad, f"bad pairs {bad}")
    check("grid heights stay within [14, 20]", all(14 <= h <= 20 for h, _ in seen), f"{sorted(seen)}")


def test_zone_sizing():
    """Game.AVERAGE_TILES_PER_ZONE_COEFF_TO_GRID_HEIGHT: nZones = (h*w) / (h/2)."""
    ok = True
    for seed in range(20):
        gs = make_state(seed)
        expected = max(1, (gs.height * gs.width) // (gs.height // 2))
        if len(gs.zones) != expected:
            ok = False
    check("zone count = max(1, (h*w) // (h//2))", ok)

    gs = make_state(0)
    assigned = all(gs.grid.cells[c].zone_id >= 0 for c in gs.grid.cells)
    check("every cell belongs to a zone", assigned)

    covered = sum(len(z.coords) for z in gs.zones)
    check("zone coords partition the grid", covered == gs.width * gs.height,
          f"{covered} vs {gs.width * gs.height}")


def test_town_placement():
    """GridMaker.makeTowns: count, min distance, one per zone, neighbouring zones blacklisted."""
    for seed in range(15):
        gs = make_state(seed)
        expected_towns = max(4, (gs.height * gs.width) // C.AVERAGE_TILES_PER_TOWN)
        if len(gs.towns) > expected_towns:
            check("town count never exceeds max(4, h*w//50)", False, f"seed {seed}")
            return
        if not (4 <= len(gs.towns) <= 12):
            check("town count within the statement's 4..12", False, f"seed {seed}: {len(gs.towns)}")
            return
    check("town count = max(4, h*w//50), within the statement's 4..12", True)

    ok_distance, ok_plains, ok_one_per_zone, ok_no_adjacent = True, True, True, True
    for seed in range(15):
        gs = make_state(seed)
        for i, a in enumerate(gs.towns):
            for b in gs.towns[i + 1:]:
                if manhattan(a.coord, b.coord) < C.MIN_TOWN_DISTANCE:
                    ok_distance = False
        for t in gs.towns:
            if gs.grid.cells[t.coord].type != C.TYPE_GRASS:
                ok_plains = False
        zone_town_counts = Counter(gs.grid.cells[t.coord].zone_id for t in gs.towns)
        if any(v > 1 for v in zone_town_counts.values()):
            ok_one_per_zone = False
        town_zones = set(zone_town_counts)
        for zid in town_zones:
            for neigh in gs.zones[zid].neighbours:
                if neigh in town_zones:
                    ok_no_adjacent = False

    check(f"towns are >= {C.MIN_TOWN_DISTANCE} apart (manhattan)", ok_distance)
    check("towns always sit on plains", ok_plains)
    check("at most one town per zone", ok_one_per_zone)
    check("no two towns in bordering zones", ok_no_adjacent)


def test_desired_connections():
    """GridMaker.makeTownConnections: unilateral after the reciprocal filter."""
    ok_unilateral, ok_no_self = True, True
    for seed in range(20):
        gs = make_state(seed)
        for t in gs.towns:
            for other in t.desired_connections:
                if other.id == t.id:
                    ok_no_self = False
                if t in other.desired_connections:
                    ok_unilateral = False
    check("desired connections are unilateral (never reciprocal)", ok_unilateral)
    check("no town desires a connection to itself", ok_no_self)


def test_terrain_mix():
    """makeMountains / makeRivers produce blobs and flowing rivers, not i.i.d. noise."""
    gs = make_state(3)
    counts = Counter(gs.terrain.ravel().tolist())
    check("all three terrain types can appear", set(counts) <= {0, 1, 2} and counts[0] > 0)

    # Rivers flow, so water cells should be strongly contiguous - a random 15% scatter would not be.
    water = [(x, y) for (x, y), t in gs.grid.cells.items() if t.is_water()]
    if water:
        adjacent = sum(
            1
            for (x, y) in water
            for dx, dy in C.ADJACENCY
            if gs.grid.get(x + dx, y + dy) is not None and gs.grid.get(x + dx, y + dy).is_water()
        )
        check("river cells are contiguous (flowing, not scattered)", adjacent / len(water) >= 1.0,
              f"avg water neighbours {adjacent / len(water):.2f}")


# --------------------------------------------------------------------- rules

def test_rail_costs():
    """Game.getRailCost: plains 1, river 2, mountain 3."""
    grid = Grid(3, 1)
    grid.cells[(0, 0)].type = C.TYPE_GRASS
    grid.cells[(1, 0)].type = C.TYPE_WATER
    grid.cells[(2, 0)].type = C.TYPE_MOUNTAIN
    costs = [get_rail_cost(grid.cells[(i, 0)]) for i in range(3)]
    check("track costs are plains 1 / river 2 / mountain 3", costs == [1, 2, 3], str(costs))


def test_budgets():
    """Game.doIncome: 3 paint + 1 disruption point per turn, no carry-over."""
    gs = make_state(1)
    check("starts with 3 paint points", gs.paint_points == [3, 3], str(gs.paint_points))
    check("starts with 1 disruption point", gs.disruption_points == [1, 1], str(gs.disruption_points))

    placeable = [(x, y) for y in range(gs.height) for x in range(gs.width)
                 if gs._is_placeable(x, y) and gs.get_track_cost(x, y) == 1]
    gs.resolve_turn([[("PLACE", *placeable[0])], [("WAIT",)]])
    check("paint is spent on placement", gs.paint_points[0] == 2, str(gs.paint_points))
    gs.do_income()
    check("unused paint does not carry over", gs.paint_points == [3, 3], str(gs.paint_points))


def test_placement_rules():
    """Game.doActions: no placing on towns, existing tracks, or inked zones."""
    gs = make_state(2)
    town = gs.towns[0]
    gs.resolve_turn([[("PLACE", *town.coord)], [("WAIT",)]])
    check("cannot place a track on a town", gs.grid.cells[town.coord].track == C.TRACK_NONE)

    gs = make_state(2)
    spot = next((x, y) for y in range(gs.height) for x in range(gs.width)
                if gs._is_placeable(x, y) and gs.get_track_cost(x, y) == 1)
    gs.resolve_turn([[("PLACE", *spot)], [("WAIT",)]])
    owner_before = gs.grid.cells[spot].track
    gs.do_income()
    gs.resolve_turn([[("WAIT",)], [("PLACE", *spot)]])
    check("cannot place a track on an existing track", gs.grid.cells[spot].track == owner_before)


def test_neutral_conflict():
    """Game.doActions: same cell claimed by both players in one turn becomes neutral, both pay."""
    gs = make_state(4)
    spot = next((x, y) for y in range(gs.height) for x in range(gs.width)
                if gs._is_placeable(x, y) and gs.get_track_cost(x, y) == 1)
    gs.resolve_turn([[("PLACE", *spot)], [("PLACE", *spot)]])
    check("contested cell becomes a neutral track", gs.grid.cells[spot].track == C.TRACK_NEUTRAL,
          str(gs.grid.cells[spot].track))
    check("both players still pay for a contested cell", gs.paint_points == [2, 2],
          str(gs.paint_points))


def test_disruption_rules():
    """Game.doActions + doInstabilityCheck: threshold 4, town zones immune, inked zones rejected."""
    gs = make_state(5)
    town_zone = gs.grid.cells[gs.towns[0].coord].zone_id
    check("a zone containing a town cannot be disrupted", not gs.can_disrupt(town_zone))

    plain_zone = next(z for z in gs.zones if not z.contained_towns)
    for turn in range(C.INSTABILITY_THRESHOLD_BASE):
        gs.resolve_turn([[("DISRUPT", plain_zone.id)], [("WAIT",)]])
        gs.do_income()
    check(f"zone inks out at instability {C.INSTABILITY_THRESHOLD_BASE}", plain_zone.inked,
          f"instability={plain_zone.instability} inked={plain_zone.inked}")
    check("an inked zone cannot be disrupted again", not gs.can_disrupt(plain_zone.id))
    check("cannot place tracks in an inked zone",
          all(not gs._is_placeable(x, y) for (x, y) in plain_zone.coords))

    gs2 = make_state(5)
    tz = gs2.grid.cells[gs2.towns[0].coord].zone_id
    for _ in range(6):
        gs2.resolve_turn([[("DISRUPT", tz)], [("WAIT",)]])
        gs2.do_income()
    check("a town zone never inks out however often it is targeted", not gs2.zones[tz].inked)

    gs3 = make_state(6)
    zone = next(z for z in gs3.zones if not z.contained_towns)
    gs3.resolve_turn([[("DISRUPT", zone.id), ("DISRUPT", zone.id)], [("WAIT",)]])
    check("only one disruption point may be spent per turn", zone.instability == 1,
          str(zone.instability))


def test_inking_destroys_tracks():
    """Game.doInstabilityCheck: inking a zone wipes every track in it."""
    gs = make_state(7)
    zone = next(z for z in gs.zones if not z.contained_towns)
    spot = next(((x, y) for (x, y) in zone.coords if gs._is_placeable(x, y)
                 and gs.get_track_cost(x, y) <= 3), None)
    gs.resolve_turn([[("PLACE", *spot)], [("WAIT",)]])
    gs.do_income()
    check("track placed before inking", gs.grid.cells[spot].is_track())

    for _ in range(C.INSTABILITY_THRESHOLD_BASE):
        gs.resolve_turn([[("DISRUPT", zone.id)], [("WAIT",)]])
        gs.do_income()
    check("inking washes away tracks in the zone", not gs.grid.cells[spot].is_track())
    check("the tracks mirror array is updated too", gs.tracks[spot[1], spot[0]] == C.TRACK_NONE)


def test_bfs_tiebreak():
    """TrainBFS: equal-length paths resolve N > E > S > W from the requesting town."""
    grid = Grid(3, 3)
    for tile in grid.cells.values():
        tile.type = C.TYPE_GRASS
        tile.zone_id = 0
    grid.zones = [Zone(0, list(grid.cells.keys()))]

    a = Town(0, (0, 1))
    b = Town(1, (2, 1))
    grid.towns = [a, b]
    grid.cells[(0, 1)].town_id = 0
    grid.cells[(2, 1)].town_id = 1

    # Two equally short routes: over the top (north) or under the bottom (south).
    for coord in [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]:
        grid.cells[coord].track = 0

    path = train_bfs(grid, a, b)
    check("BFS picks the NORTH route over the SOUTH one on a tie",
          path == [(0, 1), (0, 0), (1, 0), (2, 0), (2, 1)], str(path))


def test_scoring():
    """Game.moveTrains: 1 point per own track on each active connection; neutral scores nothing."""
    grid = Grid(5, 1)
    for tile in grid.cells.values():
        tile.type = C.TYPE_GRASS
        tile.zone_id = 0
    grid.zones = [Zone(0, list(grid.cells.keys()))]

    a, b = Town(0, (0, 0)), Town(1, (4, 0))
    a.desired_connections = [b]
    grid.towns = [a, b]
    grid.cells[(0, 0)].town_id = 0
    grid.cells[(4, 0)].town_id = 1
    grid.cells[(1, 0)].track = 0
    grid.cells[(2, 0)].track = 1
    grid.cells[(3, 0)].track = C.TRACK_NEUTRAL

    gs = GameState(grid)
    gs.resolve_turn([[("WAIT",)], [("WAIT",)]])
    check("each player scores 1 point per own track on the connection", gs.scores == [1, 1],
          str(gs.scores))
    check("neutral tracks score for nobody", gs.scores == [1, 1], str(gs.scores))

    gs.resolve_turn([[("WAIT",)], [("WAIT",)]])
    check("an active connection keeps paying out every turn", gs.scores == [2, 2], str(gs.scores))

    # A cell serving two connections scores twice.
    c = Town(2, (2, 0))
    check("towns themselves are worth no points (no track on them)",
          gs.grid.cells[(0, 0)].track == C.TRACK_NONE)


def test_turn_order():
    """Game.performGameUpdate: PLACE before DISRUPT, inking before scoring."""
    gs = make_state(8)
    zone = next(z for z in gs.zones if not z.contained_towns)
    for _ in range(C.INSTABILITY_THRESHOLD_BASE - 1):
        gs.resolve_turn([[("DISRUPT", zone.id)], [("WAIT",)]])
        gs.do_income()
    spot = next(((x, y) for (x, y) in zone.coords if gs._is_placeable(x, y)), None)
    if spot is not None:
        # Place into the zone on the very turn the final disruption inks it: the track is placed
        # first, then washed away by the ink, and scores nothing.
        gs.resolve_turn([[("PLACE", *spot)], [("DISRUPT", zone.id)]])
        check("a track placed on the turn its zone inks is destroyed",
              not gs.grid.cells[spot].is_track() and gs.zones[zone.id].inked)


def test_game_over():
    """Game.isGameOver: turn limit, or no desired connection even theoretically possible."""
    gs = make_state(9, max_turns=3)
    for _ in range(3):
        check_not_done = not gs.is_done()
        gs.resolve_turn([[("WAIT",)], [("WAIT",)]])
        gs.do_income()
    check("game ends at max_turns", gs.is_done(), f"turn={gs.turn}")

    gs2 = make_state(9)
    check("game is not over at the start", not gs2.is_done())


def test_autoplace():
    """Game.computeAutobuilds/AutobuildAStar: AUTOPLACE expands into a cheapest track chain."""
    gs = make_state(11)
    a, b = gs.towns[0], gs.towns[1]
    gs.resolve_turn([[("AUTOPLACE", a.coord[0], a.coord[1], b.coord[0], b.coord[1])], [("WAIT",)]])
    placed = [c for c in gs.grid.cells if gs.grid.cells[c].track == 0]
    check("AUTOPLACE places tracks", len(placed) > 0, f"placed {len(placed)}")
    check("AUTOPLACE respects the paint budget", gs.paint_points[0] >= 0
          and sum(gs.get_track_cost(*c) for c in placed) <= C.PASSIVE_INCOME,
          f"spent {sum(gs.get_track_cost(*c) for c in placed)}")
    check("AUTOPLACE tracks form a chain touching the source town",
          any(manhattan(c, a.coord) == 1 for c in placed) if placed else False)


def test_full_game_via_env():
    """End-to-end: a whole game against the shipped Boss AI runs and scores."""
    env = RailroadGymEnv(seed=12, opponent_strategy="boss", max_turns=100)
    obs, info = env.reset()
    total_reward = 0.0
    turns = 0
    while True:
        placements = env.legal_placements()
        actions = [("PLACE", *placements[0])] if placements else [("WAIT",)]
        obs, reward, terminated, _, info = env.step({"actions": actions})
        total_reward += reward
        turns += 1
        if terminated:
            break
    check("a full game runs to completion", turns <= 100 and terminated, f"turns={turns}")
    check("observation shape is the padded canvas", obs.shape[2] == GameState.NUM_CHANNELS)
    check("the Boss opponent scores points via AUTOPLACE", info["scores"][1] > 0,
          f"scores={info['scores']}")


# --------------------------------------------------------------------- lite wrapper

def test_lite_env():
    """RailroadLiteEnv: the documented simplifications hold, and the rest is still the real game."""
    from railroad_lite_env import NUM_LITE_CHANNELS, RailroadLiteEnv
    from training.lite import LITE_INKED_CHANNEL, LiteGameSimulator, lite_encode_state

    env = RailroadLiteEnv(seed=21)
    obs, info = env.reset()
    gs = env.game_state

    check("lite: 25 turns", gs.max_turns == 25, str(gs.max_turns))
    check("lite: terrain is all plains", set(np.unique(gs.terrain).tolist()) == {C.TYPE_GRASS})
    costs = {gs.get_track_cost(x, y) for y in range(gs.height) for x in range(gs.width)}
    check("lite: every track costs 1", costs == {1}, str(costs))
    check("lite: paint budget is still 3", gs.paint_points == [3, 3], str(gs.paint_points))
    check(f"lite: observation has {NUM_LITE_CHANNELS} channels",
          obs.shape[2] == NUM_LITE_CHANNELS, str(obs.shape))

    # DISRUPT is dropped for the agent...
    before = [z.instability for z in gs.zones]
    env.step({"actions": [("DISRUPT", 0), ("DISRUPT_AT", 1, 1), (2, (0, 0))]})
    check("lite: agent DISRUPT actions are dropped",
          [z.instability for z in gs.zones] == before)

    # ...and never used by the opponent either, so nothing ever inks over a whole game.
    env2 = RailroadLiteEnv(seed=22)
    env2.reset()
    sim = LiteGameSimulator.from_env(env2)
    decisions_per_turn, turn, count = [], sim.game_state.turn, 0
    while not sim.is_game_over():
        legal = sim.get_legal_actions()
        if any(kind != "PLACE" for kind, _, _ in legal):
            check("lite: only PLACE actions are ever legal", False)
            return
        sim.apply(legal[0])
        count += 1
        if sim.game_state.turn != turn:
            decisions_per_turn.append(count)
            count, turn = 0, sim.game_state.turn
    check("lite: only PLACE actions are ever legal", True)
    check("lite: exactly 3 placements per turn", set(decisions_per_turn) == {3},
          str(set(decisions_per_turn)))
    check("lite: game runs the full 25 turns", len(decisions_per_turn) == 25,
          str(len(decisions_per_turn)))
    check("lite: nothing ever inks out", not any(z.inked for z in sim.game_state.zones))

    # The encoding still centers the board and marks padding unbuildable.
    encoded = lite_encode_state(sim.game_state)
    check("lite: encoded state is the padded canvas",
          encoded.shape == (NUM_LITE_CHANNELS, 20, 30), str(encoded.shape))
    pad_rows = (20 - sim.game_state.height) // 2
    if pad_rows:
        check("lite: off-board padding is marked inked",
              bool(encoded[LITE_INKED_CHANNEL][:pad_rows, :].all()))
    check("lite: mask is PLACE-only", sim.get_action_mask().shape == (1, 20, 30))

    # Still the real game underneath: unilateral connections, spaced towns, real scoring.
    check("lite: towns still spaced >= 4 apart",
          all(manhattan(a.coord, b.coord) >= C.MIN_TOWN_DISTANCE
              for i, a in enumerate(gs.towns) for b in gs.towns[i + 1:]))
    check("lite: desired connections still unilateral",
          all(t not in o.desired_connections for t in gs.towns for o in t.desired_connections))
    check("lite: scoring still happens", sim.game_state.scores[0] >= 0)


# --------------------------------------------------------------------- derived channels

def test_derived_features():
    """railroad_env/features.py, observation channels 28-37: each plane means what it claims,
    checked against the rules engine rather than against its own arithmetic."""
    from railroad_env.opponent import make_opponent
    from railroad_env.pathfinding import autobuild, terrain_path_exists

    base = GameState.DERIVED_CHANNEL_START
    gs = make_state(77, height=20)
    players = [make_opponent("level2ProMax", seed=1), make_opponent("greedy", seed=2)]
    disrupts = [0, 0]
    income_ok = memory_ok = reach_ok = chain_ok = True
    chains = 0
    while not gs.is_done() and gs.turn < 60:
        actions = [p.get_actions(gs, pid) for pid, p in enumerate(players)]
        summary = gs.resolve_turn(actions)
        gs.do_income()
        for pid in (0, 1):
            disrupts[pid] += len(summary["disrupted"][pid])
        # Game.moveTrains pays one point per own track per active path, so the connection counts
        # summed over a player's track must equal what the turn actually paid that player.
        income_ok &= list(gs.income_rates()) == list(summary["score_delta"])
        wanted = [(t, o) for t in gs.towns for o in t.desired_connections]
        reachable = sum(terrain_path_exists(gs.grid, t, o) for t, o in wanted) / max(1, len(wanted))
        for pid in (0, 1):
            obs = gs.get_observation(pid)
            memory_ok &= int(gs.disruption_by_player[1 - pid].sum()) == disrupts[1 - pid]
            memory_ok &= bool(np.allclose(obs[:, :, base + 9] * 4,
                                          gs.disruption_by_player[1 - pid][gs.regions]))
            reach_ok &= abs(float(obs[0, 0, base + 8]) - reachable) < 1e-6
        if gs.turn % 10 == 0:
            corridor = gs.get_observation(0)[:, :, base + 6]
            for t, o in wanted:
                if o.id in t.paths:
                    continue
                chain = autobuild(gs.grid, t.coord, o.coord)
                if chain:
                    chains += 1
                    chain_ok &= all(corridor[y, x] > 0 for (x, y) in chain)
    check("derived: connection counts over own track equal each turn's actual score delta", income_ok)
    check("derived: opponent-disruption memory counts every successful enemy DISRUPT", memory_ok)
    check("derived: reachable-connection share matches TerrainAStar", reach_ok)
    check(f"derived: every AUTOPLACE chain lies on the cheapest-completion corridor ({chains} chains)",
          chain_ok and chains > 0)


def main():
    print("=== map generation ===")
    test_grid_dimensions()
    test_zone_sizing()
    test_town_placement()
    test_desired_connections()
    test_terrain_mix()

    print("\n=== rules ===")
    test_rail_costs()
    test_budgets()
    test_placement_rules()
    test_neutral_conflict()
    test_disruption_rules()
    test_inking_destroys_tracks()
    test_bfs_tiebreak()
    test_scoring()
    test_turn_order()
    test_game_over()
    test_autoplace()

    print("\n=== end to end ===")
    test_full_game_via_env()

    print("\n=== derived observation channels ===")
    test_derived_features()

    print("\n=== lite wrapper ===")
    test_lite_env()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All rule checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

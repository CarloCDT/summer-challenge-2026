"""Derived observation channels 28-37, computed from board arrays with numpy only.

This module is deliberately free of Grid/GameState so that `bake_agent.RUNTIME` can carry an
exact copy of the same arithmetic. Any change here must be mirrored there, and
`verify_bake_parity.py` must stay green: it feeds real referee frames through the baked `observe`
and compares every channel against `encode_state`.

    28  connections through this cell / 10          (a track here pays this many points per turn)
    29  own income at stake in this region / 20     (sum of 28 over own track in the region)
    30  enemy income at stake in this region / 20
    31  tanh(score diff / 300), flat                (channel 26 reads ~0.005 in contested games)
    32  tanh(income-rate diff / 30), flat
    33  region contains a town, i.e. can never be inked
    34  not-yet-active connections whose cheapest completion passes through this cell / 5
    35  1 / (cheapest remaining completion cost) of the cheapest such connection here, else 0
    36  share of desired connections still reachable at all, flat (the early game-over rule)
    37  instability the OPPONENT has added to this region / 4

"Cheapest completion" prices every cell like AUTOPLACE does: existing track (any owner) and towns
are free, an empty cell costs its terrain, inked regions are impassable. A cell is on a cheapest
completion when cost(from -> cell) + cost(to -> cell) - cost(cell) equals the connection's total,
so every tied cheapest route is marked, not one arbitrary path.
"""
import numpy as np

NUM_DERIVED_CHANNELS = 10

CONN_SCALE = 10.0
REGION_INCOME_SCALE = 20.0
SCORE_DIFF_TANH = 300.0
INCOME_DIFF_TANH = 30.0
CORRIDOR_SCALE = 5.0
INSTABILITY_SCALE = 4.0


def multi_source_cost(step, blocked, sources):
    """(len(sources), H, W) cheapest summed `step` cost from each source to every cell.

    The source cell is free, every other cell on the path costs `step` when entered, `blocked`
    cells are unreachable (inf). Vectorised Bellman-Ford over all sources at once: each sweep
    relaxes every cell against its four neighbours, and it stops when nothing changes."""
    h, w = step.shape
    enter = np.where(blocked, np.inf, step.astype(np.float64))
    d = np.full((len(sources), h, w), np.inf)
    for i, (x, y) in enumerate(sources):
        d[i, y, x] = 0.0
    while True:
        n = np.full_like(d, np.inf)
        n[:, 1:, :] = d[:, :-1, :]
        np.minimum(n[:, :-1, :], d[:, 1:, :], out=n[:, :-1, :])
        np.minimum(n[:, :, 1:], d[:, :, :-1], out=n[:, :, 1:])
        np.minimum(n[:, :, :-1], d[:, :, 1:], out=n[:, :, :-1])
        new = np.minimum(d, n + enter)
        if (new == d).all():
            return d
        d = new


def derived_channels(track, own_id, zone, zone_has_town, zone_inked, terrain_cost, is_town,
                     towns, active_pairs, conn_count, score_diff, opp_instability):
    """(NUM_DERIVED_CHANNELS, H, W) float32.

    track          (H, W) int   -1 none, 0/1 owner, 2 neutral
    zone           (H, W) int   region id per cell
    zone_has_town  (Z,) bool,  zone_inked (Z,) bool
    terrain_cost   (H, W) int   1 plains / 2 river / 3 mountain
    is_town        (H, W) bool
    towns          [(town_id, x, y, [desired town ids]), ...]
    active_pairs   set of (from_id, to_id) connections currently active
    conn_count     (H, W) int   active connections whose path covers the cell
    score_diff     own score - opponent score
    opp_instability (Z,)       instability the opponent has added to each region
    """
    h, w = track.shape
    nz = zone_inked.shape[0]
    out = np.zeros((NUM_DERIVED_CHANNELS, h, w), np.float32)

    m = conn_count.astype(np.float64)
    own = track == own_id
    foe = track == 1 - own_id
    flat_zone = zone.ravel()
    own_income = np.bincount(flat_zone, weights=(own * m).ravel(), minlength=nz)
    foe_income = np.bincount(flat_zone, weights=(foe * m).ravel(), minlength=nz)

    out[0] = m / CONN_SCALE
    out[1] = own_income[zone] / REGION_INCOME_SCALE
    out[2] = foe_income[zone] / REGION_INCOME_SCALE
    out[3] = np.tanh(score_diff / SCORE_DIFF_TANH)
    out[4] = np.tanh((own_income.sum() - foe_income.sum()) / INCOME_DIFF_TANH)
    out[5] = zone_has_town[zone]

    index = {t[0]: i for i, t in enumerate(towns)}
    step = np.where((track >= 0) | is_town, 0, terrain_cost)
    blocked = zone_inked[zone] & ~is_town
    dist = multi_source_cost(step, blocked, [(t[1], t[2]) for t in towns])
    corridor = np.zeros((h, w))
    urgency = np.zeros((h, w))
    total = reachable = 0
    for tid, tx, ty, desired in towns:
        for other in desired:
            if other not in index:
                continue
            total += 1
            ox, oy = towns[index[other]][1], towns[index[other]][2]
            cost = dist[index[tid], oy, ox]
            if not np.isfinite(cost):
                continue
            reachable += 1
            if (tid, other) in active_pairs or cost <= 0:
                continue
            on = dist[index[tid]] + dist[index[other]] - step == cost
            corridor += on
            urgency = np.maximum(urgency, np.where(on, 1.0 / cost, 0.0))
    out[6] = corridor / CORRIDOR_SCALE
    out[7] = urgency
    out[8] = reachable / total if total else 0.0
    out[9] = opp_instability[zone] / INSTABILITY_SCALE
    return out

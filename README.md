# Railroad Tycoon RL — CodinGame Summer Challenge 2026

A Gymnasium environment that is a **faithful Python port of the official Java referee**, plus an
RL training stack (PPO and an AlphaZero-style MCTS variant) built on top of it.

Two players draw train tracks on a shared map, connecting towns that want to be connected, while
sabotaging each other by destabilising regions of the map. The original referee ships in
[`SummerChallenge2026/`](SummerChallenge2026/); every rule in this port is checked against it by
[`verify_rules.py`](verify_rules.py).

```bash
python3 verify_rules.py          # 57 checks: Python rules vs. the Java referee, + the lite wrapper
python3 training/verify_stage_a.py   # encoding + model + simulator + MCTS work together
```

---

## 📁 Layout

```
railroad_env/               The game, ported from the Java referee
├── constants.py            Every tunable from Game.java (budgets, costs, thresholds, sizes)
├── grid.py                 Tile / Zone / Town / Grid primitives
├── grid_maker.py           Map generation: mountains, rivers, zones, towns, connections
├── pathfinding.py          TrainBFS (connections), TerrainAStar (reachability), AUTOPLACE A*
├── game_state.py           Turn resolution, scoring, disruption + the RL observation
├── environment.py          Gymnasium wrapper
├── opponent.py             passive / boss / random / greedy opponents
└── renderer.py             pygame visualization

training/
├── encoding.py             Board -> fixed 20x30 network canvas (padding + action masks)
├── model.py                TrainUNet: dual-head (policy heatmap + value) U-Net
├── simulator.py            GameSimulator: sub-turn state machine for atomic decisions
├── ppo.py, train_ppo.py    PPO (recommended)
├── mcts.py, self_play.py, replay_buffer.py, train.py    AlphaZero-style alternative
└── configs/                Ready-made run configurations

railroad_lite_env.py        Lite game: a Gym wrapper stripping the full env down (see below)
training/lite.py, lite_model.py, train_lite_ppo.py    PPO for the lite game

verify_rules.py             Rule-by-rule verification against the Java referee (57 checks)
example_play.ipynb          Watch a full game, scripted player
debug_full_agent.ipynb      Step through a game turn by turn, inspecting what the net sees
debug_lite_agent.ipynb      The same, for the lite game
docs/play_from_checkpoint.py    Watch a trained checkpoint play (pygame)
docs/play_with_rendering.py     Watch a scripted player play (pygame)
```

---

## 🎮 The game

### Map

Generated fresh every game, matching `GridMaker.java`:

| Property | Value |
|---|---|
| Height | random 14–20 |
| Width | `round(height × 1.5)` → 21×14, 23×15, 24×16, 26×17, 27×18, 29×19, 30×20 |
| Terrain | plains (0), river (1), mountain (2) — rivers *flow* from map edges and can split; mountains are grown blobs of 2–8 cells |
| Regions | `(h×w) / (h/2)` of them (≈55 on a 26×17 map, ~8 cells each), grown round-robin from evenly spaced seeds |
| Towns | `max(4, h×w/50)`, one per region, never in two bordering regions, never on an edge, ≥ 4 apart (manhattan), always on plains |

Each town has `desired_connections`: other towns it wants linked to it. **These are
one-directional** — if town 0 wants town 1, town 1 will not also want town 0.

### Turn structure

Every turn both players get **3 paint points** and **1 disruption point**. Neither carries over.

1. All `AUTOPLACE` actions expand into `PLACE_TRACKS` chains.
2. **All** `PLACE_TRACKS` from both players resolve simultaneously against the turn-start board.
3. **Then** all `DISRUPT` actions resolve.
4. Regions at instability ≥ 4 ink out, destroying every track inside them.
5. Connections are recomputed and **scored**.

### Placing track

Costs 1 / 2 / 3 paint points on plains / river / mountain. You cannot place on a town, on an
existing track, or in an inked-out region. If both players claim the same free cell on the same
turn, it becomes a **neutral** track (owner `2`) — and both still pay for it.

### Connections & scoring

For each desired pair, if any unbroken path of track-or-town cells links them, the **shortest**
such path becomes the *active connection*. Ties break by direction priority **N → E → S → W**,
walking from the requesting town toward the desired one.

At the end of every turn, each active connection pays each player **1 point per track they own
on that path**. Neutral tracks and the towns themselves pay nobody. Payouts repeat every turn the
connection stays up, so scores compound — a scripted bot that builds all its connections early
can finish a 100-turn game with 20,000+ points.

### Disruption

Spend your disruption point to add **+1 instability** to a region. At **4**, the region inks out:
every track in it is destroyed and nothing can ever be built there again. You cannot disrupt a
region that is already inked, **or one that contains a town**.

### Game over

After 100 turns, or as soon as no desired connection is even theoretically reachable any more
(every route between remaining pairs severed by inked regions). Highest score wins.

---

## 🕹️ Using the environment

```python
from railroad_env import RailroadGymEnv

env = RailroadGymEnv(
    grid_height=None,            # None = authentic random size; pin an int for a fixed map
    max_turns=100,
    opponent_strategy="passive", # "passive" | "boss" | "random" | "greedy"
    seed=42,
)
obs, info = env.reset()          # obs: (height, width, 25)

obs, reward, terminated, truncated, info = env.step({"actions": [
    ("PLACE", 4, 7),
    ("PLACE", 5, 7),
    ("DISRUPT", 12),             # or ("DISRUPT_AT", x, y) to target the region at a cell
]})
```

Actions use the game's own vocabulary — `("WAIT",)`, `("PLACE", x, y)`, `("DISRUPT", region_id)`,
`("DISRUPT_AT", x, y)`, `("AUTOPLACE", x1, y1, x2, y2)` — and a whole turn's worth is submitted
at once, exactly as a real bot's semicolon-separated output line would be. `reward` is player 0's
score delta for that turn.

### Opponents

| Name | Behaviour |
|---|---|
| `passive` | Always waits — the challenge's own default and League 1 Boss |
| `boss` | The League 2 Boss: `AUTOPLACE` between two random towns every turn. Strong. |
| `random` | Random legal placements, sometimes disrupts (RL baseline) |
| `greedy` | Extends its own track, disrupts whichever region holds the most enemy track |

### Observation (28 channels)

| Channels | Meaning |
|---|---|
| 0–2 | terrain one-hot (plains / river / mountain) |
| 3–4 | enemy / own track present |
| 5–6 | enemy / own tracks in this cell's region, **count ÷ 10** (not a density) |
| 7 | region size ÷ total cells |
| 8–9 | enemy / own tracks *on an active connection* in this region, **count ÷ 10** |
| 10–21 | per-town open-terrain route guides: `-1` this town, `-0.5` a town it wants, `+1` between |
| 22 | region instability ÷ 4 |
| 23 | region inked |
| 24 | cell is part of an active connection |
| 25 | a town stands here (ownerless, never buildable) |
| 26 | (own − enemy score) ÷ 10000, the same value in every cell |
| 27 | turn ÷ 100, the same value in every cell |

Channels 5, 6, 8 and 9 are **counts, not densities**: the quantities the game acts on are
absolute (a disruptor compares track counts; an active connection pays per track), and dividing
by region size destroyed exactly that signal. Channels 26–27 are flat planes because a
convolution is local and has no other way to see a board-wide scalar.

---

## 🪶 Lite game

`RailroadLiteEnv` is a Gym wrapper that strips the faithful env down to the part that matters
for learning to connect towns. The full env is untouched underneath — map generation, town
placement, desired connections, connection pathfinding and scoring are all still the real thing.

| | Full game | Lite game |
|---|---|---|
| Terrain | plains / river / mountain (cost 1/2/3) | **plains only** (cost 1) |
| Paint budget | 3/turn | 3/turn (unchanged) → exactly **3 placements/turn** |
| Disruption | 1 point/turn, regions ink out at instability 4 | **none**, for either player |
| Turns | 100 | **25** |
| Opponent | passive / boss / random / greedy | **half-idle random**: coin flip each turn between WAIT and spending its whole budget on random placements |
| Observation | 28 channels | **16 channels** |
| Action space | PLACE + DISRUPT | **PLACE only** |

```python
from railroad_lite_env import RailroadLiteEnv

env = RailroadLiteEnv(grid_height=None, max_turns=25, opponent_wait_probability=0.5, seed=42)
obs, info = env.reset()          # obs: (height, width, 16)
obs, reward, terminated, truncated, info = env.step({"actions": [("PLACE", 4, 7)]})
```

`DISRUPT` actions are silently dropped rather than raising — matching how the real referee skips
impossible actions. The 16 kept channels are: own/enemy track present, the 12 per-town route
guides, the inked channel (which in the lite game only ever marks the off-board padding), and
active connection. Everything dropped was either constant (terrain, instability) or a region
aggregate that stops carrying information once disruption is gone.

### Training the lite agent

```bash
python3 -m training.train_lite_ppo --config training/configs/ppo_lite.yaml
```

`LiteUNet` is a much smaller network than the full game's — **549k params** at
`base_channels=32` (or 329k at 24), versus 4.0M for `TrainUNet`:

```
conv in -> enc1 ------------------------- concat -> dec1 -> conv out (PLACE heatmap)
             |                               |
           pool                          upsample
             |                               |
           enc2 ----------- concat -----> dec2
             |                 |
           pool           upsample
             |                 |
           bottleneck ---------
             |
           critic (value)
```

Two downsampling stages and two upsampling stages, a plain convolution before the encoder and
after the decoder, and a critic reading the bottleneck through a small spatial pool. The value
head is unbounded (no `Tanh`), since PPO regresses onto discounted returns rather than a
normalized win margin. Rollouts run ~0.5s/iteration with 8 workers, against ~4-5s for the full
game.

---

## 🧠 Training

The network sees a **fixed 20×30 canvas** (the largest possible board). Smaller maps are centered
in it and the padding is marked as inked, so it reads as permanently unbuildable.

`TrainUNet` is a dual-head U-Net (~4.0M params): the decoder produces a per-cell policy heatmap
(channel 0 = PLACE, channel 1 = DISRUPT) and a value head branches off the bottleneck. Agents act
one **atomic decision** at a time — a single cell to build on, or a single disruption target —
which `GameSimulator` accumulates into whole turns.

### PPO (recommended)

```bash
python3 -m training.train_ppo --config training/configs/ppo.yaml
python3 -m training.train_ppo --config training/configs/ppo.yaml --opponent-strategy boss --num-workers 8
```

Clipped-surrogate PPO with GAE, an entropy bonus, and epsilon-greedy action selection that
decays over the run. Rollouts can be spread over worker processes with `--num-workers`.

| Config | Setup |
|---|---|
| `ppo_full.yaml` | **Full game, 5000 iterations vs. a half-idle random opponent — the main run** |
| `ppo.yaml` | Full game, passive opponent |
| `ppo_simple.yaml` | Pinned 21×14 map, 50 turns — fast pipeline check |
| `ppo_vs_random.yaml` | Full game vs. the random baseline |

The policy head is a spatial heatmap with one channel per action kind:

| Channel | Action | Notes |
|---|---|---|
| 0 | `PLACE` | **Compulsory** — the place phase only ends once the 3-point paint budget can't buy anything, so the budget is always spent in full |
| 1 | `DISRUPT` | Which region to destabilise |
| 2 | `SKIP_DISRUPT` | Decline to disrupt this turn (`allow_skip_disrupt`), offered only when a legal target exists |

`SKIP_DISRUPT` is masked across the whole board plane rather than a single cell. `DISRUPT` spans
every cell of every disruptable region — often several hundred — so a lone skip cell would carry
~1/400th of the probability mass and an untrained policy would essentially never decline. Every
cell of the channel decodes to the same action, so spreading it makes the two options comparable
a priori (measured: 48% skip rate at initialization, versus 0% before).

### AlphaZero-style (MCTS)

```bash
python3 -m training.train --config training/configs/default.yaml
```

Self-play with MCTS-improved policy targets. Slower per iteration but a much stronger learning
signal than the search-free baselines (`no_mcts.yaml`, `unet_only.yaml`,
`connect_towns_topk.yaml`), which exist mainly as fast pipeline checks.

### Watching an agent

```bash
python3 docs/play_from_checkpoint.py --checkpoint checkpoints/<run>_iter500.pt --opponent boss
tensorboard --logdir runs
```

`debug_full_agent.ipynb` steps through a game one turn at a time, plotting all 28 input channels
and the policy's PLACE/DISRUPT/SKIP_DISRUPT heatmaps for every decision.

---

## 🔍 Fidelity

`verify_rules.py` checks the port against the Java referee: map dimensions, region sizing, town
placement constraints, unilateral connections, river contiguity, track costs, budgets, placement
legality, neutral-track conflicts, disruption rules (including town-region immunity), inking,
BFS tie-break order, scoring, turn ordering, game-over conditions, and AUTOPLACE.

Two deliberate deviations, both invisible to gameplay:

- **Random streams differ.** Java's `Random` and numpy generate different sequences, so a given
  seed produces a different (but distributionally identical) map. Ranges and probabilities match.
- **`AUTOPLACE` reproduces an original quirk.** `AutobuildAStar`'s heuristic is
  `cursor.manhattanTo(cursor)` — always 0 — making it a uniform-cost search rather than a guided
  A*. The port keeps that, since "fixing" it would produce different paths than the real game.

## 📦 Install

```bash
pip install -r requirements.txt
```

Torch needs a CUDA build to use a GPU; the generic `torch` PyPI wheel is CPU-only. See
`requirements.txt` for details.

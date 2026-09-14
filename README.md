# Railroad Tycoon RL — CodinGame Summer Challenge 2026

A Gymnasium environment that faithfully ports the official Java referee, plus the
PPO → distillation → bake pipeline that produces our CodinGame submission. Experiment results and
working notes are in [CLAUDE.md](CLAUDE.md).

```bash
pip install -r requirements.txt   # torch needs a CUDA build to use a GPU
python3 verify_rules.py           # 62 checks against the Java referee in SummerChallenge2026/
```

## The game

- Maps are 14–20 rows tall, width `round(1.5 × height)`, so up to 20×30. Terrain is plains, river or
  mountain. The map is split into regions, with at most one town per region.
- Each town has one-directional `desired_connections` to other towns.
- Every turn both players get **3 paint points** and **1 disruption point**. Neither carries over.
- **Turn order:** `AUTOPLACE` expands into placements → all placements from both players resolve
  against the turn-start board (same cell ⇒ neutral track, both pay) → all disruptions → regions at
  instability 4 ink out → connections are recomputed and scored.
- Track costs 1 / 2 / 3 on plains / river / mountain. You can't build on towns, existing track or
  inked regions.
- A connection is the shortest path of track-or-town cells (BFS, N→E→S→W tie-break). Each turn it
  pays each player **1 point per track they own on it**.
- Disrupting adds 1 instability. At 4 the region inks out, destroying all track in it for good.
  Regions with a town can't be disrupted.
- The game ends after 100 turns, or earlier once no desired connection is reachable.

## Using the environment

```python
from railroad_env import RailroadGymEnv

env = RailroadGymEnv(grid_height=None, max_turns=100, opponent_strategy="passive", seed=42)
obs, info = env.reset()                     # obs: (height, width, 38)
obs, reward, terminated, truncated, info = env.step({"actions": [
    ("PLACE", 4, 7),
    ("DISRUPT", 12),
]})
```

- **Actions:** `("WAIT",)`, `("PLACE", x, y)`, `("DISRUPT", region_id)`, `("DISRUPT_AT", x, y)`,
  `("AUTOPLACE", x1, y1, x2, y2)`. A numeric `(type, params)` form with types 0–3 is also accepted.
- **Reward:** `reward_mode="own"` (default) is player 0's score delta; `"margin"` subtracts the
  opponent's.
- **Opponents:** `passive`/`level1` and `boss`/`level2` are the shipped bosses. Also `random`,
  `greedy`, `level1Pro`, `level2Pro`, `level2ProMax`, and `level2Silver`…`level2Silver4`, which are
  frozen bakes of our own agents (see `baselines/`).
- The 38 observation channels are listed in CLAUDE.md (28 base, 10 derived in
  `railroad_env/features.py`).

## Layout

```
railroad_env/          the game, ported from the Java referee
training/              encoding, RailroadUNet, PPO, distillation, quantization, configs/
bake_agent.py          checkpoint -> self-contained submission.py
bake_debug_agent.py    debug bake that prints win probability to stderr (never submit)
test_submission.py     plays a baked file over the referee's wire protocol
evaluate_checkpoints.py, compare_agents.py, diagnose_agent.py   evaluation tools
baselines/             frozen bakes used as opponents
verify_rules.py        rule-by-rule checks against the Java referee
SummerChallenge2026/   the original referee and bosses
docs/                  play_from_checkpoint.py, play_with_rendering.py
```

## Pipeline

```bash
python3 -u -m training.train_ppo --config training/configs/ppo_28ch_vs_silver3_1k.yaml   # teacher
python3 -u -m training.distill --config training/configs/distill.yaml                   # student
python3 bake_agent.py checkpoints/<student>.pt -o submission.py --quant int4             # bake
python3 test_submission.py submission.py --opponent level2Silver4 --episodes 50          # check
tensorboard --logdir runs
```

A submission must stay under 100,000 characters and answer within 50 ms per turn.

## Port notes

- Java's `Random` and numpy produce different sequences, so a seed gives a different map with the
  same distributions.
- `AUTOPLACE` keeps the original's always-zero A* heuristic (a uniform-cost search), so its paths
  match the real game.
- Map generation reproduces `GridMaker.java`'s quirks: weighted, splitting rivers; round-robin
  region growth; a fallback town-placement pass; sequential filtering of reciprocal connections.
- `Grid` tiles are the authoritative state. `GameState` keeps numpy mirrors in sync through
  `_set_track`.
- `GameSimulator` turns the game's whole-turn interface into one atomic decision at a time.
- Boards are centred on a 20×30 canvas with the padding marked as inked. Index the policy map
  through `pad_offsets`.
- Rollout workers pin torch to one thread and start with `spawn`.

## Legacy

Still working, not maintained: the AlphaZero/MCTS stack
(`python3 -m training.train --config training/configs/default.yaml`), the lite game
(`python3 -m training.train_lite_ppo --config training/configs/ppo_lite.yaml`), and the older
`ppo*.yaml` configs that predate the 28-channel observation.

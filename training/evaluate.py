"""Greedy benchmarking of an agent against each boss tier.

Used both by `evaluate_checkpoints.py` (offline, over saved checkpoints) and by the trainers
themselves, which call `evaluate_all_bosses` right after writing a checkpoint so progress against
each opponent shows up in the training log as it happens.

Games are played greedily (epsilon = 0, i.e. deployment behaviour) on a FIXED set of map seeds,
so successive evaluations differ because the agent changed, not because the maps did.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from railroad_env import RailroadGymEnv
from railroad_env.opponent import BOSS_TIERS

from .agent_model import RailroadUNet
from .ppo import _decode_action
from .simulator import GameSimulator

EVAL_COLUMNS = ("score", "opp", "margin", "win", "conn", "zero")


def play_eval_game(state_dict, model_kwargs, boss, seed, max_turns, allow_skip_disrupt,
                   force_disrupt=False):
    """One greedy game. Top-level and picklable so it can run in a multiprocessing worker.

    `force_disrupt` MUST match what training used. Masking SKIP_DISRUPT out during training
    sends exactly zero gradient to the policy head's third channel (masked_fill blocks it), so
    those weights stay frozen at whatever the warm start left there while the features feeding
    them keep drifting. Offering the action here anyway would score a behaviour the run never
    trained - and this table is how checkpoints get picked."""
    torch.set_num_threads(1)
    model = RailroadUNet(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()

    env = RailroadGymEnv(max_turns=max_turns, opponent_strategy=boss, seed=seed)
    env.reset()
    sim = GameSimulator.from_env(env, allow_skip_disrupt=allow_skip_disrupt,
                                force_disrupt=force_disrupt)

    while not sim.is_game_over():
        mask = sim.get_action_mask()
        with torch.no_grad():
            logits, _ = model(
                torch.from_numpy(sim.get_encoded_state()).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
            )
        index = int(torch.argmax(logits.flatten(1), dim=1).item())
        sim.apply(_decode_action(index, mask.shape, sim.pad_offsets))

    gs = sim.game_state
    return (
        gs.scores[0],
        gs.scores[1],
        sum(len(t.paths) for t in gs.towns),
        sum(len(t.desired_connections) for t in gs.towns),
    )


def evaluate_vs_boss(
    state_dict,
    model_kwargs: dict,
    boss: str,
    seeds: Sequence[int],
    max_turns: int,
    allow_skip_disrupt: bool,
    pool=None,
    force_disrupt: bool = False,
) -> Dict[str, float]:
    args = [(state_dict, model_kwargs, boss, s, max_turns, allow_skip_disrupt, force_disrupt)
            for s in seeds]
    results = pool.starmap(play_eval_game, args) if pool is not None else [play_eval_game(*a) for a in args]

    own = np.array([r[0] for r in results], dtype=float)
    opp = np.array([r[1] for r in results], dtype=float)
    made = sum(r[2] for r in results)
    wanted = sum(r[3] for r in results)
    return {
        "score": own.mean(),
        "opp": opp.mean(),
        "margin": (own - opp).mean(),
        "win": float(np.mean(own > opp)),
        "conn": made / max(1, wanted),
        "zero": float(np.mean(own == 0)),
    }


def evaluate_all_bosses(
    model: torch.nn.Module,
    model_kwargs: dict,
    episodes: int,
    max_turns: int,
    allow_skip_disrupt: bool,
    bosses: Sequence[str] = BOSS_TIERS,
    seed: int = 0,
    pool=None,
    force_disrupt: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Benchmarks a live model against every boss tier. The model is moved to CPU state only for
    the workers; the caller's model is left untouched.

    `force_disrupt` must mirror the training setting - see play_eval_game."""
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    seeds = [seed + i for i in range(episodes)]
    return {
        boss: evaluate_vs_boss(
            state_dict, model_kwargs, boss, seeds, max_turns, allow_skip_disrupt, pool,
            force_disrupt=force_disrupt,
        )
        for boss in bosses
    }


def format_eval_table(results: Dict[str, Dict[str, float]], title: str = "") -> str:
    """The table the trainers print after each checkpoint."""
    header = (
        f"{'boss':<16}{'score':>9}{'opp':>9}{'margin':>10}{'win':>7}{'conn':>7}{'zero':>7}"
    )
    lines = []
    if title:
        lines.append(title)
    lines.append(header)
    lines.append("-" * len(header))
    for boss, r in results.items():
        lines.append(
            f"{boss:<16}{r['score']:>9.0f}{r['opp']:>9.0f}{r['margin']:>+10.0f}"
            f"{r['win']:>7.2f}{r['conn']:>7.0%}{r['zero']:>7.0%}"
        )
    return "\n".join(lines)


def log_eval_to_tensorboard(writer, results: Dict[str, Dict[str, float]], iteration: int) -> None:
    for boss, r in results.items():
        for key, value in r.items():
            writer.add_scalar(f"eval/{boss}/{key}", value, iteration)

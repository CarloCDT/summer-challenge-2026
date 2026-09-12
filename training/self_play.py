"""Plays one full self-play episode with MCTS-selected actions and produces training examples.

Every atomic decision's (state, MCTS visit-count policy) pair is recorded; once the episode ends,
every recorded state gets the SAME value_target - the final score margin, normalized to [-1, 1] -
following the standard AlphaZero self-play labelling scheme (the actual outcome is the training
signal for every position that led to it, not a per-step bootstrap).
"""
from typing import List, Tuple

import numpy as np
import torch

from railroad_env import RailroadGymEnv

from .encoding import BOARD_HEIGHT, BOARD_WIDTH, build_place_mask, compute_pad_offsets, encode_state
from .mcts import MCTS, ACTION_KIND_CHANNEL, score_margin_to_value, visit_count_policy
from .model import TrainUNet
from .replay_buffer import Example
from .simulator import Action, GameSimulator


def _dense_policy(policy: dict, pad_offsets) -> np.ndarray:
    """Board-coordinate policy -> the padded canvas the network sees."""
    pad_top, pad_left = pad_offsets
    dense = np.zeros((2, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for (kind, x, y), p in policy.items():
        dense[ACTION_KIND_CHANNEL[kind], pad_top + y, pad_left + x] = p
    return dense


def play_episode(
    env: RailroadGymEnv,
    mcts: MCTS,
    temperature: float = 1.0,
    temperature_moves: int = 30,
    rng: np.random.RandomState = None,
) -> Tuple[List[Example], List[int]]:
    """Plays one episode from a freshly-reset `env` (caller is responsible for calling
    env.reset() first, so map seeding stays under caller control). Returns
    (list of (state, dense_policy, value_target) examples, final [player0, player1] scores)."""
    rng = rng or np.random.RandomState()
    sim = GameSimulator.from_env(env)

    recorded: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    move_count = 0

    while not sim.is_game_over():
        visit_counts = mcts.run(sim)
        temp = temperature if move_count < temperature_moves else 0.0
        policy = visit_count_policy(visit_counts, temperature=temp)

        dense_policy = _dense_policy(policy, sim.pad_offsets)
        recorded.append((sim.get_encoded_state(), dense_policy, sim.get_action_mask()))

        actions: List[Action] = list(policy.keys())
        probs = np.array(list(policy.values()), dtype=np.float64)
        probs /= probs.sum()  # guard against float drift so np.random.choice doesn't complain
        action = actions[rng.choice(len(actions), p=probs)]
        sim.apply(action)
        move_count += 1

    final_scores = list(sim.game_state.scores)
    value_target = score_margin_to_value(final_scores[0] - final_scores[1])

    examples: List[Example] = [
        (state, policy, value_target, mask) for state, policy, mask in recorded
    ]
    return examples, final_scores


def play_episode_no_search(
    env: RailroadGymEnv,
    model: torch.nn.Module,
    device: str = "cpu",
    temperature: float = 1.0,
    temperature_moves: int = 30,
    rng: np.random.RandomState = None,
) -> Tuple[List[Example], List[int]]:
    """Same as play_episode, but player 0 acts directly off the network's own masked policy
    output instead of running MCTS - one forward pass per atomic decision instead of
    num_simulations-many, and no Node/PUCT tree bookkeeping. The policy_target recorded for
    each state is a one-hot on whichever action was actually sampled (there's no search to
    produce an improved visit-count distribution), so the policy head is trained to imitate its
    own past samples rather than an MCTS-improved target - a much weaker signal, useful for fast
    pipeline iteration, not for producing a strong agent."""
    rng = rng or np.random.RandomState()
    sim = GameSimulator.from_env(env)
    model.eval()

    recorded: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    move_count = 0

    while not sim.is_game_over():
        state = sim.get_encoded_state()
        mask = sim.get_action_mask()
        legal_actions: List[Action] = sim.get_legal_actions()

        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            policy_logits, _ = model(x, m)
            flat_probs = torch.softmax(policy_logits.flatten(1), dim=1).view_as(policy_logits)
            probs = flat_probs[0].cpu().numpy()

        pad_top, pad_left = sim.pad_offsets
        action_probs = np.array(
            [probs[ACTION_KIND_CHANNEL[kind], pad_top + y, pad_left + x] for kind, x, y in legal_actions],
            dtype=np.float64,
        )
        total = action_probs.sum()
        # Illegal cells are masked to ~0 probability by the model itself, so this only ever
        # triggers from float underflow across a huge number of legal cells - fall back to
        # uniform rather than dividing by (near-)zero.
        action_probs = action_probs / total if total > 1e-12 else np.full(len(legal_actions), 1.0 / len(legal_actions))

        temp = temperature if move_count < temperature_moves else 0.0
        if temp < 1e-3:
            action = legal_actions[int(np.argmax(action_probs))]
        else:
            scaled = action_probs ** (1.0 / temp)
            scaled /= scaled.sum()
            action = legal_actions[rng.choice(len(legal_actions), p=scaled)]

        dense_policy = np.zeros((2, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        kind, x, y = action
        dense_policy[ACTION_KIND_CHANNEL[kind], pad_top + y, pad_left + x] = 1.0
        recorded.append((state, dense_policy, mask))

        sim.apply(action)
        move_count += 1

    final_scores = list(sim.game_state.scores)
    value_target = score_margin_to_value(final_scores[0] - final_scores[1])

    examples: List[Example] = [
        (state, policy, value_target, mask) for state, policy, mask in recorded
    ]
    return examples, final_scores


def play_episode_topk(
    env: RailroadGymEnv,
    model: torch.nn.Module,
    device: str = "cpu",
    k: int = 3,
) -> Tuple[List[Example], List[int]]:
    """Simplest possible strategy: DISRUPT is ignored entirely (model has a single PLACE
    channel - see TrainUNet's `policy_channels=1`), and a whole turn is one forward pass - the
    top-k legal cells by score get placed directly, with no per-cell re-evaluation within the
    turn and no MCTS/lookahead. This only works because none of a turn's own placements change
    any OTHER cell's legality (occupied/town/inked status is fixed at turn start), so picking a
    static top-k up front is equivalent to picking them one at a time. Operates directly on the
    real `env` (no GameSimulator) since there's no search needing an isolated clone."""
    model.eval()
    gs = env.game_state

    recorded: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    while not gs.is_done():
        state = encode_state(gs)
        mask = build_place_mask(gs, gs.paint_points[0])  # (1, H, W), PLACE only

        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            policy_logits, _ = model(x, m)
            logits = policy_logits[0, 0].cpu().numpy()  # (H, W)

        pad_top, pad_left = compute_pad_offsets(gs.height, gs.width)
        legal_cells = list(zip(*np.where(mask[0] > 0)))  # canvas [(row, col), ...]
        if not legal_cells:
            env.step({"actions": [(0, (0, 0))]})
            continue

        legal_cells.sort(key=lambda yx: logits[yx[0], yx[1]], reverse=True)
        chosen = legal_cells[:k]

        dense_policy = np.zeros((1, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        for y, x in chosen:
            dense_policy[0, y, x] = 1.0 / len(chosen)
        recorded.append((state, dense_policy, mask))

        env.step({"actions": [(1, (int(col) - pad_left, int(row) - pad_top)) for row, col in chosen]})

    final_scores = list(gs.scores)
    value_target = score_margin_to_value(final_scores[0] - final_scores[1])

    examples: List[Example] = [
        (state, policy, value_target, mask) for state, policy, mask in recorded
    ]
    return examples, final_scores


def self_play_worker(
    state_dict_cpu: dict,
    policy_channels: int,
    strategy: str,
    strategy_kwargs: dict,
    env_kwargs: dict,
    seed: int,
) -> Tuple[List[Example], List[int]]:
    """Runs one self-play game to completion, for use as a `multiprocessing` worker (see
    `training/train.py`'s `num_workers`). Always runs on CPU regardless of what device the main
    process trains on: workers are meant to run many-at-once, a single-image forward pass gains
    little from a GPU anyway, and CPU-only keeps every worker's CUDA footprint at zero instead
    of N worker processes each opening their own CUDA context. Must stay a top-level function
    (not a closure) so it's picklable for the `spawn` start method, which re-imports this module
    fresh in each worker process rather than forking the parent's already-initialized CUDA
    context (unsafe to mix with CUDA - see train.py).

    Pins itself to a single CPU thread: PyTorch otherwise defaults every process to
    intra-op-parallelizing across ALL cores, so N worker processes each doing that means N*(all
    cores) threads fighting over (all cores) - measured ~28x CPU-time-vs-wall-time oversubscription
    with 4 workers on this 32-core box, turning 4 tiny single-image forward passes into a ~30s
    stall. The real parallelism here is meant to come from separate processes, not from each one
    also multithreading internally - the model and inputs are far too small for that to help."""
    torch.set_num_threads(1)
    model = TrainUNet(policy_channels=policy_channels)
    model.load_state_dict(state_dict_cpu)
    model.eval()

    rng = np.random.RandomState(seed)
    env = RailroadGymEnv(seed=int(rng.randint(0, 2**31 - 1)), **env_kwargs)
    env.reset()

    if strategy == "topk":
        return play_episode_topk(env, model, device="cpu", k=strategy_kwargs["topk"])

    if strategy == "mcts":
        mcts = MCTS(
            model,
            device="cpu",
            c_puct=strategy_kwargs["c_puct"],
            num_simulations=strategy_kwargs["num_simulations"],
            eval_batch_size=strategy_kwargs["eval_batch_size"],
        )
        return play_episode(
            env, mcts, temperature=1.0, temperature_moves=strategy_kwargs["temperature_moves"], rng=rng
        )

    return play_episode_no_search(
        env, model, device="cpu", temperature=1.0, temperature_moves=strategy_kwargs["temperature_moves"], rng=rng
    )

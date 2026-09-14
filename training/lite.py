"""Training pieces specific to the lite game: encoding, simulator, and PPO rollout.

The PPO algorithm itself (GAE, the clipped-surrogate update) is shared with the full game - see
`training/ppo.py`. Only the observation shape, the action space (PLACE only) and the environment
differ here.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from railroad_env.game_state import GameState
from railroad_lite_env import LITE_CHANNELS, NUM_LITE_CHANNELS, RailroadLiteEnv, lite_observation

from .encoding import BOARD_HEIGHT, BOARD_WIDTH, compute_pad_offsets
from .lite_model import LiteUNet
from .ppo import Transition, _gae
from .simulator import GameSimulator

# Index of the "off-board / inked" channel within the lite layout - set to 1.0 on the padding.
LITE_INKED_CHANNEL = LITE_CHANNELS.index(GameState.INKED_CHANNEL)

# Lite games are 25 turns at 3 placements each, so scores land in the hundreds rather than the
# tens of thousands the full game reaches. Rewards are divided by this to keep returns - and
# therefore the value loss - at an O(1) scale.
LITE_SCORE_NORM = 50.0


def lite_encode_state(game_state: GameState) -> np.ndarray:
    """(NUM_LITE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH) float32, channels-first, board centered in
    the canvas with the surrounding padding marked as inked/unbuildable."""
    obs = lite_observation(game_state.get_observation())  # (h, w, 16)
    h, w, _ = obs.shape
    pad_top, pad_left = compute_pad_offsets(h, w)

    canvas = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, NUM_LITE_CHANNELS), dtype=np.float32)
    canvas[pad_top:pad_top + h, pad_left:pad_left + w, :] = obs

    on_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    on_board[pad_top:pad_top + h, pad_left:pad_left + w] = True
    canvas[:, :, LITE_INKED_CHANNEL] = np.where(on_board, canvas[:, :, LITE_INKED_CHANNEL], 1.0)

    return np.transpose(canvas, (2, 0, 1)).astype(np.float32)


class LiteGameSimulator(GameSimulator):
    """GameSimulator with the lite game's observation and action space: 16 input channels and a
    single PLACE policy channel. Because `_compute_legal` never returns DISRUPT actions, the
    base class's phase cascade resolves the turn as soon as the paint budget runs out."""

    def get_encoded_state(self) -> np.ndarray:
        return lite_encode_state(self.game_state)

    def get_action_mask(self) -> np.ndarray:
        """(1, BOARD_HEIGHT, BOARD_WIDTH), 1.0 = a track may legally be placed there."""
        mask = np.zeros((1, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        if self._done:
            return mask
        pad_top, pad_left = self.pad_offsets
        for _, x, y in self._legal_actions_cache:
            mask[0, pad_top + y, pad_left + x] = 1.0
        return mask

    def _compute_legal(self, phase: str) -> List[Tuple[str, int, int]]:
        if phase == "disrupt":
            return []  # the lite game has no disruption
        return super()._compute_legal(phase)


def _decode_lite_action(flat_index: int, pad_offsets) -> Tuple[str, int, int]:
    """Flat index over the (1, H, W) masked policy -> a PLACE action in board coordinates."""
    row, col = divmod(int(flat_index), BOARD_WIDTH)
    pad_top, pad_left = pad_offsets
    return ("PLACE", col - pad_left, row - pad_top)


def play_one_lite_game(
    model: torch.nn.Module,
    env_kwargs: dict,
    device: str,
    gamma: float,
    gae_lambda: float,
    epsilon: float,
    seed: int,
) -> Tuple[List[Transition], List[int], int]:
    """One full lite game, one PLACE decision per PPO timestep. Returns (transitions, final
    [player0, player1] scores, episode length). Builds its own RNG from `seed` so it is safe to
    call inside a worker process."""
    local_rng = np.random.RandomState(seed)
    env = RailroadLiteEnv(seed=int(local_rng.randint(0, 2**31 - 1)), **env_kwargs)
    env.reset()
    sim = LiteGameSimulator.from_env(env)

    states, masks, action_idxs, log_probs, values, rewards = [], [], [], [], [], []

    while not sim.is_game_over():
        state = sim.get_encoded_state()
        mask = sim.get_action_mask()

        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits, value = model(x, m)
            flat = logits.flatten(1)
            dist = torch.distributions.Categorical(logits=flat)
            # epsilon-greedy: a purely stochastic policy almost never assembles the contiguous
            # corridor a connection needs, while argmax rides the network's spatial correlation
            # and clusters picks near existing track. See git history of PROJECT_SUMMARY.md.
            action_idx = dist.sample() if local_rng.rand() < epsilon else torch.argmax(flat, dim=1)
            log_prob = dist.log_prob(action_idx)

        turn_before = sim.game_state.turn
        sim.apply(_decode_lite_action(int(action_idx.item()), sim.pad_offsets))
        # last_turn_reward is only meaningful on the decision that actually resolved a turn -
        # otherwise it still holds a previous resolution's value.
        raw_reward = sim.last_turn_reward if sim.game_state.turn != turn_before else 0.0

        states.append(state)
        masks.append(mask)
        action_idxs.append(int(action_idx.item()))
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))
        rewards.append(raw_reward / LITE_SCORE_NORM)

    advantages, returns = _gae(rewards, values, gamma, gae_lambda)
    transitions: List[Transition] = [
        {
            "state": states[i], "mask": masks[i], "action_idx": action_idxs[i],
            "log_prob": log_probs[i], "advantage": advantages[i], "return": returns[i],
        }
        for i in range(len(states))
    ]
    return transitions, list(sim.game_state.scores), len(states)


def lite_rollout_worker(
    state_dict_cpu: dict,
    model_kwargs: dict,
    env_kwargs: dict,
    gamma: float,
    gae_lambda: float,
    epsilon: float,
    seed: int,
) -> Tuple[List[Transition], List[int], int]:
    """Top-level, picklable multiprocessing worker - one game per call. Always CPU and pinned to
    a single thread: PyTorch otherwise intra-op-parallelizes every process across all cores,
    which oversubscribes badly once several workers run at once."""
    torch.set_num_threads(1)
    model = LiteUNet(**model_kwargs)
    model.load_state_dict(state_dict_cpu)
    model.eval()
    return play_one_lite_game(model, env_kwargs, "cpu", gamma, gae_lambda, epsilon, seed)


def collect_lite_rollout(
    env_kwargs: dict,
    model: torch.nn.Module,
    model_kwargs: dict,
    device: str,
    gamma: float,
    gae_lambda: float,
    games_per_iteration: int,
    rng: np.random.RandomState,
    epsilon: float = 0.0,
    pool=None,
) -> Tuple[List[Transition], List[List[int]], List[int]]:
    """Plays `games_per_iteration` lite games. Seeds are always drawn here in the main process,
    so a given top-level seed reproduces the same games regardless of worker count."""
    model.eval()
    seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(games_per_iteration)]

    if pool is not None:
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        results = pool.starmap(
            lite_rollout_worker,
            [(state_dict_cpu, model_kwargs, env_kwargs, gamma, gae_lambda, epsilon, s) for s in seeds],
        )
    else:
        results = [
            play_one_lite_game(model, env_kwargs, device, gamma, gae_lambda, epsilon, s)
            for s in seeds
        ]

    all_transitions: List[Transition] = []
    final_scores_list: List[List[int]] = []
    episode_lengths: List[int] = []
    for transitions, final_scores, ep_len in results:
        all_transitions.extend(transitions)
        final_scores_list.append(final_scores)
        episode_lengths.append(ep_len)

    return all_transitions, final_scores_list, episode_lengths

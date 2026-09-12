"""PPO rollout collection and update step for the full Railroad Tycoon env.

Each atomic decision is a single PPO timestep, so the policy gets a proper per-step advantage.
`RailroadUNet` is the shared actor-critic; its policy head is a spatial heatmap with one channel
per action kind:

    channel 0  PLACE          - which cell to build on (compulsory: the paint budget must be spent)
    channel 1  DISRUPT        - which region to destabilise
    channel 2  SKIP_DISRUPT   - decline to disrupt this turn (only with allow_skip_disrupt)

GameSimulator handles the sub-turn accumulation and phase cascading.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from railroad_env import RailroadGymEnv

from .agent_model import RailroadUNet
from .simulator import ACTION_KIND_CHANNEL, GameSimulator

# Per-turn score deltas run from 0 to a few hundred points in a 100-turn game. Rewards are divided
# by a score_norm before GAE to keep returns - and therefore the value loss - at a numerically
# comfortable O(1) scale. The critic is unbounded, so this is about conditioning, not about
# fitting inside an output range; set it near a typical final score.
DEFAULT_SCORE_NORM = 1000.0

#: Reserved `opponent_pool` key: play this game against the agent's own current weights.
SELF_PLAY = "self"


def _sample_opponent(pool: Dict[str, float], rng: np.random.RandomState) -> str:
    """Draw one opponent name from a {name: weight} pool. Weights need not sum to 1."""
    names = sorted(pool)
    weights = np.array([float(pool[n]) for n in names], dtype=np.float64)
    if weights.min() < 0 or weights.sum() <= 0:
        raise ValueError(f"opponent_pool weights must be non-negative and sum > 0, got {pool}")
    return str(rng.choice(names, p=weights / weights.sum()))

Transition = Dict[str, object]  # state, mask, action_idx, log_prob, value, reward, advantage, return


def _gae(rewards: List[float], values: List[float], gamma: float, gae_lambda: float) -> Tuple[List[float], List[float]]:
    """Standard GAE(lambda): bootstraps with a terminal value of 0 (these episodes end for
    real - max_turns - not cut off mid-trajectory). advantage[t] = sum_l (gamma*lambda)^l * delta[t+l],
    delta[t] = reward[t] + gamma*value[t+1] - value[t]. Returns (advantages, returns) where
    returns[t] = advantages[t] + values[t] (the standard PPO value-function regression target)."""
    values_with_bootstrap = values + [0.0]
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_with_bootstrap[t + 1] - values_with_bootstrap[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


_CHANNEL_KIND = {v: k for k, v in ACTION_KIND_CHANNEL.items()}


def _decode_action(flat_index: int, mask_shape, pad_offsets):
    """Flat index over the (C, H, W) masked policy -> a GameSimulator action in board coords."""
    _, height, width = mask_shape
    channel, remainder = divmod(flat_index, height * width)
    row, col = divmod(remainder, width)
    pad_top, pad_left = pad_offsets
    return (_CHANNEL_KIND[channel], col - pad_left, row - pad_top)


def _play_one_game(
    model: torch.nn.Module,
    env_kwargs: dict,
    device: str,
    gamma: float,
    gae_lambda: float,
    epsilon: float,
    seed: int,
    allow_skip_disrupt: bool = True,
    score_norm: float = DEFAULT_SCORE_NORM,
    reward_mode: str = "own",
    force_disrupt: bool = False,
    action_selection: str = "epsilon_greedy",
    opponent_pool: Optional[Dict[str, float]] = None,
    self_play_epsilon: float = 0.0,
    win_bonus: float = 0.0,
) -> Tuple[List[Transition], List[int], int]:
    """Plays one full game with the current policy, one atomic decision per PPO step.
    Returns (that game's transitions, its final [player0, player1] scores, its episode length).
    Takes a single `seed` (not a shared RandomState) and builds its own local RNG from it, so
    this is safe to call from a separate worker process - see `_rollout_worker`/`collect_rollout`.

    `action_selection` picks how an action is drawn, and it is a correctness knob, not a taste
    one:

    "epsilon_greedy" (legacy, the default so existing configs are unchanged): argmax with
    probability 1-`epsilon`, otherwise a draw from the policy. `log_prob` is recorded as the
    policy's own log-probability of whatever was taken, so the PPO ratio is pi_new/pi_old and
    starts at exactly 1 - that part is correct, and measurement confirms it.

    What is wrong is the SAMPLING DISTRIBUTION. Transitions are drawn from the behaviour mixture
    mu = (1-epsilon)*argmax + epsilon*pi, while PPO's surrogate is an expectation over a ~ pi_old.
    Estimating it from mu is biased: argmax actions are wildly over-represented relative to their
    probability under pi (at `20260910-224257_iter200` the median pi(argmax) is 0.044 for an
    action taken ~80% of the time), so the update chases whatever the argmax already prefers and
    the approx_kl/clip_frac diagnostics are measured on the wrong distribution.

    "onpolicy": always draw from the masked categorical, making the behaviour policy the policy
    and the estimator unbiased. The old docstring justified argmax with "pure sampling scored
    0/20 games", but that was measured on an untrained net; warm-starting from a competent
    checkpoint removes the concern.

    NOT a clip_frac fix - measured, it slightly RAISES it. When pi(a) is small a modest logit
    step is a large relative probability change, so ratios move fast; sampling visits those
    low-probability actions more often than argmax does. clip_frac is a step-size question
    (`lr`, `clip_epsilon`, `ppo_epochs`), independent of this knob.

    `opponent_pool` maps opponent name -> probability, sampled once per game from this game's own
    RNG, so a rollout batch mixes opponents instead of overfitting to one. The reserved name
    "self" swaps in `SelfPlayOpponent` driven by this same model."""
    local_rng = np.random.RandomState(seed)

    # Sample this game's opponent. "self" is not a registered strategy, so the env is built with
    # a harmless placeholder and the real opponent is swapped in before the simulator deep-copies
    # it - GameSimulator reads env.opponent in its constructor.
    kwargs = dict(env_kwargs)
    opponent_name = _sample_opponent(opponent_pool, local_rng) if opponent_pool else None
    if opponent_name is not None:
        kwargs["opponent_strategy"] = "passive" if opponent_name == SELF_PLAY else opponent_name

    env = RailroadGymEnv(seed=int(local_rng.randint(0, 2**31 - 1)), **kwargs)
    env.reset()
    if opponent_name == SELF_PLAY:
        from .self_play_opponent import SelfPlayOpponent
        env.opponent = SelfPlayOpponent(
            model, seed=int(local_rng.randint(0, 2**31 - 1)), epsilon=self_play_epsilon,
            allow_skip_disrupt=allow_skip_disrupt, force_disrupt=force_disrupt,
        )
    sim = GameSimulator.from_env(
        env, allow_skip_disrupt=allow_skip_disrupt, reward_mode=reward_mode,
        force_disrupt=force_disrupt,
    )

    states, masks, action_idxs, log_probs, values, rewards = [], [], [], [], [], []

    while not sim.is_game_over():
        state = sim.get_encoded_state()
        mask = sim.get_action_mask()  # (C, H, W) - see this module's docstring

        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits, value = model(x, m)
            dist = torch.distributions.Categorical(logits=logits.flatten(1))
            if action_selection == "onpolicy":
                action_idx = dist.sample()
            elif local_rng.rand() < epsilon:
                action_idx = dist.sample()
            else:
                action_idx = torch.argmax(logits.flatten(1), dim=1)
            log_prob = dist.log_prob(action_idx)

        turn_before = sim.game_state.turn
        sim.apply(_decode_action(int(action_idx.item()), mask.shape, sim.pad_offsets))
        # last_turn_reward is only meaningful on the decision that actually resolved a
        # turn - it's a stale value from a previous resolution otherwise, since it's never
        # reset to None on a non-resolving apply().
        #
        # SCORE_NORM keeps the regression target near unit scale. (RailroadUNet's value head is
        # UNBOUNDED - the Tanh this comment used to cite belongs to the legacy AlphaZero
        # model.py, not here - but raw per-turn score deltas in the hundreds would still swamp
        # the value loss against a policy loss of order 0.01, and would need a learning rate
        # this network does not use.)
        raw_reward = sim.last_turn_reward if sim.game_state.turn != turn_before else 0.0
        reward = raw_reward / score_norm

        states.append(state)
        masks.append(mask)
        action_idxs.append(int(action_idx.item()))
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))
        rewards.append(reward)

    # Terminal win/loss bonus, in raw score units, applied to the final decision only.
    #
    # WHY: the arena ranks on match outcomes while `reward_mode: margin` optimises the score
    # difference, and the two measurably come apart - in a 250-game round robin, 3 of 10 pairings
    # had margin and win rate pointing in OPPOSITE directions. Margin alone says a +12,000 win and
    # a +200 win differ by 60x; the ladder says they are worth the same, and that a +200 win and a
    # -200 loss are the whole game. This term buys back that distinction. A draw pays nothing.
    #
    # SCALE, and this is the part to watch. The bonus is divided by score_norm like every other
    # reward, so win_bonus=5000 is +-5.0 on the final step. Measured on 114307_iter300, the mean
    # |discounted return| per decision is 4.34 against level2Pro but 0.18 against level2Silver and
    # 0.08 in self-play - so the same bonus is a modest nudge against the builder bosses and
    # completely dominates the signal against the contested ones. That is deliberate here (those
    # are exactly the games whose outcome is in doubt), but it does make the opponent mix matter
    # even more than it already did, and it is the reason per-opponent reward normalisation is
    # worth doing before adding anything else to the reward.
    #
    # REACH: GAE credits a terminal reward directly over ~1/(1 - gamma*lambda) decisions and
    # delegates the rest to the critic - ~19 decisions at lambda=0.95, ~44 at 0.98, against ~250
    # decisions in a game. Raising gae_lambda is therefore not an independent knob from this one:
    # it decides how much of the game actually feels the bonus first-hand.
    if win_bonus and rewards:
        own, foe = sim.game_state.scores[0], sim.game_state.scores[1]
        outcome = (own > foe) - (own < foe)  # +1 win, -1 loss, 0 draw
        rewards[-1] += outcome * win_bonus / score_norm

    advantages, returns = _gae(rewards, values, gamma, gae_lambda)
    transitions: List[Transition] = [
        {
            "state": states[i], "mask": masks[i], "action_idx": action_idxs[i],
            "log_prob": log_probs[i], "advantage": advantages[i], "return": returns[i],
        }
        for i in range(len(states))
    ]
    return transitions, list(sim.game_state.scores), len(states)


def _rollout_worker(
    state_dict_cpu: dict,
    model_kwargs: dict,
    env_kwargs: dict,
    gamma: float,
    gae_lambda: float,
    epsilon: float,
    seed: int,
    allow_skip_disrupt: bool = True,
    score_norm: float = DEFAULT_SCORE_NORM,
    reward_mode: str = "own",
    force_disrupt: bool = False,
    action_selection: str = "epsilon_greedy",
    opponent_pool: Optional[Dict[str, float]] = None,
    self_play_epsilon: float = 0.0,
    win_bonus: float = 0.0,
) -> Tuple[List[Transition], List[int], int]:
    """Top-level, picklable multiprocessing worker - one game per call. Always runs on CPU
    and pins itself to a single thread, for the exact same reasons as
    training/self_play.py::self_play_worker (see its docstring): a single-image forward pass
    gains little from a GPU, and N worker processes each defaulting to intra-op-parallelizing
    across every core causes massive oversubscription.

    Self-play costs no extra IPC: the opponent reuses the very state dict already shipped here
    for the agent, so a self-play game pickles nothing beyond a normal one."""
    torch.set_num_threads(1)
    model = RailroadUNet(**model_kwargs)
    model.load_state_dict(state_dict_cpu)
    model.eval()
    return _play_one_game(
        model, env_kwargs, "cpu", gamma, gae_lambda, epsilon, seed,
        allow_skip_disrupt=allow_skip_disrupt, score_norm=score_norm, reward_mode=reward_mode,
        force_disrupt=force_disrupt, action_selection=action_selection,
        opponent_pool=opponent_pool, self_play_epsilon=self_play_epsilon,
        win_bonus=win_bonus,
    )


def collect_rollout(
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
    allow_skip_disrupt: bool = True,
    score_norm: float = DEFAULT_SCORE_NORM,
    reward_mode: str = "own",
    force_disrupt: bool = False,
    action_selection: str = "epsilon_greedy",
    opponent_pool: Optional[Dict[str, float]] = None,
    self_play_epsilon: float = 0.0,
    win_bonus: float = 0.0,
) -> Tuple[List[Transition], List[List[int]], List[int]]:
    """Plays `games_per_iteration` full games with the current policy. Returns (all
    transitions across all games, each game's final [player0, player1] scores, each game's
    episode length).

    `pool`: an optional persistent `multiprocessing` pool (see train_ppo.py's `num_workers`) -
    when given, the games_per_iteration games are dispatched across it (real parallelism, one
    game per worker process) instead of played one after another in this process. Seeds are
    always drawn sequentially from `rng` here in the main process (never inside a worker), so a
    given top-level seed reproduces the same per-game seeds regardless of num_workers."""
    model.eval()
    seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(games_per_iteration)]

    if pool is not None:
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        results = pool.starmap(
            _rollout_worker,
            [
                # POSITIONAL - starmap, so this tuple must track _rollout_worker's signature
                # exactly. A new parameter goes on the end of both, or every argument after the
                # insertion point silently shifts by one.
                (state_dict_cpu, model_kwargs, env_kwargs, gamma, gae_lambda, epsilon, s,
                 allow_skip_disrupt, score_norm, reward_mode, force_disrupt, action_selection,
                 opponent_pool, self_play_epsilon, win_bonus)
                for s in seeds
            ],
        )
    else:
        results = [
            _play_one_game(model, env_kwargs, device, gamma, gae_lambda, epsilon, s,
                           allow_skip_disrupt=allow_skip_disrupt, score_norm=score_norm,
                           reward_mode=reward_mode, force_disrupt=force_disrupt,
                           action_selection=action_selection, opponent_pool=opponent_pool,
                           self_play_epsilon=self_play_epsilon, win_bonus=win_bonus)
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


def ppo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    ppo_epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    device: str,
) -> Dict[str, float]:
    """Standard clipped-surrogate PPO update over `ppo_epochs` passes of shuffled minibatches
    from `transitions` (collected under the policy BEFORE this update - the clipped ratio is
    exactly what keeps reusing that stale data for several epochs safe)."""
    states = torch.from_numpy(np.stack([t["state"] for t in transitions])).to(device)
    masks = torch.from_numpy(np.stack([t["mask"] for t in transitions])).to(device)
    actions = torch.tensor([t["action_idx"] for t in transitions], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([t["log_prob"] for t in transitions], dtype=torch.float32, device=device)
    returns = torch.tensor([t["return"] for t in transitions], dtype=torch.float32, device=device)

    advantages = torch.tensor([t["advantage"] for t in transitions], dtype=torch.float32, device=device)

    # Explained variance of the critic, measured BEFORE advantages are normalised (normalising
    # destroys the scale this needs). return - value IS the raw advantage, so no extra forward
    # pass is required. 1.0 = perfect critic, 0.0 = no better than predicting the mean return,
    # < 0 = worse than the mean. Healthy PPO sits at 0.8-0.95; measured 0.44 on iter200, which
    # matters because GAE's real-reward horizon is only 1/(1 - gamma*lambda) ~ 19 decisions
    # (~6 turns) - every consequence beyond that is the critic's job, so this number bounds how
    # much long-horizon credit assignment is actually working. Part of the shortfall is
    # irreducible (stochastic policy, stochastic opponent), so watch the trend, not the level.
    return_var = returns.var()
    explained_variance = (
        float("nan") if return_var == 0 else float(1.0 - (advantages.var() / return_var))
    )

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(transitions)
    model.train()
    # BatchNorm MUST stay in eval mode here, and this is not a tuning preference.
    #
    # Rollout runs under model.eval(), so old_log_probs come from the running statistics. In
    # train mode BatchNorm switches to per-minibatch statistics, so the network computes a
    # DIFFERENT function - and BatchNorm sits in the conv trunk, so it moves the policy logits
    # directly. Measured on an unmodified iter200 with identical weights and zero gradient steps:
    # eval mode gives ratio == 1.0000 for every sample, train mode gives a median ratio of 0.43
    # with 86% of samples already outside the clip range. PPO's premise (pi_old generated the
    # data) is violated before the first optimiser step, which is why train/clip_frac sat at
    # 0.4-0.8 for whole runs and why it barely moved when lr was cut 10x.
    #
    # Freezing the running stats is also what deployment does: bake_agent.py folds BatchNorm
    # into the preceding convolution out of running_mean/running_var, so the eval-mode network
    # IS the submitted network. Training in train mode optimised a function that was neither the
    # one that collected the data nor the one that ships.
    #
    # Dropout is deliberately left alive - it sits only in the value head and never touches the
    # policy logits, so it regularises the critic without perturbing the ratio.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "clip_frac": [], "approx_kl": []}

    for _ in range(ppo_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]

            logits, values = model(states[idx], masks[idx])
            dist = torch.distributions.Categorical(logits=logits.flatten(1))
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(1), returns[idx])

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["clip_frac"].append(((ratio - 1.0).abs() > clip_epsilon).float().mean().item())
                # Schulman's k3 estimator, (r - 1) - log r. The obvious k1, (old - new).mean(),
                # is unbiased only in expectation and routinely goes NEGATIVE on a single
                # minibatch - earlier runs logged -0.14 - which makes it useless for spotting a
                # policy that has moved too far. k3 is non-negative by construction and has
                # lower variance, at no extra cost since `ratio` is already computed.
                log_ratio = new_log_probs - old_log_probs[idx]
                stats["approx_kl"].append(((ratio - 1.0) - log_ratio).mean().item())
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.mean().item())

    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["explained_variance"] = explained_variance
    return out

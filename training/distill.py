"""Distil a large PPO teacher into a small student that fits CodinGame's source limit.

Why distillation rather than training the small net with PPO from scratch: PPO delivers one
scalar reward per turn, while distillation delivers a gradient across all ~1800 action cells of
every state (KL against the teacher's masked policy). For a capacity-constrained student that
difference in signal density is decisive, and it also makes this the cheapest possible test of
whether the small architecture has enough capacity at all - agreement with the teacher tells you
within an hour what a cold PPO run would take a day to reveal.

The student is capped by the teacher, so the intended follow-up is a PPO fine-tune starting from
the distilled weights (which, unlike a fresh small net, is a good initialisation).

VALUE DISTILLATION (`value_loss_weight`). The loss above is KL on the policy logits alone, so the
student's value head never receives a gradient and ships with its random initialisation - which is
why `bake_debug_agent.py` printed a P(win) that barely moved. Setting `value_loss_weight` adds an
MSE term against the teacher's value, computed once per unique state. It is free in submission
size, because `bake_agent.py` drops the critic entirely; it is only the debug bake and any future
PPO fine-tune that read the value head, and the fine-tune is the case that actually matters - a
warm start whose critic is random throws away its first iterations re-fitting one.

Scale, so the weight can be reasoned about rather than guessed. Measured on teacher
`20260911-114307_iter200` over 261 states: value mean 1.58, std 1.41, so a critic that predicts
zero starts at MSE ~4.5 and one that predicts the mean sits at ~2.0. Policy KL in the shipped run
went 1.14 -> 0.57. A weight of 0.25 therefore starts the two terms at roughly the same magnitude.
Watch `distill/argmax_agreement` against a value_loss_weight=0 run: the policy is what ships, and
the value term shares the trunk with it.

DAgger: rolling out only the teacher would train the student on states the student itself never
reaches. `student_rollout_prob` ramps from 0 to 1 so that later iterations collect states from
the student's own trajectory while the labels still come from the teacher.

    python3 -u -m training.distill --config training/configs/distill.yaml
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from railroad_env import RailroadGymEnv
from railroad_env.opponent import BOSS_TIERS, OPPONENT_STRATEGIES

from .agent_model import RailroadUNet
from .quantization import INT4_QMAX, INT8_QMAX, attach_fake_quant, plain_state_dict
from .evaluate import evaluate_all_bosses, format_eval_table, log_eval_to_tensorboard
from .ppo import SELF_PLAY, _decode_action, _sample_opponent
from .simulator import GameSimulator

MASK_FILL = -1e9


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over the whole (C, H, W) action plane, illegal cells removed."""
    logits = logits.flatten(1).masked_fill(~mask.flatten(1), MASK_FILL)
    return F.log_softmax(logits, dim=1)


def _play_and_label(args):
    """One game. Returns per-turn states + teacher logits + teacher values, and per-decision masks.

    The observation is constant for every sub-decision within a turn - only the mask changes as
    the paint budget is spent - so the teacher runs once per turn and the states are deduplicated
    against the decisions that share them. That cuts both the data shipped back from the worker
    and the student forwards during training by roughly 5x.
    """
    (teacher_sd, teacher_kwargs, student_sd, student_kwargs,
     seed, max_turns, opponent, use_student, epsilon, force_disrupt) = args
    torch.set_num_threads(1)

    teacher = RailroadUNet(**teacher_kwargs)
    teacher.load_state_dict(teacher_sd)
    teacher.eval()
    actor = teacher
    if use_student:
        actor = RailroadUNet(**student_kwargs)
        actor.load_state_dict(student_sd)
        actor.eval()

    env = RailroadGymEnv(max_turns=max_turns, opponent_strategy=opponent, seed=seed)
    env.reset()
    # force_disrupt must mirror the TEACHER's training. A teacher trained with it never gave
    # its SKIP_DISRUPT channel a gradient, so labelling states where that action is offered
    # teaches the student to imitate frozen logits.
    sim = GameSimulator.from_env(env, allow_skip_disrupt=teacher_kwargs["policy_channels"] >= 3,
                                 force_disrupt=force_disrupt)
    rng = np.random.RandomState(seed)

    # tvalues is the critic's label. It is per STATE, not per decision, and is kept in float32
    # while the logits are float16: a regression target has no ~1800-way softmax to hide rounding
    # in, and one scalar per state costs nothing to ship back from the worker.
    states, tlogits, tvalues, masks, index = [], [], [], [], []
    cur_state_key = None

    while not sim.is_game_over():
        state = sim.get_encoded_state()
        mask = sim.get_action_mask()

        key = state.tobytes()
        if key != cur_state_key:
            with torch.no_grad():
                tl, tv = teacher(torch.from_numpy(state).unsqueeze(0))
                al = tl if actor is teacher else actor(torch.from_numpy(state).unsqueeze(0))[0]
            states.append(state.astype(np.float16))
            tlogits.append(tl[0].numpy().astype(np.float16))
            tvalues.append(float(tv[0]))
            cur_state_key = key
            cur_actor_logits = al[0].numpy()
        index.append(len(states) - 1)
        masks.append(mask.astype(bool))

        # The roll-out policy picks the move (teacher early, student once DAgger ramps up); the
        # label always comes from the teacher.
        scored = np.where(mask > 0, cur_actor_logits, -np.inf)
        if epsilon > 0 and rng.rand() < epsilon:
            legal = np.flatnonzero(mask.ravel() > 0)
            choice = int(rng.choice(legal))
        else:
            choice = int(np.argmax(scored))
        sim.apply(_decode_action(choice, mask.shape, sim.pad_offsets))

    return (np.stack(states), np.stack(tlogits), np.asarray(tvalues, dtype=np.float32),
            np.stack(masks), np.asarray(index, dtype=np.int64))


def collect(pool, teacher, teacher_kwargs, student, student_kwargs, seeds,
            max_turns, opponent, student_rollout_prob, epsilon, rng, force_disrupt=False,
            opponent_pool=None):
    """`opponent_pool` ({name: weight}) samples a different opponent per game.

    Distillation only ever teaches the student on the states it actually visits, so the opponent
    IS the syllabus. Rolling out against a single easy boss produces a student drilled on
    positions from games won 1.00 by +12,819, and never shown the inked endgames that decide
    real matches - the teacher's competence there cannot transfer if those positions are absent
    from the data. Mirror the mix the agent will face."""
    t_sd = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
    # quantized=True: a DAgger rollout should explore the states the SHIPPED agent visits.
    s_sd = {k: v.cpu() for k, v in plain_state_dict(student, quantized=True).items()}
    jobs = [
        (t_sd, teacher_kwargs, s_sd, student_kwargs, int(s), max_turns,
         _sample_opponent(opponent_pool, rng) if opponent_pool else opponent,
         bool(rng.rand() < student_rollout_prob), epsilon, force_disrupt)
        for s in seeds
    ]
    results = pool.map(_play_and_label, jobs) if pool else [_play_and_label(j) for j in jobs]

    states, tlogits, tvalues, masks, index = [], [], [], [], []
    offset = 0
    for st, tl, tv, mk, ix in results:
        states.append(st); tlogits.append(tl); tvalues.append(tv)
        masks.append(mk); index.append(ix + offset)
        offset += len(st)
    return (np.concatenate(states), np.concatenate(tlogits), np.concatenate(tvalues),
            np.concatenate(masks), np.concatenate(index))


def distill(
    teacher_checkpoint: str,
    channels=(36, 36, 40),
    value_hidden: int = 128,
    num_iterations: int = 300,
    games_per_iteration: int = 32,
    num_workers: int = 16,
    epochs_per_iteration: int = 4,
    minibatch_size: int = 64,
    lr: float = 1e-3,
    lr_end: float = 1e-5,
    max_grad_norm: float = 1.0,
    value_loss_weight: float = 0.0,
    value_huber: Optional[float] = 2.0,
    max_turns: int = 30,
    opponent: str = "level2",
    opponent_pool: Optional[dict] = None,
    epsilon: float = 0.1,
    dagger_warmup: int = 30,
    eval_every: int = 25,
    eval_episodes: int = 100,
    eval_bosses: str = ",".join(BOSS_TIERS),
    quant_aware: Optional[str] = None,
    checkpoint_every: int = 25,
    checkpoint_dir: str = "checkpoints",
    log_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.RandomState(seed)
    run_name = time.strftime("%Y%m%d-%H%M%S") + "-distill"
    writer = SummaryWriter(log_dir or f"runs/{run_name}")
    print(f"Distilling on {device}; logs in {writer.log_dir}")

    blob = torch.load(teacher_checkpoint, map_location="cpu")
    teacher_kwargs = blob["model_kwargs"]
    teacher = RailroadUNet(**teacher_kwargs)
    teacher.load_state_dict(blob["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Inherited from the teacher, never chosen here: the student is being taught to imitate this
    # teacher, so it has to be taught on the action space the teacher was trained on.
    if opponent_pool:
        unknown = [n for n in opponent_pool if n != SELF_PLAY and n not in OPPONENT_STRATEGIES]
        if unknown:
            raise ValueError(f"opponent_pool has unknown opponents {unknown}. "
                             f"Available: {sorted(OPPONENT_STRATEGIES)}")
        if SELF_PLAY in opponent_pool:
            raise ValueError("distillation has no 'self' opponent - who DRIVES is controlled by "
                             "dagger_warmup (teacher first, student later); this pool picks who "
                             "the agent PLAYS AGAINST.")
        print(f"Opponent pool (states the student is taught on): {opponent_pool}")

    force_disrupt = bool(blob.get("force_disrupt", False))
    print(f"Teacher force_disrupt={force_disrupt}"
          + ("" if "force_disrupt" in blob else "  (not recorded in the checkpoint; assuming False)"))

    student_kwargs = dict(
        in_channels=teacher_kwargs["in_channels"],
        policy_channels=teacher_kwargs["policy_channels"],
        channels=tuple(channels),
        value_hidden=value_hidden,
    )
    student = RailroadUNet(**student_kwargs).to(device)

    # Quantization-aware training. The student is graded at full precision but SHIPS at int4, and
    # that round-trip is the biggest remaining loss in the pipeline (0.570 -> 0.500 over 250
    # head-to-head games). Wrapping the conv weights makes every forward pass use rounded weights
    # while gradients still update the full-precision ones, so the student learns a configuration
    # that survives rounding instead of one that merely tolerates it.
    if quant_aware:
        qmax = {"int4": INT4_QMAX, "int8": INT8_QMAX}.get(quant_aware)
        if qmax is None:
            raise ValueError(f"quant_aware must be 'int4', 'int8' or null, got {quant_aware!r}")
        n_wrapped = attach_fake_quant(student, qmax)
        print(f"Quantization-aware training: {quant_aware} (qmax {qmax}) on {n_wrapped} conv "
              f"weights. Rollout and eval use the ROUNDED weights; checkpoints store the "
              f"full-precision ones, which bake_agent.py then quantizes identically.")

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    s_policy = s_params - sum(p.numel() for n, p in student.named_parameters()
                              if n.startswith("value_head"))
    print(f"teacher {teacher_checkpoint}  {teacher_kwargs}  {t_params:,} params")
    print(f"student channels={tuple(channels)}  {s_params:,} params "
          f"({s_policy:,} policy-only, which is what gets baked)")

    if value_loss_weight:
        print(f"Value distillation: {'Huber(beta=%s)' % value_huber if value_huber else 'MSE'} "
              f"against the teacher's value, weight {value_loss_weight}. "
              f"Without it the student's value head gets no gradient at all and ships with its "
              f"random initialisation. bake_agent.py drops the critic, so this costs 0 characters "
              f"in submission.py.")
    else:
        print("Value distillation OFF (value_loss_weight=0): the student's value head will keep "
              "its random initialisation. distill/value_mse is still logged, unweighted.")

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    bosses = [b.strip() for b in eval_bosses.split(",") if b.strip()]
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers) if num_workers > 1 else None
    try:
        for it in range(num_iterations):
            frac = it / max(num_iterations - 1, 1)
            for g in opt.param_groups:
                g["lr"] = lr + (lr_end - lr) * frac
            beta = min(1.0, it / max(dagger_warmup, 1))

            t0 = time.time()
            # collect() copies both state dicts to CPU itself - never move the live module, or
            # Adam's state is left on the previous device and the next step throws.
            states, tlogits, tvalues, masks, index = collect(
                pool, teacher, teacher_kwargs, student, student_kwargs,
                rng.randint(0, 2**31 - 1, size=games_per_iteration),
                max_turns, opponent, beta, epsilon, rng, force_disrupt=force_disrupt,
                opponent_pool=opponent_pool)
            collect_time = time.time() - t0

            S = torch.from_numpy(states).to(device=device, dtype=torch.float32)
            TL = torch.from_numpy(tlogits).to(device=device, dtype=torch.float32)
            TV = torch.from_numpy(tvalues).to(device=device, dtype=torch.float32)
            MK = torch.from_numpy(masks).to(device)
            IX = torch.from_numpy(index).to(device)

            t0 = time.time()
            losses, vlosses, agrees = [], [], []
            n_dec = len(IX)
            for _ in range(epochs_per_iteration):
                perm = torch.randperm(n_dec, device=device)
                for start in range(0, n_dec, minibatch_size):
                    sel = perm[start:start + minibatch_size]
                    six = IX[sel]
                    uniq, inv = torch.unique(six, return_inverse=True)
                    sl, sv = student(S[uniq])
                    sl = sl[inv]
                    tl = TL[six]
                    mk = MK[sel]

                    log_p_s = _masked_log_softmax(sl, mk)
                    log_p_t = _masked_log_softmax(tl, mk)
                    p_t = log_p_t.exp()
                    loss = (p_t * (log_p_t - log_p_s)).sum(dim=1).mean()

                    # Regress the teacher's value as well, once per UNIQUE state rather than once
                    # per decision - the value does not change between a turn's sub-decisions, so
                    # weighting it by decision count would just over-sample early-turn states.
                    # Without this term the student's value head never receives a gradient and
                    # ships with its random initialisation, which is why P(win) read as a constant.
                    #
                    # Huber, not plain MSE, because the target's SCALE swings between iterations:
                    # the opponent pool is sampled per game, and teacher-value variance measured
                    # 3.6 / 6.3 / 23.6 over three consecutive iterations. Under MSE that hands the
                    # shared trunk a ~6x larger gradient on the unlucky iteration, and the policy
                    # KL spiked with it. Huber bounds each state's gradient at `value_huber`.
                    pv, tv_ = sv.squeeze(-1), TV[uniq]
                    vloss = (F.smooth_l1_loss(pv, tv_, beta=value_huber) if value_huber
                             else F.mse_loss(pv, tv_))
                    # Logged as true MSE regardless, so the number stays comparable across runs
                    # and against a value_loss_weight=0 baseline.
                    vlosses.append(float(F.mse_loss(pv.detach(), tv_)))
                    if value_loss_weight:
                        loss = loss + value_loss_weight * vloss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                    opt.step()
                    losses.append(loss.item())
                    with torch.no_grad():
                        agrees.append((log_p_s.argmax(1) == log_p_t.argmax(1)).float().mean().item())
            train_time = time.time() - t0

            kl = float(np.mean(losses))
            agree = float(np.mean(agrees))
            vmse = float(np.mean(vlosses))
            # Correlation over the whole iteration, not per minibatch: a minibatch holds ~15
            # unique states, far too few to read a correlation off. This is the number that says
            # whether the critic is real - it sat near 0 for a student distilled on logits alone.
            was_training = student.training
            student.eval()
            with torch.no_grad():
                pred = torch.cat([student(S[i:i + 256])[1].squeeze(-1)
                                  for i in range(0, len(S), 256)])
            student.train(was_training)
            vcorr = float(np.corrcoef(pred.cpu().numpy(), tvalues)[0, 1]) if len(S) > 2 else 0.0
            writer.add_scalar("distill/kl", kl, it)
            writer.add_scalar("distill/value_mse", vmse, it)
            writer.add_scalar("distill/value_corr", vcorr, it)
            writer.add_scalar("distill/teacher_value_var", float(np.var(tvalues)), it)
            writer.add_scalar("distill/argmax_agreement", agree, it)
            writer.add_scalar("distill/dagger_beta", beta, it)
            writer.add_scalar("distill/decisions", n_dec, it)
            print(f"[iter {it}] kl={kl:.4f} vmse={vmse:.4f} vcorr={vcorr:+.3f} "
                  f"agree={agree:.3f} beta={beta:.2f} "
                  f"states={len(S)} decisions={n_dec} "
                  f"collect={collect_time:.1f}s train={train_time:.1f}s")

            if (it + 1) % checkpoint_every == 0:
                path = ckpt_dir / f"{run_name}_iter{it + 1}.pt"
                torch.save({"iteration": it + 1,
                            "model_state_dict": plain_state_dict(student, quantized=False),
                            "model_kwargs": student_kwargs,
                            # Inherited from the teacher so bake_agent.py and evaluate.py
                            # reproduce the action space the student was actually taught.
                            "allow_skip_disrupt": student_kwargs["policy_channels"] >= 3,
                            "force_disrupt": force_disrupt}, path)
                print(f"[iter {it}] checkpoint saved: {path}")

            if eval_episodes > 0 and bosses and (it + 1) % eval_every == 0:
                t0 = time.time()
                # Grade the weights that SHIP. Under quant_aware the student's own state_dict
                # carries parametrization keys a plain RailroadUNet cannot load, and its
                # full-precision weights are not what gets submitted either - so hand the
                # evaluator a plain model holding the rounded weights.
                graded = student
                if quant_aware:
                    graded = RailroadUNet(**student_kwargs).to(device)
                    graded.load_state_dict(plain_state_dict(student, quantized=True))
                    graded.eval()
                res = evaluate_all_bosses(
                    graded, student_kwargs, eval_episodes, max_turns,
                    student_kwargs["policy_channels"] >= 3, bosses, pool=pool,
                    force_disrupt=force_disrupt)
                print(format_eval_table(
                    res, title=f"\n  student @ iter {it + 1} - {eval_episodes} greedy episodes "
                               f"vs each boss ({time.time() - t0:.0f}s)"))
                log_eval_to_tensorboard(writer, res, it)
                print()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        writer.close()
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--teacher", type=str, default=None)
    args, extra = ap.parse_known_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.teacher:
        cfg["teacher_checkpoint"] = args.teacher
    if "channels" in cfg and cfg["channels"] is not None:
        cfg["channels"] = tuple(cfg["channels"])
    distill(**cfg)


if __name__ == "__main__":
    main()

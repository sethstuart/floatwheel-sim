"""Train PPO to balance the Floatwheel, judged against the Refloat PID.

Usage:
    python -m tools.train_ppo [--timesteps 300000] [--scene balance|ride]
                              [--act-on torque|setpoint] [--envs 4] [--warmup 0.25]
                              [--push-mean-s 4] [--out runs/ppo_balance] [--seed 0]

Judged, not merely trained: the learned policy and the firmware PID are run on the same
eval episodes and reported side by side (return, fall rate, max |pitch|, mean |current|).
A policy that "balances" by freezing in the middle of the tolerance band is worthless,
so mean |current| is the number that says whether it is doing anything at all.

Curriculum: the first `--warmup` fraction trains with no shoves at all (learn to stand
up), then pushes are switched on for the rest. Starting under disturbance wastes most
of the samples on falling over.

evaluate_policy() is importable: tools/tune_sweep.py uses it for the gain sweeps so the
sweep and the training report measure the same thing the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from sim import spec
from sim.env import FloatwheelEnv


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# NEXT 367: the training fleet, decided once, applied to every env this tool builds. `--protections`
# and `--arm k=v` set them in main(). Module-level rather than threaded through seven signatures,
# because the failure mode of threading them is a run that TRAINS with the fleet and EVALUATES without
# it — the same asymmetry that made the hand PD and the policy non-comparable to begin with. Warm-start
# data is collected on the armed board for the same reason. A number that does not say what the board
# was armed with is not a number, so report.json records both.
PROTECTIONS = False
ARM: dict[str, float] = {}
# NEXT 347: the clone's fit on states it was never fitted on, kept for report.json and for the gate in
# main(). NaN means "never measured", which the gate treats as a failure rather than a pass.
CLONE_CORR = float("nan")


def _env_params(overrides: dict | None) -> dict:
    """Firmware params for an env: what the caller asked for, plus the behaviours we armed."""
    return {**ARM, **(overrides or {})}


def _set_fleet(protections: bool, arms: list[str], act_on: str = "torque") -> None:
    """Apply the CLI's choice of board, once, before any env is built, and say it out loud.

    Two traps get shouted about here rather than left to the reader, and both are the same species as
    NEXT 362's compiled-out channel and NEXT 365's inert control row:

    * `--protections` with no `--arm`: `firmware_params.py` ships every tiltback strength at 0.0, so the
      fleet gets stepped and still cannot move the board — a run that reads like it tested the safety
      behaviours and tested none of them.
    * `--protections` with `--act-on torque` (NEXT 367, measured 2026-11-16): in torque mode the policy
      IS the balance controller, `controller.update()` never runs, and the whole fleet lives in there.
      Training a pure-torque policy "with protections" is therefore inert — the 20k-step comparison on
      this box came back byte-identical to the fleet-off run in BOTH scenes (return 1583.1, max|pitch|
      1.33, mean|A| 2.77 either way), while the setpoint PID baseline did move (mean|A| 1.9 -> 22.5).
      Not refused, because the setpoint baseline in the same report still wants the fleet; but it is
      printed and recorded in report.json, so nobody quotes a torque-policy number as if the board had
      been fighting back.
    """
    global PROTECTIONS, ARM
    ARM = {}
    for kv in arms:
        k, sep, v = kv.partition("=")
        if not sep or not k.strip():
            raise SystemExit("--arm wants KEY=VALUE, got %r" % kv)
        try:
            ARM[k.strip()] = float(v)
        except ValueError:
            raise SystemExit("--arm %s wants a number, got %r" % (k.strip(), v))
    PROTECTIONS = bool(protections)
    INERT_IN_TORQUE = PROTECTIONS and act_on == "torque"
    if PROTECTIONS and not ARM:
        print("NOTE: --protections with no --arm arms NOTHING: every tiltback strength ships at 0.0, "
              "so the fleet will be stepped and can still leave the board untouched. Name the "
              "behaviours you mean, e.g. --arm tiltback_speed=2.0 --arm tiltback_speed_angle=15")
    if INERT_IN_TORQUE:
        print("WARNING: --protections is INERT with --act-on torque. The policy is the controller there, "
              "controller.update() never runs, and the fleet lives inside it — so the trained policy "
              "never meets a tiltback. Expect its numbers to match the fleet-off run exactly. Use "
              "--act-on setpoint to train against the board that fights back.")
    print("fleet: protections=%s arm=%s%s"
          % (PROTECTIONS, ARM or "none", " (inert in torque mode)" if INERT_IN_TORQUE else ""))


def _env_factory(scene, act_on, pushes, overrides, seed_base, torque_scale=1.0, torque_map="quad",
                 protections=None):
    def make():
        return FloatwheelEnv(
            scene=scene,
            act_on=act_on,
            push_mean_s=pushes,
            param_overrides=_env_params(overrides),
            torque_scale=torque_scale,
            torque_map=torque_map,
            protections=PROTECTIONS if protections is None else protections,
            seed=seed_base,
        )

    return make


def evaluate_policy(
    model,
    norm=None,
    scene: str = "balance",
    act_on: str = "torque",
    episodes: int = 12,
    seed: int = 999,
    pushes: float | None = 4.0,
    overrides: dict | None = None,
    deterministic: bool = True,
    torque_scale: float = 1.0,
    torque_map: str = "quad",
    noise_deg: float = 0.0,
    protections: bool | None = None,
) -> dict:
    """Run `episodes` and return the numbers a tune is judged by.

    model=None means "no policy": a zero action, which with act_on="setpoint" is exactly
    the firmware PID holding a level setpoint - the baseline the policy has to beat.

    torque_scale must match the one used in training, or the policy is judged on an
    actuator with different limits than the one it learned to drive.
    """
    env = FloatwheelEnv(
        scene=scene,
        act_on=act_on,
        push_mean_s=pushes,
        param_overrides=_env_params(overrides),
        torque_scale=torque_scale,
        torque_map=torque_map,
        protections=PROTECTIONS if protections is None else protections,
        seed=seed,
        noise_deg=noise_deg,
    )
    if norm is not None:
        norm.training = False
        norm.reward = False

    rows = []
    for _ in range(episodes):
        obs, info = env.reset()
        o = norm.normalize_obs(obs) if norm is not None else obs
        total = 0.0
        steps = 0
        max_pitch = 0.0
        cur_sum = 0.0
        fell = False
        while True:
            if model is None:
                action = np.zeros(1)
            else:
                action, _ = model.predict(o.reshape(1, -1), deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            o = norm.normalize_obs(obs) if norm is not None else obs
            total += r
            steps += 1
            max_pitch = max(max_pitch, abs(info["pitch"]))
            cur_sum += abs(info["current"])
            fell = bool(term)
            if term or trunc:
                break
        rows.append(
            {
                "return": total,
                "steps": steps,
                "seconds": info["time"],
                "fell": fell,
                "max_pitch_deg": max_pitch,
                "mean_abs_current_A": cur_sum / max(1, steps),
                "final_speed": info["speed"],
            }
        )

    def stat(key):
        a = np.array([r[key] for r in rows], dtype=float)
        return float(a.mean()), float(a.std())

    ret, retsd = stat("return")
    mp, _ = stat("max_pitch_deg")
    ac, _ = stat("mean_abs_current_A")
    st, _ = stat("steps")
    fall = float(np.mean([r["fell"] for r in rows]))
    return {
        "return": round(ret, 1),
        "return_sd": round(retsd, 1),
        "fall_rate": round(fall, 3),
        "steps": round(st, 0),
        "max_pitch_deg": round(mp, 2),
        "mean_abs_current_A": round(ac, 2),
        "episodes": episodes,
        "rows": rows,
    }


def hand_action(obs, kp: float = -1.4, kd: float = -0.02) -> float:
    """The two-line balance law from tools/random_rollout.py, in action units.

    Both gains negative: torque has to oppose BOTH lean and lean-rate, and through this
    model's sign convention that means the same sign for both. Opposite signs - the
    intuitive mistake - makes the "damping" fight the spring and the board falls faster
    the harder you damp (measured in tools/probe_hand.sh).
    """
    pitch = float(np.asarray(obs).reshape(-1)[0]) * 10.0     # OBS_SCALE: /10
    rate = float(np.asarray(obs).reshape(-1)[2]) * 100.0     # and /100
    return float(np.clip(kp * pitch + kd * rate, -1.0, 1.0))


def collect_hand_data(scene, act_on, torque_map, n_episodes=40, seed=1234):
    """States and actions visited while the hand controller rides, for warm starting."""
    # The warm-start states have to come from the board the policy will actually train on. Collecting
    # them with the fleet compiled out and then training it armed would teach it a board that does not
    # exist, and the clone's held-out score would measure the mismatch rather than the fit.
    env = FloatwheelEnv(
        scene=scene, act_on=act_on, push_mean_s=4.0, torque_map=torque_map,
        param_overrides=_env_params(None), protections=PROTECTIONS, seed=seed
    )
    X, Y = [], []
    for _ in range(n_episodes):
        o, _ = env.reset()
        while True:
            a = hand_action(o)
            o2, _r, term, trunc, _i = env.step(np.array([a]))
            X.append(np.asarray(o, dtype=np.float32))
            Y.append([a])
            o = o2
            if term or trunc:
                break
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def behaviour_clone(model, X, Y, epochs=60, batch=128, lr=1e-3, log_std=-1.5, holdout=None) -> float:
    """Supervised pre-fit of the policy mean onto the hand controller, then PPO refines.

    On an unstable plant PPO from a zero-mean initial policy never reaches the regime
    where the correct sign pays: every episode is over in ~0.9 s, long before a catch
    could ever be credited, and 400k steps of that produced a policy that fell in 100%
    of episodes while the hand PD held 100%. Warm starting is the standard answer, and
    it is honest here because the thing being cloned is a controller we wrote, not a
    teacher model. Returns the final BC loss.
    """
    import torch

    Xt = torch.as_tensor(X, dtype=torch.float32).to(model.device)
    Yt = torch.as_tensor(Y, dtype=torch.float32).to(model.device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    model.policy.train()
    final = float("nan")
    for ep in range(epochs):
        idx = torch.randperm(len(Xt), device=model.device)
        tot = 0.0
        for i in range(0, len(idx), batch):
            b = idx[i : i + batch]
            out = model.policy(Xt[b])
            # NEXT 347 (measured on SB3 2.9.2): policy(obs) returns a 3-tuple
            # (action, latent, value). The old code took out[1] -- the LATENT, which here
            # is shape (B,1) and near-constant, so its MSE drove itself to 0.0000 while the
            # emitted action never moved: the clone printed a perfect fit and left the
            # policy riding at 0.24 A, i.e. no torque at all. out[0] is the mean the env
            # gets. If a future SB3 reshuffles the tuple, the shape guard below fails loud
            # instead of silently fitting furniture.
            pi = out[0] if isinstance(out, tuple) else out
            if tuple(pi.shape) != tuple(Yt[b].shape):
                raise SystemExit(
                    "behaviour_clone: policy output %s is not the action %s -- check what "
                    "policy(obs) returns on this stable-baselines3 before trusting any loss"
                    % (tuple(pi.shape), tuple(Yt[b].shape))
                )
            loss = torch.nn.functional.mse_loss(pi, Yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            opt.step()
            tot += float(loss.detach()) * len(b)
        final = tot / max(1, len(Xt))
        if ep % 20 == 19:
            print("  behaviour clone epoch %2d  loss %.4f" % (ep + 1, final))
    # The number a BC loss cannot fake: does the EMITTED mean track the hand law?
    # (The loss lied once -- NEXT 347 -- so the fit is measured separately from it.)
    # Judged on the HOLDOUT when the caller gave us one. Scoring the states we just fitted on measures
    # memorisation, and PPO is about to drive into states the hand rollout never visited -- which is
    # exactly why the row asked for 200 held-out states rather than "any 200 states".
    global CLONE_CORR
    if holdout is not None:
        Xj, Yj = holdout
        judged = "held-out"
    else:
        Xj, Yj = X[: min(4096, len(X))], Y[: min(4096, len(Y))]
        judged = "TRAINING"
        print("  NOTE: no holdout passed - this fit is scored on the states it was fitted on, so a "
              "memorising clone would score perfectly here and still not reproduce hand_action(obs)")
    CLONE_CORR = float("nan")
    with torch.no_grad():
        out = model.policy(torch.as_tensor(Xj, dtype=torch.float32).to(model.device))
        fit = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()
    if fit.ndim == 2 and fit.shape[1] == 1:
        yc = np.asarray(Yj)[: fit.shape[0], 0]
        CLONE_CORR = float(np.corrcoef(fit[:, 0], yc)[0, 1])
        print("  behaviour clone corr(emitted mean, hand_action) = %.4f on %d %s states"
              % (CLONE_CORR, fit.shape[0], judged))
        if CLONE_CORR < 0.9:
            print("  WARNING: corr < 0.9 -- the clone did not actually learn the hand law; "
                  "PPO will start from noise on an unstable plant")
    else:
        print("  WARNING: could not measure the clone's fit (policy output %s) -- treating that as "
              "unproven, not as a pass" % (tuple(fit.shape),))
    # The clone now knows a good mean; start exploration tight around it (std ~0.22)
    # instead of the bang-bang noise that made the first runs unusable.
    with torch.no_grad():
        model.policy.log_std.fill_(log_std)  # SB3 2.x name; `logstd` is 1.x
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--scene", default="balance", choices=["balance", "ride"])
    ap.add_argument("--act-on", default="torque", choices=["torque", "setpoint"])
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.25)
    ap.add_argument("--push-mean-s", type=float, default=4.0)
    ap.add_argument("--torque-scale", type=float, default=1.0)
    ap.add_argument("--torque-map", default="quad", choices=["quad", "linear"])
    ap.add_argument("--warmstart", default="auto", choices=["auto", "hand", "none"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=12)
    # NEXT 367: which board this run gets. Off by default, because that is what every run in this lane
    # has measured — the fleet is not unarmed by default, it is never stepped at all.
    ap.add_argument("--protections", action="store_true",
                    help="step sim/protections.py in training, evaluation AND warm-start collection")
    ap.add_argument("--arm", action="append", default=[], metavar="K=V",
                    help="arm one behaviour, repeatable: --arm tiltback_speed=2.0 "
                         "--arm tiltback_speed_angle=15 (strengths ship at 0.0)")
    # NEXT 347: the clone gate refuses to train on an unproven warm start. Overridable, because a run you
    # knowingly accept should still print the number that made it refuse -- the override is the record.
    ap.add_argument("--allow-bad-clone", action="store_true",
                    help="train anyway even if the behaviour clone fails the held-out fit gate")
    args = ap.parse_args()
    _set_fleet(args.protections, args.arm, args.act_on)

    tag = "%s_%s" % (args.scene, args.act_on)
    out = args.out or os.path.join("runs", "ppo_" + tag)
    os.makedirs(out, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # NEXT 348, measured on this plant (30k-step shootout, 4 envs): CPU 1499 steps/s vs
    # CUDA 525 -- a 64x64 MLP is too small to pay for the GPU round-trips, exactly what
    # SB3's own warning says. auto therefore means CPU here; --device cuda still works.
    dev = args.device if args.device != "auto" else ("cpu" if _device() == "cuda" else _device())
    print("train %s: %d steps on %s envs, device=%s" % (tag, args.timesteps, args.envs, dev))

    vec = make_vec_env(
        _env_factory(
            args.scene, args.act_on, None, None, args.seed, args.torque_scale, args.torque_map
        ),
        n_envs=args.envs,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
    )
    # norm_obs=False on purpose: FloatwheelEnv already divides every field by OBS_SCALE
    # (that is what the scale list is for), and double-normalising broke the behaviour
    # clone - BC fit the hand law to loss 0.0000 on raw observations, then the policy
    # was handed rescaled ones and rode at 0.28 A, which is to say: no torque at all.
    # Reward normalisation is still worth having.
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec,
        # log_std_init: SB3 starts at std 1.0, which on a 28 N*m hub means the first
        # policy is a bang-bang thrash that falls in every episode (measured: 20k steps
        # gave fall_rate 1.00 at std 1.01). Start curious-but-not-erratic.
        policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.0),
        learning_rate=3e-4,
        # gamma 0.998, not 0.99: at a 100 Hz control rate 0.99 is a ~1 s horizon, which
        # cannot credit the recovery that happens 0.5-1.5 s after a shove. The first
        # 400k-step run with 0.99 fell in 92% of episodes; the horizon, not the network,
        # was the binding constraint. 4096 steps/rollout so the long horizon has data.
        n_steps=4096,
        batch_size=256,
        gamma=0.998,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        max_grad_norm=0.5,
        device=dev,
        verbose=1,
        seed=args.seed,
    )

    ws = args.warmstart
    if ws == "auto":
        # Only the pure-torque policy needs it: with the PID under it the board is
        # already stable at t=0, so exploration starts in a survivable regime.
        ws = "hand" if args.act_on == "torque" else "none"
    if ws == "hand":
        X, Y = collect_hand_data(args.scene, args.act_on, args.torque_map)
        # A different seed, so the judge sees states the fitted rollout never visited. Three episodes
        # is ~3,000 rows, comfortably the "200 held-out states" the row asks for.
        Xh, Yh = collect_hand_data(args.scene, args.act_on, args.torque_map,
                                   n_episodes=3, seed=args.seed + 987654)
        print("warm start: behaviour clone on %d samples from the hand PD, judged on %d HELD-OUT states "
              "(seed %d, not the fit seed %d)" % (len(X), len(Xh), args.seed + 987654, 1234))
        behaviour_clone(model, X, Y, holdout=(Xh, Yh))
        # NEXT 347's Measure as a gate and not a footnote: prove the clone against hand_action() on 200
        # held-out states BEFORE PPO is allowed to run. A warning that lets training continue is how a
        # clone that never learned the law still gets 300k steps and comes back with a healthy-looking
        # report -- which is the exact shape of the 0.24 A bug this row was filed for.
        if not args.allow_bad_clone and (CLONE_CORR != CLONE_CORR or CLONE_CORR < 0.9):
            raise SystemExit(
                "clone is unproven: corr against hand_action() on %d held-out states is %s (need >= 0.9). "
                "The warm start did not learn the law, so PPO would be exploring an unstable plant from "
                "noise; fix the clone, or pass --allow-bad-clone to run it anyway with the number in "
                "report.json." % (len(Xh), "NaN (never measured)" if CLONE_CORR != CLONE_CORR
                                  else "%.4f" % CLONE_CORR)
            )

    t0 = time.time()
    warm = int(args.timesteps * args.warmup)
    print("--- phase 1: no shoves, %d steps" % warm)
    model.learn(total_timesteps=warm)
    print("--- phase 2: shoves every ~%.1f s, %d steps" % (args.push_mean_s, args.timesteps - warm))
    vec.env_method("set_push_mean", args.push_mean_s)
    model.learn(total_timesteps=args.timesteps - warm, reset_num_timesteps=False)
    train_s = time.time() - t0

    model.save(os.path.join(out, "model.zip"))
    vec.save(os.path.join(out, "vecnormalize.pkl"))
    vec.training = False
    vec.reward = False

    learned = evaluate_policy(
        model,
        vec,
        scene=args.scene,
        act_on=args.act_on,
        episodes=args.eval_episodes,
        pushes=args.push_mean_s,
        torque_scale=args.torque_scale,
        torque_map=args.torque_map,
    )
    baseline = evaluate_policy(
        None,
        None,
        scene=args.scene,
        act_on="setpoint",
        episodes=args.eval_episodes,
        pushes=args.push_mean_s,
    )
    calm = evaluate_policy(
        model,
        vec,
        scene=args.scene,
        act_on=args.act_on,
        episodes=args.eval_episodes,
        pushes=None,
        torque_scale=args.torque_scale,
        torque_map=args.torque_map,
    )

    print("%-22s %9s %8s %9s %10s %10s" % ("", "return", "fall%", "max|pitch|", "mean|A|", "steps"))
    for label, res in [("learned (shoves)", learned), ("learned (no shoves)", calm), ("firmware PID", baseline)]:
        print(
            "%-22s %9.1f %8.0f %9.2f %10.2f %10.0f"
            % (
                label,
                res["return"],
                100 * res["fall_rate"],
                res["max_pitch_deg"],
                res["mean_abs_current_A"],
                res["steps"],
            )
        )
    print("train wall %.0f s for %d steps (%.0f steps/s)" % (train_s, args.timesteps, args.timesteps / max(1.0, train_s)))

    report = {
        "tag": tag,
        "timesteps": args.timesteps,
        "envs": args.envs,
        "device": dev,
        # NEXT 367: which board this number was measured on. A report that does not say whether the
        # fleet was stepped is exactly how a protections-free number ends up quoted as an armed one.
        "protections": PROTECTIONS,
        "armed": dict(ARM),
        # NEXT 367: a torque run's policy never meets the fleet at all — the fleet lives inside
        # controller.update(), which torque mode does not call. Recorded here, not only warned on stdout,
        # because report.json is the artifact that gets quoted a month from now and the warning won't be.
        "protections_inert_in_torque": bool(PROTECTIONS and tag.endswith("_torque")),
        # NEXT 347: the clone's proof, in the artifact rather than only on stdout. NaN when the warm start
        # was skipped or the fit could not be measured -- a missing proof and a passed proof must not
        # serialise the same way.
        "clone_corr_holdout": None if CLONE_CORR != CLONE_CORR else round(CLONE_CORR, 4),
        "train_seconds": round(train_s, 1),
        "steps_per_s": round(args.timesteps / max(1.0, train_s), 0),
        "learned": {k: v for k, v in learned.items() if k != "rows"},
        "learned_calm": {k: v for k, v in calm.items() if k != "rows"},
        "pid_baseline": {k: v for k, v in baseline.items() if k != "rows"},
        "spec_estimate": {k: spec.g(k) for k in ("TORQUE_MAX", "COM_HEIGHT", "RIDER_MASS", "WHEEL_RADIUS")},
    }
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print("wrote %s" % os.path.join(out, "report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Random-policy rollouts: proves the skeleton closes end to end.

Usage:
    python -m tools.random_rollout [--episodes N] [--scene balance|ride]
                                   [--act-on torque|setpoint] [--seed S]

A random policy is the falsification test: it must fall over fast. If a random policy
balances, the reward or the plant is wrong and nothing trained on it means anything.
The second half of the run is the control - the Refloat PID holding a zero setpoint -
which must survive; if *it* falls too, the two results say the sim is broken, not the
policy.

Traces land in out/rollout_*.npz for later plots and renders.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from sim import spec
from sim.env import FloatwheelEnv


def run_episode(env, policy) -> dict:
    obs, info = env.reset()
    total = 0.0
    steps = 0
    max_pitch = 0.0
    trace = []
    while True:
        action = policy(obs, steps)
        obs, r, term, trunc, info = env.step(action)
        total += r
        steps += 1
        max_pitch = max(max_pitch, abs(info["pitch"]))
        trace.append((obs.copy(), r))
        if term or trunc:
            return {
                "return": total,
                "steps": steps,
                "fell": bool(term),
                "seconds": info["time"],
                "max_pitch_deg": max_pitch,
                "trace": trace,
            }


def sweep(label: str, env, policy, episodes: int) -> dict:
    rows = [run_episode(env, policy) for _ in range(episodes)]
    rets = np.array([r["return"] for r in rows])
    lens = np.array([r["steps"] for r in rows])
    fell = np.array([r["fell"] for r in rows])
    print(
        "%-22s return %7.1f +- %5.1f   steps %6.0f +- %4.0f   fell %3.0f%%   max|pitch| %5.1f deg"
        % (
            label,
            rets.mean(),
            rets.std(),
            lens.mean(),
            lens.std(),
            100.0 * fell.mean(),
            np.mean([r["max_pitch_deg"] for r in rows]),
        )
    )
    out = os.path.join("out", "rollout_%s.npz" % label.replace(" ", "_"))
    os.makedirs("out", exist_ok=True)
    np.savez_compressed(
        out,
        returns=rets,
        steps=lens,
        fell=fell,
        obs=np.vstack([np.vstack([t[0] for t in r["trace"]]) for r in rows[:2]]),
    )
    return {"rows": rows, "path": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--scene", default="balance", choices=["balance", "ride"])
    ap.add_argument("--act-on", default="torque", choices=["torque", "setpoint"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hand-kp", type=float, default=0.7)
    ap.add_argument("--hand-kd", type=float, default=0.007)
    ap.add_argument("--torque-map", default="quad", choices=["quad", "linear"])
    args = ap.parse_args()

    t0 = time.time()
    done = 0
    print(
        "floatwheel rollout: scene=%s act_on=%s episodes=%d control_hz=%s step_s=%s"
        % (args.scene, args.act_on, args.episodes, spec.g("CONTROL_HZ"), spec.g("STEP_S"))
    )

    # 1) random policy: must fall
    env = FloatwheelEnv(scene=args.scene, act_on=args.act_on, seed=args.seed)
    rs = np.random.default_rng(args.seed)
    sweep("random_%s" % args.act_on, env, lambda o, s: rs.uniform(-1, 1, 1), args.episodes)
    done += args.episodes

    # 2) control: the firmware PID holding a zero setpoint must survive
    env2 = FloatwheelEnv(
        scene=args.scene, act_on="setpoint", push_mean_s=4.0, seed=args.seed + 1
    )
    sweep("pid_hold_setpoint0", env2, lambda o, s: np.zeros(1), args.episodes)
    done += args.episodes

    # 3) control with no pushes: the bare balance loop, upper bound on return
    env3 = FloatwheelEnv(
        scene=args.scene, act_on="setpoint", push_mean_s=None, seed=args.seed + 2
    )
    sweep("pid_no_pushes", env3, lambda o, s: np.zeros(1), args.episodes)
    done += args.episodes

    # 4) ride: hold a zero setpoint while chasing whatever speed the scene asked for;
    #    the speed-tracking policy itself is what SB3 will learn in step 2 of the goal.
    env4 = FloatwheelEnv(scene="ride", act_on="setpoint", push_mean_s=4.0, seed=args.seed + 3)
    sweep("ride_zero_setpoint", env4, lambda o, s: np.zeros(1), args.episodes)
    done += args.episodes

    # 5) hand torque: a two-line P + rate-D controller driving the SAME action mapping
    #    the RL policy uses. This is the discriminating test for act_on="torque" - if a
    #    hand controller written from the physics cannot hold it up, the env's torque
    #    path is wrong and no amount of training will fix it. If it can, and PPO cannot,
    #    that is a training problem and gets reported as one.
    if args.act_on == "torque":
        env5 = FloatwheelEnv(
            scene=args.scene,
            act_on="torque",
            push_mean_s=4.0,
            torque_map=args.torque_map,
            seed=args.seed + 4,
        )

        def hand(o, s, kp=args.hand_kp, kd=args.hand_kd):
            pitch = float(o[0]) * 10.0        # OBS_SCALE divides by 10
            rate = float(o[2]) * 100.0        # and the rate by 100
            return np.array([max(-1.0, min(1.0, kp * pitch + kd * rate))])

        sweep("hand_torque", env5, hand, args.episodes)
        done += args.episodes

    print("wall %.1f s for %d episodes" % (time.time() - t0, done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

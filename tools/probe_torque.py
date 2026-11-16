"""Does an action in act_on="torque" reach the physics at all?

Usage:
    python -m tools.probe_torque

Written because six wildly different hand-controller gains (kp from -2.8 to +2.8, signs
included) produced byte-identical rollouts - 89 steps, 65.1 return, every one. That
signature means the action is not reaching the simulator, so this steps the env by hand
with a fixed action and prints what actually moved: ctrl, the applied actuator force,
wheel speed and pitch. If those columns do not change with the action, the bug is in the
env or the model, not in the training run.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from sim import model as M
from sim.env import FloatwheelEnv


def main() -> int:
    probe = FloatwheelEnv(scene="balance", act_on="torque", push_mean_s=None, seed=0)
    m = probe.model
    print("model topology (nv=%d, nu=%d):" % (m.nv, m.nu))
    for j in range(1, m.njnt):
        print(
            "    joint %-8s type=%d qposadr=%d dofadr=%s"
            % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), m.jnt_type[j],
               m.jnt_qposadr[j], m.jnt_dofadr[j])
        )
    for i in range(m.nu):
        print(
            "    actuator %-8s trnid=%s trntype=%s gear=%s"
            % (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                list(m.actuator_trnid[i]),
                int(m.actuator_trntype[i]),
                m.actuator_gear[i][0],
            )
        )
    print("    _root_dof=%s  wheel_speed accessor=%s" % (M._root_dof(m), M.wheel_speed(m, probe.data)))
    for action in (0.0, 1.0, -1.0):
        env = FloatwheelEnv(scene="balance", act_on="torque", push_mean_s=None, seed=0)
        env.reset()
        print("--- action %+.1f" % action)
        survived = 0
        for _ in range(400):
            _o, _r, term, trunc, _i = env.step(np.array([action]))
            survived += 1
            if term or trunc:
                break  # keep stepping past termination and you print a wreck, not a result
        d = env.data
        # Not qvel[1]: the base is a free joint, so qvel[0:6] is the deck and the wheel
        # dof is the last one. Ask the model, do not guess - guessing here printed
        # "wheel = -0.00 rad/s" for a wheel that was in fact spinning hard.
        wdof = int(env.model.actuator_trnid[0][0])
        print(
            "    engaged=%s  torque=%+.2f N*m  current=%+.2f A  data.ctrl=%+.3f"
            % (
                env.controller.engaged,
                env.controller.torque,
                env.controller.current,
                d.ctrl[0],
            )
        )
        print(
            "    wheel_dof=%d  wheel=%+.2f rad/s  speed=%+.2f m/s  pitch=%+.2f deg"
            "  qfrc_actuator=%+.2f  time=%.2f s"
            % (
                wdof,
                d.qvel[wdof],
                M.forward_speed(env.model, d),
                math.degrees(M.pitch_of(env.model, d)),
                d.qfrc_actuator[wdof],
                d.time,
            )
        )
        print("    actuator gear=%s  ctrlrange=%s" % (
            env.model.actuator_gear[0][0],
            list(env.model.actuator_ctrlrange[0]),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

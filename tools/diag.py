"""Print what a model variant actually does when torque is applied, to separate
"the sim is wrong" from "my test setup is wrong".

Usage:
    python -m tools.diag                 # normal model
    python -m tools.diag --fixture       # clamped (flywheel-on-a-stand) model
    python -m tools.diag --torque 12 --ms 200
"""

from __future__ import annotations

import argparse
import math

import mujoco

from sim import model as M
from sim import spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--torque", type=float, default=spec.g("TORQUE_MAX"))
    ap.add_argument("--ms", type=float, default=200.0)
    ap.add_argument("--air", action="store_true", help="lift the wheel off the floor")
    ap.add_argument(
        "--nocontact",
        action="store_true",
        help="take the tyre out of the contact model: separates ground friction "
        "from a solver/DoF problem when a wheel refuses to turn",
    )
    args = ap.parse_args()

    model = M.load(fixture=args.fixture)
    if args.nocontact:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tyre")
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
    data = mujoco.MjData(model)
    data.qpos[:] = M.upright_qpos(model)
    if args.air:
        data.qpos[model.jnt_qposadr[0] + 2] += 0.4

    print(
        "bodies=%d joints=%d geoms=%d eq=%d nu=%d nv=%d"
        % (model.nbody, model.njnt, model.ngeom, model.neq, model.nu, model.nv)
    )
    for i in range(model.neq):
        print(
            "  eq %d type=%s obj1=%d obj2=%d active=%d"
            % (
                i,
                model.eq_type[i],
                model.eq_obj1id[i],
                model.eq_obj2id[i],
                model.eq_active0[i] if model.neq else 0,
            )
        )
    for i in range(model.njnt):
        d = model.jnt_dofadr[i]
        print(
            "  joint %-6s type=%s dofadr=%d armature=%.4f damping=%.3f frictionloss=%.3f"
            % (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                model.jnt_type[i],
                d,
                model.dof_armature[d],
                model.dof_damping[d],
                model.dof_frictionloss[d],
            )
        )

    wdof = model.jnt_dofadr[M._jid(model, "wheel")]
    steps = int(args.ms / 1000.0 / model.opt.timestep)
    marks = {int(steps * f) for f in (0.1, 0.5, 1.0)}
    for i in range(steps):
        M.set_torque(model, data, args.torque)
        mujoco.mj_step(model, data)
        if i + 1 in marks or i == steps - 1:
            t = (i + 1) * model.opt.timestep
            # qfrc_constraint[wheel dof] = contact+solver force projected onto the
            # wheel: the resisting torque, which is the number that explains a wheel
            # that will not turn.
            print(
                "  t=%.3f s  wheel=%8.2f rad/s  qacc=%8.2f  ctrl=%.3f  "
                "qfrc_actuator=%7.2f  qfrc_constraint=%8.2f Nm  ncon=%d  pitch=%6.2f deg"
                % (
                    t,
                    M.wheel_speed(model, data),
                    data.qacc[wdof],
                    data.ctrl[0],
                    data.qfrc_actuator[wdof],
                    data.qfrc_constraint[wdof],
                    data.ncon,
                    math.degrees(M.pitch_of(model, data)),
                )
            )
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tyre")
    print(
        "tyre centre world z = %.4f m (radius %.3f -> clearance %.4f)"
        % (
            data.geom_xpos[gid][2],
            spec.g("WHEEL_RADIUS"),
            data.geom_xpos[gid][2] - spec.g("WHEEL_RADIUS"),
        )
    )
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print("  contact %s <-> %s dist=%.5f" % (g1, g2, c.dist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

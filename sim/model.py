"""Build the Floatwheel MJCF from sim/spec.py and load it into MuJoCo.

Usage:
    python -m sim.model --dump          print the generated XML
    python -m sim.model --selftest      load it, report dofs/inertia, drop + drive tests

Topology, and why:

    floor
      deck  (free joint: x, y, z, quat - carries deck + rider mass)
        wheel  (hinge about the deck's y axis at the axle; the hub motor is a torque
                on THIS joint, because a hub motor's stator is bolted to the deck: the
                reaction torque goes into the deck, not into the wheel. A separate
                "rotor" body inside the wheel would have been wrong - a torque between
                wheel and rotor is internal and puts no net torque on the wheel-deck
                pair. Rotor inertia enters as the joint's armature.)

The board is an inverted pendulum about the axle: the deck+rider centre of mass sits
at COM_HEIGHT above it, so upright is unstable and only the controller keeps it up -
which is the point of the sim. The rider's geoms are placed so that the *measured*
combined centre of mass of deck+rider equals COM_HEIGHT; selftest prints what it
actually measured.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import spec

GRAV = 9.81
DECK_LEN, DECK_W, DECK_T = 0.50, 0.19, 0.035
DECK_CLEARANCE = 0.008  # gap between the tyre's crown and the platform underside
RIDER_H = 0.85          # rider box height, centred on the rider mass centre


def deck_z() -> float:
    """Platform centre height above the axle.

    The tyre passes up through the deck line, so the platform cannot sit lower than
    the top of the tyre plus a clearance. At the 0.075 m I first used, the platform
    box cut 69.5 mm into the tyre: the wheel rubbed the deck in every run (found as
    a `tyre <-> platform dist=-0.0695` contact and a wheel that would not turn under
    full torque). Real boards put the platform above the wheel line for this reason.
    """
    return spec.g("WHEEL_RADIUS") + DECK_T / 2 + DECK_CLEARANCE


def rider_z_centre() -> float:
    """Height above the axle that puts the deck+rider CoM at COM_HEIGHT."""
    md = spec.g("DECK_MASS")
    mr = spec.g("RIDER_MASS")
    h = spec.g("COM_HEIGHT")
    return (h * (md + mr) - md * deck_z()) / mr


def mjcf(fixture: bool = False, slope_deg: float | None = None) -> str:
    """Build the model XML.

    fixture=True builds the board bolted to the world (flywheel mode on a stand): the
    deck body has no joint, so only the wheel moves and the motor reaction goes into
    the mount. That is a different model, not a constraint - a weld equality would
    have been wrong, because MuJoCo merges the whole subtree under a welded body and
    that locks the wheel with the deck. Without it the deck is a free body, and a
    board spun up in mid air tumbles about its own centre of mass instead - correct
    physics, useless for measuring the drivetrain.
    """
    R = spec.g("WHEEL_RADIUS")
    W = spec.g("WHEEL_WIDTH")
    deck_mass = spec.g("DECK_MASS")
    rider_mass = spec.g("RIDER_MASS")
    tau_max = spec.g("TORQUE_MAX")
    mu = spec.g("MU_SLIDING")
    mu_roll = spec.g("MU_ROLL")
    mu_tors = spec.g("MU_TORSIONAL")
    step = spec.g("STEP_S")
    arm = spec.g("ARMATURE_INERTIA")
    wheel_mass = spec.g("WHEEL_MASS")
    rz = rider_z_centre()
    zr = "%.4f" % rz
    dz = deck_z()
    joint_tag = '<freejoint name="root"/>' if not fixture else "<!-- bolted: no joint -->"
    # On a stand the wheel hangs clear of the ground. Bolted at ground level the tyre
    # is pressed into the floor and its friction holds the wheel still - measured:
    # 0 rad/s at full torque with contacts, 261 rad/s without them.
    deck_z0 = spec.g("WHEEL_RADIUS") + (0.05 if fixture else 0.0)
    # A slope, expressed the cheap and honest way: tipping gravity by gx = g sin(theta) is the
    # small-angle equivalent of a hill for the accelerations that matter here, and it leaves the
    # contact normal at ~mg cos(theta) — which is exactly the difference between a hill and a
    # slide. On a hill the tyre still has ~mu*mg of grip to spend and the wheel and deck accelerate
    # together; on a slide that grip is gone and they do not. NEXT 361's probe needs that
    # distinction to be expressible, and nothing in the default path: slope_deg is 0.0 for every
    # published number in this repo.
    if slope_deg is None:
        slope_deg = spec.g("SLOPE_DEG")           # 0.0 unless a scenario swept it
    gx = float(np.sin(np.radians(slope_deg))) * GRAV

    return f"""
<mujoco model="floatwheel">
  <compiler angle="radian" inertiafromgeom="true"/>

  <option timestep="{step}" integrator="RK4" iterations="60" tolerance="1e-9"
          solver="Newton" gravity="{gx:.6f} 0 -{GRAV}"/>

  <default>
    <default class="deck">
      <geom contype="1" conaffinity="1" friction="{mu} {mu_roll} {mu_tors}"
            solref="0.006 1" solimp="0.85 0.95 0.001" margin="0.0008"/>
    </default>
    <default class="tyre">
      <geom contype="1" conaffinity="1" friction="{mu} {mu_roll} {mu_tors}"
            solref="0.004 1" solimp="0.9 0.95 0.0005" margin="0.0008"/>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.05" pos="0 0 0"
          friction="{mu} {mu_roll} {mu_tors}" solref="0.004 1" solimp="0.9 0.95 0.0005"
          contype="1" conaffinity="1"/>
    <light name="key" pos="1.8 1.8 3.2" dir="-0.4 -0.4 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="-2.2 0.6 2.0" dir="0.5 0 -1" diffuse="0.35 0.4 0.5"/>

    <body name="deck" pos="0 0 {deck_z0}" childclass="deck">
      {joint_tag}
      <geom name="platform" type="box" size="{DECK_LEN/2} {DECK_W/2} {DECK_T/2}"
            pos="0 0 {dz}" mass="{deck_mass}" rgba="0.16 0.18 0.22 1"/>
      <!-- Rider: one box whose mass centre is placed so the deck+rider CoM lands on
           COM_HEIGHT. No floor contact: a rider ending 1 m off the ground would be a
           MuJoCo mess, and termination on |pitch| > 45 deg is what matters here. -->
      <geom name="rider" type="box" size="0.19 0.17 {RIDER_H/2}" pos="0 0 {zr}"
            mass="{rider_mass}" contype="0" conaffinity="0" rgba="0.75 0.25 0.20 1"/>
      <site name="imu" type="box" size="0.03 0.04 0.01" pos="0 0 {dz + 0.03}"
            rgba="0 1 0 0"/>

      <body name="wheel" pos="0 0 0">
        <joint name="wheel" type="hinge" axis="0 1 0" damping="0.02"
               frictionloss="0.06" armature="{arm}" limited="false"/>
        <geom name="tyre" type="cylinder" size="{R} {W/2}" euler="1.5707963 0 0"
              mass="{wheel_mass}" class="tyre" rgba="0.12 0.12 0.14 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- Hub motor. ctrl is normalised -1..1, gear scales it to newton-metres and
         forcerange clamps at TORQUE_MAX; the current-mode ramp/limits live in
         sim.plant, mirroring how Refloat asks for current, not torque. -->
    <motor name="throttle" joint="wheel" gear="{tau_max}" ctrlrange="-1 1"
           forcerange="-{tau_max} {tau_max}"/>
  </actuator>

  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <jointpos name="wheel_pos" joint="wheel"/>
    <jointvel name="wheel_vel" joint="wheel"/>
  </sensor>
</mujoco>
"""


def set_mu(model, mu: float) -> None:
    """Hot-change the sliding friction of every contact: the runtime version of MU_SLIDING.

    NEXT 363. The grip ladder can only test a surface you START on, because mjcf() bakes MU_SLIDING into
    the three friction sites (the deck class, the tyre class, the floor plane) when the model is built.
    That is enough to ask "does this surface slide", but not "what happens when a rider who is already at
    speed hits a slick patch" — and the latter is the only case the firmware's wheelslip detector can even
    arm on, since main.c:487 wants duty > 0.3 and duty is back-EMF dominated, so a standing-start slide is
    below the detector's own arming gate. Every geom carries the same triple by construction, so writing
    them all is what a rebuild would have done; the rolling and torsional columns stay untouched, because
    they are not what a slide is made of.
    """
    for g in range(model.ngeom):
        model.geom_friction[g][0] = float(mu)


def load(fixture: bool = False, slope_deg: float | None = None) -> "mujoco.MjModel":
    """Build the plant. `slope_deg` tips the world (see `mjcf`'s note); None reads
    spec.SLOPE_DEG, which is 0.0 for every published number in this repo."""
    return mujoco.MjModel.from_xml_string(mjcf(fixture=fixture, slope_deg=slope_deg))


def _jid(model: "mujoco.MjModel", name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError("no joint named %s" % name)
    return jid


def _body_id(model: "mujoco.MjModel", name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError("no body named %s" % name)
    return bid


def describe(model: "mujoco.MjModel") -> dict:
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "mass_all": round(float(model.body_mass.sum()), 2),
        "step_s": model.opt.timestep,
        "torque_range_Nm": (
            float(model.actuator_gear[0][0]),
            float(model.actuator_forcerange[0][1]),
        ),
    }


def pendulum_com_height(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    """Measured height above the axle of the deck body's own mass centre (platform +
    rider geoms). MuJoCo has no per-geom mass field, so this reads the body inertia
    frame position rather than summing geoms. The wheel is a child body whose mass
    acts at the axle, so it contributes no gravity torque about the axle and is
    excluded here."""
    bid = _body_id(model, "deck")
    mujoco.mj_forward(model, data)
    return float(data.xipos[bid][2] - data.xpos[bid][2])


def _root_dof(model: "mujoco.MjModel"):
    """DoF address of the deck's free joint, or None in the bolted (fixture) model."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    return None if jid < 0 else int(model.jnt_dofadr[jid])


def upright_qpos(model: "mujoco.MjModel", x: float = 0.0) -> np.ndarray:
    q = np.zeros(model.nq)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if jid >= 0:  # no free joint in the fixture model
        jadr = model.jnt_qposadr[jid]
        q[jadr + 0], q[jadr + 1], q[jadr + 2] = x, 0.0, spec.g("WHEEL_RADIUS")
        q[jadr + 3] = 1.0
    return q


def _deck_body(model: "mujoco.MjModel") -> int:
    return _body_id(model, "deck")


def _deck_rot(model: "mujoco.MjModel", data: "mujoco.MjData"):
    """3x3 body-to-world rotation matrix of the deck (xmat is stored 9-wide per body)."""
    bid = _deck_body(model)
    return data.xmat[bid].reshape(3, 3)


def pitch_of(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    """Deck pitch in radians, **positive = leaning BACKWARD**, and its rate must be
    the derivative of this same quantity (checked in selftest).

    Two independent things pin this down, and neither is taste:

    * The sign comes from the firmware, not from geometry. Refloat's law is
      `error = setpoint - balance_pitch; p = error * kp * Kt` (pid.c:59-61). Read with
      "forward lean is positive" that command drives the board *away* from the fall -
      the closed loop draws a saturated 40 A and falls anyway. In Refloat's convention
      (balance_pitch positive leaning back) the same text is stabilizing, so the
      convention lives here, in the sensor, and sim/control.py stays byte-faithful to
      pid.c.
    * The value is read off the deck's body-to-world matrix rather than from the
      quaternion by hand: the deck z axis in world is the third column.
    """
    r = _deck_rot(model, data)
    return math.atan2(-float(r[0, 2]), float(r[2, 2]))


def roll_of(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    """Deck roll in radians; positive = top of the rider tipping to the board's right."""
    r = _deck_rot(model, data)
    return math.atan2(-float(r[1, 2]), float(r[2, 2]))


def pitch_rate(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    """Deck angular velocity in rad/s, in the SAME sign convention as pitch_of (freejoint
    qvel is body-frame). Negated from the raw joint rate because firmware mixes the
    two (`rate_p = -pitch_rate * kp2 * Kt` alongside `error = setpoint - balance_pitch`),
    so a rate that disagrees with the angle's sign makes the loop destabilise while
    both numbers look fine in isolation. selftest measures d(pitch)/dt against this.
    """
    rd = _root_dof(model)
    return float("nan") if rd is None else -data.qvel[rd + 4]


def yaw_rate(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    rd = _root_dof(model)
    return float("nan") if rd is None else data.qvel[rd + 5]


def forward_speed(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    """Velocity of the deck frame origin (the axle), m/s."""
    rd = _root_dof(model)
    return float("nan") if rd is None else data.qvel[rd]


def wheel_angle(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    return data.qpos[model.jnt_qposadr[_jid(model, "wheel")]]


def wheel_speed(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    return data.qvel[model.jnt_dofadr[_jid(model, "wheel")]]


def set_torque(model: "mujoco.MjModel", data: "mujoco.MjData", torque: float):
    """Command hub torque in N*m; normalised for the actuator, clamped by forcerange."""
    gear = float(model.actuator_gear[0][0])
    data.ctrl[0] = max(-1.0, min(1.0, torque / gear))


def selftest() -> int:
    model = load()
    print("model:", describe(model))
    for p in spec.check():
        print("spec warning:", p)
    h_measured = pendulum_com_height(model, mujoco.MjData(model))
    print("pendulum CoM: asked %.3f m, built %.3f m" % (spec.g("COM_HEIGHT"), h_measured))
    if abs(h_measured - spec.g("COM_HEIGHT")) > 0.02:
        print("FAIL: rider placement did not land the CoM on COM_HEIGHT")
        return 1

    rc = 0
    # 0) attitude signs, against hand-set rotations. Getting these signs backwards
    #    trains a controller that steers the wrong way and still looks plausible.
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    jadr = model.jnt_qposadr[0]
    jadr_dof = model.jnt_dofadr[0]
    theta = 0.2  # rad about the board's y axis: the top tips toward +x = lean forward
    data.qpos[jadr + 3 : jadr + 7] = [math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0]
    mujoco.mj_forward(model, data)
    p, r = pitch_of(model, data), roll_of(model, data)
    print(
        "sign check: +%.3f rad about y (leaning forward) -> pitch %+.3f roll %+.3f"
        % (theta, p, r)
    )
    # pitch is positive leaning BACK (firmware convention, see pitch_of), so a forward
    # tip must read negative.
    if abs(p + theta) > 1e-3 or abs(r) > 1e-3:
        print("FAIL: pitch formula/sign is wrong (want pitch = -angle about y)")
        return 1
    phi = 0.15  # rad about the board's x axis
    data.qpos[jadr + 3 : jadr + 7] = [math.cos(phi / 2), math.sin(phi / 2), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    p, r = pitch_of(model, data), roll_of(model, data)
    print("sign check: +%.3f rad about x -> pitch %+.3f roll %+.3f" % (phi, p, r))
    if abs(r - phi) > 1e-3 or abs(p) > 1e-3:
        print("FAIL: roll formula/sign is wrong")
        return 1

    # 0b) pitch_rate must be the derivative of pitch_of. The control law adds the two
    #     together, so if they ever disagree in sign the balance loop drives the fall
    #     instead of catching it, while each accessor still looks correct alone.
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = upright_qpos(model)
    tip = math.radians(3.0)
    data.qpos[jadr + 3 : jadr + 7] = [math.cos(tip / 2), 0.0, math.sin(tip / 2), 0.0]
    mujoco.mj_forward(model, data)
    t0, p0 = data.time, pitch_of(model, data)
    rates = []
    for _ in range(int(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        rates.append(pitch_rate(model, data))
    slope = (pitch_of(model, data) - p0) / (data.time - t0)
    reported = sum(rates) / len(rates)
    print(
        "rate check: measured d(pitch)/dt %+.3f rad/s vs pitch_rate %+.3f rad/s"
        % (slope, reported)
    )
    if slope * reported <= 0 or abs(reported - slope) > 0.15 * abs(slope):
        print("FAIL: pitch_rate is not the derivative of pitch_of")
        return 1

    # 1) a *perturbed* upright board must fall. Starting it exactly upright with zero
    #    velocity is an exact unstable equilibrium, so it would sit there forever -
    #    that is correct behaviour, not a working sim, so the test tips it first.
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = upright_qpos(model)
    tip = math.radians(2.0)
    data.qpos[jadr + 3 : jadr + 7] = [math.cos(tip / 2), 0.0, math.sin(tip / 2), 0.0]
    data.qvel[jadr_dof + 4] = 0.05  # a small push to break the symmetry
    for _ in range(int(1.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    p = math.degrees(pitch_of(model, data))
    z = data.qpos[model.jnt_qposadr[0] + 2]
    print("tipped 2 deg, uncontrolled after 1.5 s: pitch %.1f deg, deck origin z %.3f m" % (p, z))
    if abs(p) < 30:
        print("FAIL: an unactuated inverted pendulum did not fall over")
        rc = 1
    if z < 0.02:
        print("FAIL: the deck sank through the floor")
        rc = 1

    # 2) airborne spin-up (what MODE_FLYWHEEL is on the real board): no ground
    #    reaction, so all the motor torque goes into wheel angular momentum and the
    #    deck must take up the opposite angular momentum. Checks the hub-motor
    #    reaction path and the armature figure.
    # 2a) clamped spin-up (flywheel mode on a stand): the deck is welded to the world,
    #     so the wheel spins about a fixed axle and the first-order response against
    #     the analytic inertia is a real measurement. MuJoCo has no dof_inertia
    #     field, hence the analytic cylinder + armature figure here.
    wheel_mass, wheel_R = spec.g("WHEEL_MASS"), spec.g("WHEEL_RADIUS")
    damping = 0.02  # keep in sync with the wheel joint's damping in mjcf()
    I_wheel = 0.5 * wheel_mass * wheel_R * wheel_R + spec.g("ARMATURE_INERTIA")
    clamp = load(fixture=True)
    cdata = mujoco.MjData(clamp)
    cdata.qpos[:] = upright_qpos(clamp)
    tau = spec.g("TORQUE_MAX")
    spin_s = 0.5
    for _ in range(int(spin_s / clamp.opt.timestep)):
        set_torque(clamp, cdata, tau)
        mujoco.mj_step(clamp, cdata)
    w = wheel_speed(clamp, cdata)
    predicted = (tau / damping) * (1.0 - math.exp(-damping * spin_s / I_wheel))
    print(
        "clamped %.2f s at %.0f N*m: wheel %.0f rad/s vs first-order prediction %.0f "
        "(I analytic %.4f, implied %.4f kg m^2)"
        % (spin_s, tau, w, predicted, I_wheel, tau * spin_s / w if w else float("inf"))
    )
    if not 0.9 < w / predicted < 1.1:
        print("FAIL: spin-up did not follow the wheel inertia (check ARMATURE_INERTIA/damping)")
        rc = 1

    # 2b) free-air short burst: nothing holds the deck, so the motor reaction must
    #     spin it the other way. Sign only - a free board tumbles about its own CoM,
    #     which is why the quantitative check above needs the clamp.
    mujoco.mj_resetData(model, data)
    data.qpos[:] = upright_qpos(model)
    data.qpos[jadr + 2] += 0.4  # wheel clear of the floor
    for _ in range(int(0.05 / model.opt.timestep)):
        set_torque(model, data, tau)
        mujoco.mj_step(model, data)
    wy_air = data.qvel[jadr_dof + 4]
    print(
        "free air %.2f s at %.0f N*m: wheel %+.0f rad/s, deck reaction %+.2f rad/s"
        % (0.05, tau, wheel_speed(model, data), wy_air)
    )
    if wy_air > -0.001:
        print("FAIL: motor torque produced no opposite reaction on the deck")
        rc = 1

    # 3) flooring it with no balance controller should throw the rider backward. Not
    #    an assertion about correctness - Refloat's booster term exists for this.
    mujoco.mj_resetData(model, data)
    data.qpos[:] = upright_qpos(model)
    fell_back_at = None
    for i in range(int(1.0 / model.opt.timestep)):
        set_torque(model, data, spec.g("TORQUE_MAX"))
        mujoco.mj_step(model, data)
        if fell_back_at is None and math.degrees(pitch_of(model, data)) < -20:
            fell_back_at = (i + 1) * model.opt.timestep
    print(
        "full torque, no controller: pitched back past -20 deg at %s, ended at "
        "%.2f m/s, wheel %.0f rad/s"
        % (
            "%.3f s" % fell_back_at if fell_back_at else "never",
            forward_speed(model, data),
            wheel_speed(model, data),
        )
    )
    if fell_back_at is None:
        print("FAIL: full throttle with no controller did not upset the board")
        rc = 1

    # 3) torque must saturate at TORQUE_MAX, not run away
    mujoco.mj_resetData(model, data)
    data.qpos[:] = upright_qpos(model)
    data.qvel[:] = 0.0
    data.ctrl[0] = 50.0  # way out of range: must be clamped
    mujoco.mj_step(model, data)
    wdof = model.jnt_dofadr[_jid(model, "wheel")]
    tau_applied = float(data.qfrc_actuator[wdof])
    print("actuator force with ctrl=50: %.1f N*m (must be <= %.1f)" % (tau_applied, spec.g("TORQUE_MAX")))
    if abs(tau_applied) > spec.g("TORQUE_MAX") + 1e-6:
        print("FAIL: actuator exceeded TORQUE_MAX")
        rc = 1

    print("selftest ok" if rc == 0 else "selftest FAILED")
    return rc


if __name__ == "__main__":
    import sys

    if "--dump" in sys.argv:
        print(mjcf())
        raise SystemExit(0)
    raise SystemExit(selftest())

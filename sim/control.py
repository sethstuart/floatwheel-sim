"""The balance/throttle controller, mirroring Refloat's control law.

Usage:
    python -m sim.control                # closed-loop test: hold the board upright
    python -m sim.control --pushes       # ... with random pushes

Why mirror instead of reinventing: the point of the sim is that a gain changed here
means the same thing on the board, so the parameter names, the units they are entered
in, and the order the terms are summed in come from the firmware
(refloat src/main.c:695-735 pid_control(), src/pid.c:51-89, src/booster.c:37-80) and
the default values come from src/conf/settings.xml via sim/firmware_params.py.

Faithful to the firmware:
  * p = error * kp * Kt                       (pid.c:61)
  * i += error * ki * Kt * LOOP_HERTZ_COMPAT * dt, clamped at ki_limit * Kt (pid.c:63)
  * rate_p = -pitch_rate * kp2 * Kt           (pid.c:69, "used instead of d")
  * asymmetric braking scales via 1 Hz EMAs beyond +-500 erpm   (pid.c:72-89)
  * booster: deadband by angle, linear ramp, speed stiffening   (booster.c:37-80)
  * torque -> current through TORQUE_CONSTANT_COMPAT, then the softstart ramp and
    current_max/current_min clamps, then a 25 Hz EMA             (main.c:698-734)

Approximations, named so they do not quietly become truth:
  * balance_pitch comes from a complementary filter, not Refloat's Mahony filter;
    its alpha is derived from mahony_kp.
  * pitch_rate is treated as deg/s, consistent with kp2=0.6 giving meaningful
    damping; if the firmware's IMU reports rad/s this gain is ~57x too soft.
  * the tiltbacks, torque/turn/brake tilt and noseangling are ported in
    sim/protections.py (NEXT 349) with the firmware's line references, and stay OFF unless
    a case asks for them, so the gain tables in docs/TUNING.md remain comparable with
    themselves. ATR joined them in `8698e89`, ten behaviours in all — and it disproved a
    claim this file used to make: I said ATR could not be ported without a motor model
    because `accel_diff` needs expected-vs-measured acceleration (`atr.c:74+`). The plant
    already measures d(erpm)/dt, which is the measured half, and the expected half comes from
    the torque it already applies, so the port was possible all along and the "half-port would
    be worse than none" reasoning was an excuse, not a constraint. The fault state machine and
    `wheelslip` are still unported (see `sim/protections.py`'s docstring for what that costs).
  * the control loop is stepped by whoever drives it (this sim steps it at MuJoCo's
    `model.opt.timestep`, 1 kHz). Do not read `CONTROL_HZ` as the step rate — it is the
    firmware's *loop_frequency* estimate, and the filter alphas are re-derived from the
    actual dt in `_track_step_frequency`, because trusting that constant here mis-tuned
    every EMA in this class by 10x until NEXT 354 found it.
    NEXT 350 asks the rider for the board's `loop_frequency`; until it answers, that number is
    his question to answer, not an assumption to bury in three filter gains.
"""

from __future__ import annotations

import math
import random

import mujoco

from . import firmware_params, model as M, spec
from .protections import Protections

# src/lib/utils.h:34-35
LOOP_HERTZ_COMPAT = 720.0
TORQUE_CONSTANT_COMPAT = 1.5 * 15 * 0.027  # 0.6075 N*m per amp
SOFTSTART_RAMP_A_PER_S = 100.0             # main.c:707


def ema_alpha(cutoff_hz: float, update_hz: float) -> float:
    """filters/ema.c:24-31, second-order Taylor approximation of 1 - e^-omega."""
    omega = min(2.0 * math.pi * cutoff_hz / update_hz, 0.5)
    return omega - 0.5 * omega * omega


class Params:
    """Firmware defaults, overridable per experiment. Names are the settings.xml
    field names so a sweep result can be typed straight into VESC Tool."""

    def __init__(self, **overrides):
        self.v = dict(firmware_params.DEFAULTS)
        self.v["torque_per_current"] = TORQUE_CONSTANT_COMPAT
        self.v["torque_max"] = spec.g("TORQUE_MAX")
        self.v["motor_poles"] = spec.g("MOTOR_POLES")
        self.v["wheel_radius"] = spec.g("WHEEL_RADIUS")
        # Not a settings.xml field — VESC's own config int (motor_data.c:80) — but it behaves
        # like the others above: a hardware constant that Params carries under the name the
        # firmware uses, so a pin can put two pack sizes side by side. It is load-bearing:
        # tiltback_lv/tiltback_hv ship as PER-CELL volts and only mean anything once scaled.
        self.v["si_battery_cells"] = int(spec.g("BATTERY_CELLS"))
        self.v["softstart_start_A"] = 0.0
        self.v.update(overrides)

    def __getitem__(self, k):
        return self.v[k]

    def __setitem__(self, k, val):
        self.v[k] = val

    def get(self, k, default=None):
        return self.v.get(k, default)

    def keys(self):
        return self.v.keys()

    def copy(self, **overrides):
        p = Params()
        p.v = dict(self.v)
        p.v.update(overrides)
        return p


class BalanceController:
    """One instance = one board. Call update() once per control tick."""

    def __init__(self, params: Params | None = None, control_hz: float | None = None,
                 protections: bool = False, handtest: bool = False):
        self.p = params or Params()
        self.hz = control_hz or spec.g("CONTROL_HZ")
        self.a_current = ema_alpha(25.0, self.hz)   # main.c:176 balance_current EMA
        self.a_scale = ema_alpha(1.0, self.hz)       # pid.c brake scales, 1 Hz
        self.a_booster = ema_alpha(1.0, self.hz)     # booster.c torque EMA
        # Off by default: docs/TUNING.md's gain tables were measured without them, and a
        # protection that silently changes a published number is worse than none.
        self.protections = Protections(self.p, self.hz, handtest=handtest) if protections else None
        self.prot_report = {}
        self.t = 0.0
        self._step_hz = None
        self.reset()

    def _track_step_frequency(self, dt: float) -> None:
        """Re-derive the EMA alphas from the step the loop is ACTUALLY taking.

        `CONTROL_HZ` is a tagged `estimate` (NEXT 350 is asking the rider for the real
        `loop_frequency`) and it disagrees with how this class is really driven: run_case
        steps at MuJoCo's `model.opt.timestep`, 1 kHz, while CONTROL_HZ says 100. An alpha
        is per-step, so configuring at 100 Hz and stepping at 1 kHz made `a_booster` — a 1 Hz
        torque filter — behave like ~10 Hz, and `a_scale` likewise. That is not a rounding
        error: the booster is the thing that decides whether a launch from standstill is
        grabby or lazy, and every table in docs/TUNING.md was measured through it.

        Measuring the rate rather than trusting the constant makes the controller right under
        either answer to NEXT 350, and keeps the estimate from becoming a hidden assumption
        buried in three filter gains. Mirrors the firmware, which tracks its own measured
        frequency and reconfigures (main.c:154-163, 785).
        """
        step_hz = 1.0 / dt if dt > 0 else self.hz
        if self._step_hz is None or abs(step_hz - self._step_hz) > 1e-6:
            self._step_hz = step_hz
            self.hz = step_hz
            self.a_current = ema_alpha(25.0, step_hz)   # main.c:176 balance_current EMA
            self.a_scale = ema_alpha(1.0, step_hz)       # pid.c brake scales, 1 Hz
            self.a_booster = ema_alpha(1.0, step_hz)     # booster.c torque EMA

    def reset(self, balance_pitch_deg: float = 0.0):
        self.i_term = 0.0
        self.rate_p = 0.0
        self.p_term = 0.0
        self.booster_torque = 0.0
        self.booster_prop = 0.0            # what main.c:698 fed the booster last tick (pinned)
        self.pitch_used = 0.0              # the imu.pitch of that same line, for the same pin
        # The bridge duty the detector is being told, 0..1. Derived, not injected, unless a scenario
        # asks for a specific one -- see the note at the protections call site below (NEXT 362).
        self.duty_used = 0.0
        self.balance_current = 0.0
        self.softstart_limit = self.p["softstart_start_A"]
        self.p_fwd_scale = 1.0
        self.p_bwd_scale = 1.0
        self.rate_fwd_scale = 1.0
        self.rate_bwd_scale = 1.0
        self.balance_pitch = balance_pitch_deg
        # What the rider/policy asked for. The protections add their angle to THIS, never
        # to self.setpoint, or the offset would be re-added every tick and run away.
        self.setpoint_base = balance_pitch_deg
        self.setpoint = balance_pitch_deg
        self.engaged = False
        self.torque = 0.0
        self.current = 0.0
        self.t = 0.0
        self.prot_report = {}
        if self.protections is not None:
            self.protections.reset()

    # --- balance filter (approximation of balance_filter.c) -------------------
    def update_balance_pitch(self, dt: float, pitch_deg: float, pitch_rate_deg_s: float):
        kp = max(self.p.get("mahony_kp", 2.0), 1e-3)
        alpha = kp * dt / (kp * dt + 1.0)
        self.balance_pitch = (1 - alpha) * (self.balance_pitch + pitch_rate_deg_s * dt) + alpha * pitch_deg

    def engage(self, pitch_deg: float, roll_deg: float, foot_adc: float):
        """main.c:1040: start only inside the pitch/roll tolerance with feet detected."""
        if self.engaged:
            return True
        ok = (
            abs(pitch_deg) <= self.p.get("startup_pitch_tolerance", 4.0)
            and abs(roll_deg) <= self.p.get("startup_roll_tolerance", 45.0)
            and foot_adc >= self.p.get("fault_adc1", 2.0)
        )
        if ok:
            self.engaged = True
            self.reset(balance_pitch_deg=pitch_deg)
            self.engaged = True
        return self.engaged

    # --- booster (booster.c:37-80) --------------------------------------------
    def _booster(self, proportional: float, abs_erpm: float, braking: bool) -> float:
        if braking:
            torque = self.p["brkbooster_current"] * self.p["torque_per_current"]
            angle = self.p["brkbooster_angle"]
            ramp = self.p["brkbooster_ramp"]
        else:
            torque = self.p["booster_current"] * self.p["torque_per_current"]
            angle = self.p["booster_angle"]
            ramp = self.p["booster_ramp"]

        if abs_erpm > 3000:  # booster.c:54-66
            stiffness = min(1.0, (abs_erpm - 3000) / 10000.0)
            if braking:
                torque *= 1.0 + stiffness
            else:
                angle /= 1.0 + stiffness

        ap = abs(proportional)
        if ap <= angle or ramp <= 0:
            target = 0.0
        elif ap - angle < ramp:
            target = torque * math.copysign(1.0, proportional) * (ap - angle) / ramp
        else:
            target = torque * math.copysign(1.0, proportional)
        self.booster_torque += self.a_booster * (target - self.booster_torque)
        return self.booster_torque

    # --- one control tick ------------------------------------------------------
    def update(
        self,
        dt: float,
        pitch_deg: float,
        pitch_rate_deg_s: float,
        roll_deg: float,
        erpm: float,
        setpoint_deg: float | None = None,
        forward_speed: float | None = None,
        yaw_rate_dps: float | None = None,
        battery_volt: float | None = None,
        duty: float | None = None,
        rider_input: float | None = None,
        darkride: bool = False,
    ) -> float:
        """Return the hub torque in N*m. Angles in degrees, erpm from the wheel.

        `forward_speed`, `yaw_rate_dps`, `battery_volt` and `duty` are injected scenario
        inputs for the protections: a 1-D sim going straight never develops yaw, and a
        battery is a scenario input rather than something to model in order to test a
        voltage-triggered lean. Omit one and the behaviour needing it stays inactive.

        `rider_input` is the same kind of injection and belongs to the rider, not the board: the
        stick in [-1, 1] that the remote turns into a lean-in trim (remote.c:149). `darkride`
        flips that trim's sign (remote.c:151-153). Both are inert unless `inputtilt_remote_type`
        says a stick is fitted, which is how the firmware ships (NEXT 365).
        """
        if not self.engaged:
            self.current = 0.0
            self.torque = 0.0
            return 0.0

        self._track_step_frequency(dt)
        self.update_balance_pitch(dt, pitch_deg, pitch_rate_deg_s)
        if setpoint_deg is not None:
            self.setpoint_base = setpoint_deg
        self.t += dt

        if self.protections is not None:
            # NEXT 362 gave the detector a duty at all — until then it had none unless a scenario
            # injected one, so the flag could not latch on an ordinary run. NEXT 363 then replaced that
            # torque proxy with the electrical state, which is what main.c:487's `duty > 0.3` is a test
            # on. A scenario may still inject one (duty_run drives the duty tiltback that way).
            if duty is None:
                duty = measured_duty(erpm, self.current, battery_volt)
            self.duty_used = duty                      # recorded so a probe can audit it
            off = self.protections.update(
                dt,
                erpm=erpm,
                # Direction of travel, which the tiltbacks key on. Fall back to the erpm
                # sign only if the caller did not say: the gear may flip that sign, and a
                # tiltback that trusts the encoder convention trusts the wrong thing.
                forward_speed=(
                    forward_speed if forward_speed is not None else erpm / 1000.0
                ),
                torque_nm=self.torque,            # last tick's, as the firmware sees it
                balance_offset=self.setpoint_base - self.balance_pitch,
                yaw_rate_dps=yaw_rate_dps,
                battery_volt=battery_volt,
                duty=duty,
                braking=self.current < 0,  # motor->braking, motor_data.c:128-152
                now=self.t,
                # The rider's own trim, forwarded untouched: the deadband remap and the angle limit
                # are the remote's business (remote.c:92-97, :149), not the controller's.
                rider_input=rider_input,
                darkride=darkride,
            )
            self.prot_report = off
            self.setpoint = self.setpoint_base + off["setpoint_offset"]
        else:
            # Clear the report rather than leaving the last one standing: the booster reads
            # `braketilt_deg` out of it now, and an offset left over from a tick when protections
            # were enabled would keep leaning the launch assist after they were switched off.
            self.prot_report = {}
            self.setpoint = self.setpoint_base

        kt = self.p["torque_per_current"]
        error = self.setpoint - self.balance_pitch

        # pid.c:59-69
        self.p_term = error * self.p["kp"] * kt
        self.i_term += error * self.p["ki"] * kt * LOOP_HERTZ_COMPAT * dt
        ki_limit = self.p["ki_limit"] * kt
        if ki_limit > 0:
            self.i_term = max(-ki_limit, min(ki_limit, self.i_term))
        self.rate_p = -pitch_rate_deg_s * self.p["kp2"] * kt

        # pid.c:72-89: braking makes the opposite-direction gain scale
        braking = erpm < -500
        reverse = erpm > 500
        kb = self.p["kp_brake"]
        kb2 = self.p["kp2_brake"]
        self.p_fwd_scale += self.a_scale * ((kb if braking else 1.0) - self.p_fwd_scale)
        self.rate_fwd_scale += self.a_scale * ((kb2 if braking else 1.0) - self.rate_fwd_scale)
        self.p_bwd_scale += self.a_scale * ((kb if reverse else 1.0) - self.p_bwd_scale)
        self.rate_bwd_scale += self.a_scale * ((kb2 if reverse else 1.0) - self.rate_bwd_scale)
        self.p_term *= self.p_fwd_scale if self.p_term > 0 else self.p_bwd_scale
        self.rate_p *= self.rate_fwd_scale if self.rate_p > 0 else self.rate_bwd_scale

        # main.c:698 — `booster_proportional = d->setpoint - d->brake_tilt.setpoint.value -
        # d->imu.pitch`. The two halves of that line are the whole point and I skipped the second
        # one for two rounds: the PID is fed the raw setpoint (line 696, which by then already
        # carries brake tilt and every other offset), while the booster is fed the setpoint with
        # brake tilt taken BACK OUT. The firmware wants the launch assist to keep pushing against
        # a deck the brake tilt is pulling out from under the rider, so braking into a hill is
        # exactly where the two terms are supposed to disagree — and "brake_tilt is 0 until
        # braking" is only an excuse for skipping it during the part that does not matter.
        # `braketilt_deg` is the ported `brake_tilt.setpoint.value` (protections.py:839). With
        # protections off it is absent, so the published no-protection table is untouched.
        brake_tilt_deg = self.prot_report.get("braketilt_deg", 0.0)
        booster_prop = self.setpoint - brake_tilt_deg - pitch_deg
        # Diagnostics, not bookkeeping for its own sake: the pin in tools/probe_protections.py
        # compares these three against each other, so a future "simplification" back to
        # (setpoint - pitch) fails a test instead of quietly changing how the board brakes.
        self.booster_prop = booster_prop
        self.pitch_used = pitch_deg
        booster = self._booster(booster_prop, abs(erpm), braking)

        # main.c:703-721
        pitch_based = (self.rate_p + booster) / kt
        if self.softstart_limit < self.p.get("abs_current_max", self.p["torque_max"] / kt):
            pitch_based = min(abs(pitch_based), self.softstart_limit) * (
                -1.0 if pitch_based < 0 else 1.0
            )
            self.softstart_limit += SOFTSTART_RAMP_A_PER_S * dt

        new_current = (self.p_term + self.i_term) / kt + pitch_based
        cmax = self.p.get("abs_current_max", 42.0)
        cmin = self.p.get("abs_current_min", -40.0)
        new_current = max(cmin, min(cmax, new_current))

        self.balance_current += self.a_current * (new_current - self.balance_current)
        self.current = self.balance_current
        self.torque = max(
            -self.p["torque_max"], min(self.p["torque_max"], self.current * kt)
        )
        return self.torque


def erpm_of(wheel_rad_s: float, poles: float) -> float:
    return wheel_rad_s / (2.0 * math.pi) * 60.0 * poles


def measured_duty(erpm: float, current: float, battery_volt: float | None = None) -> float:
    """What `motor->duty_now` would read -- the number main.c:487's `duty > 0.3` is a test on.

    NEXT 363. A bridge duty is not the rider's demand, it is a voltage ratio: back-EMF plus the
    winding's IR drop, over the pack. At 2 m/s that is a few percent, and that is not an oversight in
    the firmware -- it is why the wheelslip detector is allowed to arm only once the board is actually
    moving. Handing it `torque / TORQUE_MAX` instead (what this port did until 362) says 1.0 at any
    speed the controller leans on authority, which is a detector that stands its modifiers down on dry
    asphalt -- the gate's `gripped_latch` row measured exactly that, at mu 1.1.

    Two of the three constants this reads are NEXT 350 estimates and tagged there, because they decide
    when the detector may arm at all; the sign is kept, since VESC reports a negative duty while
    regenerating and motor_data.c:144 only fabs()s the copy it filters for other readers.
    """
    v_pack = float(battery_volt) if battery_volt else (
        spec.g("BATTERY_CELLS") * spec.g("VOLT_PER_CELL_NOMINAL"))
    if v_pack <= 0.0:
        return 0.0
    back_emf = abs(erpm) / max(1.0, spec.g("ERPM_AT_FULL_DUTY")) * v_pack
    ratio = (back_emf + abs(current) * spec.g("MOTOR_RESISTANCE")) / v_pack
    return math.copysign(min(1.0, ratio), current if current else erpm)


def run_case(
    seconds: float,
    scenario: str = "hold",
    params: Params | None = None,
    seed: int = 0,
    protections: bool = False,
    slick_after: tuple | None = None,          # NEXT 363: (t, mu) — grip changes mid-run
    handtest: bool = False,
    rider_input: float | None = None,          # NEXT 365: the remote stick, held for the run
    darkride: bool = False,
) -> dict:
    """Close the loop and measure something that discriminates between gains.

    Importable - tools/tune_sweep.py runs whole grids through this.

    Starting exactly upright is a vacuous test - an exact inverted pendulum with zero
    velocity in perfect equilibrium stands there forever with no controller at all
    (sim.model.selftest proves it), so every gain "survives". Everything here starts
    with an initial lean and rate, and `hold` adds a shove mid-episode.

    scenario "hold":   hold upright, push mid-episode, report recovery time.
    scenario "launch": hold a -1.5 deg setpoint (forward demand), report speed.

    The NEXT 349 scenarios drive INTO a protection, so each one is observed rather than
    assumed - `protections=True` is what makes them mean anything, and each reports which
    behaviour fired and how far it got:

      "speed_ramp"  hard forward demand until the over-speed tiltback should bite.
      "duty_run"    inject a rising duty cycle through tiltback_duty (duty tiltback).
      "torque_hold" hold enough current past torquetilt_start_current to get leaned.
      "turn"        sustained injected yaw rate at speed (turn tilt wants rate AND aggregate).
      "brownout"    inject a battery sag through tiltback_lv and back, to watch the debounce.
      "hard_brake"  spin up, then demand current against travel, for the brake-tilt nose lift.
    """
    model = M.load()
    data = mujoco.MjData(model)
    data.qpos[:] = M.upright_qpos(model)
    rd = M._root_dof(model)
    # rider stepping on: a small lean already developing
    start_tip = math.radians(0.35)
    jadr = model.jnt_qposadr[0]
    data.qpos[jadr + 3 : jadr + 7] = [
        math.cos(start_tip / 2),
        0.0,
        math.sin(start_tip / 2),
        0.0,
    ]
    data.qvel[rd + 4] = math.radians(2.0)

    ctrl = BalanceController(params, protections=protections, handtest=handtest)
    dt = model.opt.timestep
    m_total = spec.g("DECK_MASS") + spec.g("RIDER_MASS")
    h = spec.g("COM_HEIGHT")
    rng = random.Random(seed)

    steps = int(seconds / dt)
    push_at = rng.uniform(1.5, 3.0) if scenario == "hold" else None
    push_N = rng.uniform(200.0, 400.0)  # a hard shove with both hands
    pushed = False
    peak_seen = False
    recovered_at = None
    fell_at = None
    max_pitch = 0.0
    cur_sum = 0.0
    # What the protections did, if they were on. Reported, not assumed: a scenario that
    # was supposed to trigger a tiltback and reports zero did not trigger it, whatever
    # the scenario's name says.
    prot_max = {}
    which_seen = {}
    fired_at = {}
    # Snapshot of the tick where brake tilt was largest, for tools/probe_protections.py's
    # main.c:698 check: the booster's error is the one term in the loop that brake tilt is supposed
    # to bend, and the only honest way to test it is to keep what the controller was actually fed.
    booster_audit = None
    # NEXT 361: what the contact is actually doing, measured rather than declared. `gap` is wheel
    # surface speed minus deck speed. On a hill the tyre is working hard and the gap stays near
    # zero, because wheel and deck still accelerate together; in a slide the wheel runs away from
    # the deck and the gap is metres per second. That one number is what lets a gate tell a hill
    # from a slide without being told which one it is looking at — which this repo could not do
    # before this row: it could only be handed the answer.
    slip_audit = None
    slip_seen = False
    for k in range(steps):
        pitch = math.degrees(M.pitch_of(model, data))
        roll = math.degrees(M.roll_of(model, data))
        qrate = math.degrees(M.pitch_rate(model, data))
        if not ctrl.engaged:
            ctrl.engage(pitch, roll, foot_adc=3.0)
        if push_at is not None and not pushed and data.time >= push_at:
            # A shove on the rider at torso height, as an angular impulse about the
            # axle using the pendulum's I = m h^2. Shoving the *base* instead does
            # nothing at all: a uniform velocity boost of a rigid rig is a symmetry of
            # the dynamics, which is why an earlier version of this "push" reported a
            # 0.0 s recovery and a max pitch no different from no push.
            i_pend = m_total * h * h
            dw = rng.choice([-1.0, 1.0]) * push_N * 0.1 * M.RIDER_H / i_pend
            data.qvel[rd + 4] += dw
            pushed = True
            recovered_at = None
            peak_seen = False
        # Negative setpoint = demand to lean forward = go forward (see pitch_of).
        # Scenario drivers: what the rider asks for, and what has to be injected because a
        # straight-line 1-D sim cannot generate it (duty, battery voltage, yaw rate).
        t = data.time
        # NEXT 363: a surface that changes under a rider who is already at speed. Before the switch the
        # run rides on whatever MU_SLIDING the caller set; after it, on slick_after[1]. Nothing else about
        # the run changes, so the only new thing in the experiment is where the grip is. Idempotent, so no
        # need to track whether it has been applied.
        if slick_after is not None and t >= slick_after[0]:
            M.set_mu(model, slick_after[1])
        setpoint = None
        inject = {}
        if scenario == "launch":
            setpoint = -1.5
        elif scenario == "speed_ramp":
            # Same demand as "launch" (-1.5 deg), which this plant rides for the full 10 s
            # and reaches ~2.5 m/s. A steeper demand sits at the edge of the 2 deg the
            # torque authority can hold, and then the board falls just as it crosses the
            # speed threshold and the test measures the wrong thing. The over-speed
            # tiltback is what has to make this scenario interesting, not the rider.
            setpoint = -1.5
        elif scenario == "duty_run":
            setpoint = -1.5
            inject["duty"] = min(0.99, 0.4 + 0.15 * t)        # crosses tiltback_duty 0.7
        elif scenario == "torque_hold":
            setpoint = -3.0                                   # sustained past start_current
        elif scenario == "turn":
            setpoint = -2.0
            inject["yaw_rate_dps"] = 55.0 if t > 1.0 else 0.0
        elif scenario == "brownout":
            setpoint = -1.0
            if t < 3.0:
                inject["battery_volt"] = 42.0
            elif t < 4.0:
                inject["battery_volt"] = 42.0 - 38.6 * (t - 3.0)   # sag to 3.4 V
            elif t < 7.0:
                inject["battery_volt"] = 3.4
            else:
                inject["battery_volt"] = 42.0                 # recovery: should come back
        elif scenario == "stomp_slick":
            # NEXT 363: ride out the run-up on the surface the run started on, then ask for hard
            # acceleration at the moment the grip is pulled away (slick_after). The deck has the momentum,
            # the wheel has nothing to push against, and — the part the earlier slick rungs were missing —
            # the DEMANDED force exceeds mu*N. A board merely cruising at 0.26 m/s^2 needs ~23 N, which a
            # mu of 0.08 carries without blinking, so a slick patch under a gentle rider is not a slide at
            # all and the run comes back identical to the gripped one.
            stomp_at = slick_after[0] if slick_after else 14.0
            # And let off 0.4 s later, because that is what a rider does when the wheel breaks away, and
            # because a rider who stays standing through the slide is the whole point of the row: the
            # detector has to latch while somebody is still on the board. 0.4 s is a rider's reaction time,
            # not a tuned constant — and it is deliberately longer than the 0.2 s the firmware's own release
            # needs (main.c:504), so the flag cannot be latched only because the demand never came back.
            setpoint = -1.5 if (t < stomp_at or t > stomp_at + 0.4) else -4.0
        elif scenario == "hard_brake":
            # +3.0, not a gentle +1.5. Brake tilt is a transient: the firmware's own comment
            # calls it a "delayed lingering nose lift" (brake_tilt.c:74), and its gates need
            # current < 0, |erpm| > 2000 and the proportional term on the far side of zero
            # all at once - which happens during the brake ONSET, not in the settled
            # deceleration. A gentle demand never satisfies them together.
            setpoint = -2.5 if t < seconds * 0.5 else 3.0     # spin up, then stand on it
        tau = ctrl.update(
            dt,
            pitch,
            qrate,
            roll,
            erpm_of(M.wheel_speed(model, data), ctrl.p["motor_poles"]),
            setpoint_deg=setpoint,
            forward_speed=float(M.forward_speed(model, data)),
            **inject,
            # A scenario holds the stick at one position for the run, the same way SLOPE_DEG is a
            # scenario constant rather than hardware: the board's own trim maths stays in the ported
            # remote, and a sweep varies this the way it varies a hill (NEXT 365).
            rider_input=rider_input,
            darkride=darkride,
        )
        M.set_torque(model, data, tau)
        mujoco.mj_step(model, data)
        # Measure after the step: the angle at the top of this loop is the one from
        # before the push, which is what made push_recovery_s report 0.0 s. And time
        # recovery from the *peak*, not from the shove: an impulse changes the rate,
        # not the angle, so at the instant of the shove the board is still at 0 deg
        # and "back under 0.5 deg" would be trivially true.
        post = abs(math.degrees(M.pitch_of(model, data)))
        max_pitch = max(max_pitch, post)
        cur_sum += abs(ctrl.current)
        bk = ctrl.prot_report.get("braketilt_deg", 0.0) if ctrl.prot_report else 0.0
        if abs(bk) > abs(booster_audit["braketilt_deg"] if booster_audit else 0.0):
            booster_audit = {"braketilt_deg": bk, "booster_prop": ctrl.booster_prop,
                             "setpoint": ctrl.setpoint, "pitch_used": ctrl.pitch_used,
                             "current": ctrl.current}
        wsp = M.wheel_speed(model, data)
        gap = abs(wsp * spec.g("WHEEL_RADIUS") - float(M.forward_speed(model, data)))
        if slip_audit is None or gap > slip_audit["gap_mps"]:
            slip_audit = {"gap_mps": gap, "wheel_rad_s": wsp,
                          "deck_mps": float(M.forward_speed(model, data)),
                          "t_s": round(data.time, 3), "duty": ctrl.duty_used,
                          "wheelslip": bool(ctrl.prot_report.get("wheelslip", 0.0))
                                       if ctrl.prot_report else False,
                          "atr_accel_diff": (ctrl.prot_report.get("atr_accel_diff", 0.0)
                                             if ctrl.prot_report else 0.0)}
        if ctrl.prot_report and ctrl.prot_report.get("wheelslip", 0.0):
            slip_seen = True
        if ctrl.prot_report:
            # `atr_deg` was in this list's firmware-faithful twin (protections.py's report dict)
            # from the day ATR landed and missing from HERE until the NEXT 349 ATR row asked for
            # its peak and got 0.00 — the classic "compiled out and reported as working" shape,
            # except nothing was compiled: the harness simply never recorded the key, so every
            # reader of this dict saw a feature that never leans. If a behaviour is mirrored, it
            # has to be accumulable here or no gate can see it lean.
            for _k in ("tiltback_deg", "noseangling_deg", "torquetilt_deg",
                       "turntilt_deg", "braketilt_deg", "atr_deg"):
                _v = abs(ctrl.prot_report[_k])
                if _v > prot_max.get(_k, 0.0):
                    prot_max[_k] = _v
            _w = ctrl.prot_report.get("tiltback_which") or "-"
            which_seen[_w] = which_seen.get(_w, 0) + 1
            # When did it first amount to something: with the ramp speed this is the
            # delay between the limit being passed and the rider feeling it.
            if abs(ctrl.prot_report["tiltback_deg"]) > 0.5 and "tiltback" not in fired_at:
                fired_at["tiltback"] = round(data.time, 2)
        if pushed and not peak_seen and post > 0.5:
            peak_seen = True
        if pushed and peak_seen and post < 0.5 and recovered_at is None:
            recovered_at = data.time - push_at
        if post > spec.g("STOP_PITCH_DEG"):
            fell_at = data.time
            break

    out = {
        "survived_s": round(fell_at if fell_at is not None else seconds, 2),
        "fell": fell_at is not None,
        "max_pitch_deg": round(max_pitch, 2),
        "mean_abs_current_A": round(float(cur_sum) / max(1, steps), 2),
        "final_speed_mps": round(float(M.forward_speed(model, data)), 2),
    }
    if recovered_at is not None:
        out["push_recovery_s"] = round(recovered_at, 2)
    elif push_at is not None:
        out["push_recovery_s"] = "never"
    if protections:
        out["protections"] = {k: round(v, 2) for k, v in sorted(prot_max.items())}
        out["tiltback_seen"] = which_seen
        out["tiltback_fired_at_s"] = fired_at.get("tiltback")
        out["booster_audit"] = booster_audit
        # NEXT 361: the contact's own evidence, so a `wheelslip` reading can mean "the plant said
        # so" rather than "the scenario asked for it".
        out["slip_audit"] = slip_audit
        out["slip_seen"] = slip_seen
    return out


def main() -> int:
    import sys

    scenarios = ["hold", "launch"]
    m_total = spec.g("DECK_MASS") + spec.g("RIDER_MASS")
    hold = m_total * 9.81 * spec.g("COM_HEIGHT")
    tau_max = spec.g("TORQUE_MAX")
    print(
        "balance authority: %.0f N*m vs %.0f N*m gravity at %.2f m -> can hold %.2f deg lean"
        % (tau_max, hold, spec.g("COM_HEIGHT"), math.degrees(math.asin(min(1.0, tau_max / hold))))
    )
    for scenario in scenarios:
        print("--- %s ---" % scenario)
        for name, over, extra in [
            ("firmware defaults", {}, {}),
            ("stiffer kp40/kp2 1.2", {"kp": 40.0, "kp2": 1.2}, {}),
            ("softer kp8/kp2 0.2", {"kp": 8.0, "kp2": 0.2}, {}),
            ("no rate damping kp2 0", {"kp2": 0.0}, {}),
            # NEXT 365: the rider's lean-in trim, driven through the plant. These two rows differ by
            # the stick and NOTHING else, and both run with protections ON — the trim lives inside the
            # protections stage, so a row with protections off cannot show it at all. (The first
            # version of this row did exactly that: it printed a line byte-identical to the row above
            # it, which is the same "compiled out and reported as working" failure NEXT 362 caught for
            # the wheelslip detector, caught here by comparing against its own twin.) Half-scale stick
            # at a 10 deg limit with the deadband off = 5 deg of demanded lean, ramped by the remote's
            # own filter (remote.c:149-156).
            ("trim none prot-on", {"inputtilt_remote_type": 1.0,
                                   "inputtilt_angle_limit": 10.0,
                                   "inputtilt_deadband": 0.0}, {"protections": True}),
            ("trim 0.5 prot-on", {"inputtilt_remote_type": 1.0,
                                  "inputtilt_angle_limit": 10.0,
                                  "inputtilt_deadband": 0.0},
             {"protections": True, "rider_input": 0.5}),
        ]:
            r = run_case(10.0, scenario, Params(**over), **extra)
            print("  %-22s %s" % (name, r))

    # The protections are driven into their limits by a tool, not here: each behaviour has
    # to be PROVED to fire, proved silent when the features are compiled out, and proved
    # silenced under MODE_HANDTEST - which needs per-behaviour arming values, not one
    # everything-on table where the biggest offset dominates and everything falls over.
    #     python tools/probe_protections.py            # the gate
    #     python tools/probe_protections.py --sweep     # each behaviour's own knob
    print("protections: driven per behaviour by tools/probe_protections.py (NEXT 349)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

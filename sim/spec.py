"""Every physical and control number the Floatwheel sim uses, in one place.

Why a python file and not the XML: the numbers are the part the rider will correct, and
the model builder reads them from here. Each value carries a source tag:

    measured - taken off the real board
    refloat  - read out of the Refloat firmware (file given)
    derived  - computed from other numbers here
    estimate - my engineering guess, REPLACE ME

Run `python -m sim.spec --dump` to print the table with its tags, or
`python -m sim.spec --check` to assert the numbers are self-consistent.

Geometry model: single driven hub wheel, deck+rider rigid above the axle, an
inverted pendulum balanced by wheel torque. That topology is not a guess - it is
what the firmware implements: pid_control() in Refloat's src/main.c:695-734 turns
(setpoint - balance_pitch) into motor current with a rate-P term, a booster, a
softstart current ramp and current limits, and the board only starts when
|balance_pitch| < startup_pitch_tolerance (main.c:1040).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Value:
    v: float
    src: str = "estimate"
    note: str = ""


# --- wheel / drivetrain -------------------------------------------------------
WHEEL_RADIUS = Value(0.127, "estimate", "~10in off-road tyre (ADV = adventure)")
WHEEL_WIDTH = Value(0.056, "estimate", "2.2in tyre")
WHEEL_MASS = Value(4.5, "estimate", "hub motor + rim + tyre, unsprung")
# Rotor inertia enters as the wheel joint's armature (sim/model.py): the rotor bell
# and magnets are rigid with the wheel, so a separate rotor body would only double
# count the same inertia.
ARMATURE_INERTIA = Value(0.012, "estimate", "rotor bell + magnets referred to the shaft")

# Direct drive: no belt/gear. motor_data.c:104 reads si_motor_poles and the FOC
# flux linkage from the VESC motor config, so wheel torque = Kt * current.
MOTOR_POLES = Value(30, "refloat src/motor_data.c:104", "hub motor pole count, est")
TORQUE_PER_CURRENT = Value(0.62, "estimate", "N*m/A - sets accel and the balance authority")
TORQUE_MAX = Value(28.0, "estimate", "N*m continuous peak the hub can put to the ground")
CURRENT_MAX = Value(42.0, "refloat src/main.c:714-718", "flywheel-mode limit is 40A")
CURRENT_MIN = Value(-40.0, "estimate", "regen/brake limit")
CURRENT_RAMP_A_PER_S = Value(400.0, "refloat src/main.c:707", "softstart ramps 100A/... see note")
ELEC_TAU_S = Value(0.004, "estimate", "current loop lag")

# --- the electrical state the DETECTOR reads (NEXT 363) -------------------------
# `motor->duty_now` is what main.c:487's `duty > 0.3` test reads, and it is a voltage ratio, not a
# demand: back-EMF plus the winding's IR drop, over the pack. The port used to hand it
# torque/TORQUE_MAX, which says 1.0 at any speed the moment the controller leans on authority -- and
# that is a detector which stands its modifiers down on dry asphalt. These three carry the honest
# version, and all three are NEXT 350 numbers: they set when the wheelslip detector is even allowed
# to arm, so treat every "would the firmware have noticed" answer as caveated until the rider confirms.
MOTOR_RESISTANCE = Value(0.02, "estimate", "ohms phase-to-phase; the IR term of duty (350)")
ERPM_AT_FULL_DUTY = Value(30000.0, "estimate", "electrical rpm at WOT on a full pack, i.e. back-EMF"
                         " = pack volts; sets where duty stops being small (350)")
VOLT_PER_CELL_NOMINAL = Value(3.5, "estimate", "V/cell used when a run injects no battery_volt"
                              " (12S -> 42 V, what the scenarios already use)")

# --- masses and the pendulum --------------------------------------------------
DECK_MASS = Value(6.5, "estimate", "deck + battery + VESC + lights")
RIDER_MASS = Value(78.0, "measured?", "riding weight; correct me")
# Distance from the wheel axle to the combined deck+rider centre of mass. This
# single number sets how hard balancing is: holding torque = m*g*h*sin(pitch).
COM_HEIGHT = Value(0.95, "estimate", "axle to CoM of deck+rider rigid together")
COM_TRAVEL = Value(0.0, "estimate", "fore/aft offset of CoM from the axle plane")

# --- control law, mirroring Refloat's names -----------------------------------
# Refloat: new_current = torque_to_current(pid.p + pid.i) + pitch_based, where
# pitch_based = torque_to_current(rate_p + booster.torque), both clamped by the
# softstart limit and by current_max/current_min.
PID_KP = Value(35.0, "estimate", "on balance_pitch error, deg -> N*m")
PID_KI = Value(1.2, "estimate", "")
RATE_P_KP = Value(8.0, "estimate", "on raw pitch (not balance pitch), per rad/s")
BOOSTER_KP = Value(4.0, "estimate", "pitch-based booster, refloat booster.c")
SETPOINT_DEG = Value(0.0, "estimate", "lean angle the rider asks the board to hold")
BALANCE_FILTER_ALPHA = Value(0.06, "estimate", "refloat balance_filter.c blending acc+gyro")

TILTBACK_TORQUE = Value(14.0, "estimate", "N*m where torque tiltback starts, torque_tilt.c")
TILTBACK_CURRENT = Value(36.0, "estimate", "A where torque tiltback bites")
BRAKE_TILT_DEG = Value(6.0, "estimate", "brake_tilt.c lean-back setpoint")
TURN_TILT_DEG = Value(8.0, "estimate", "turn_tilt.c lean-steer limit")
STARTUP_PITCH_TOL_DEG = Value(12.0, "refloat src/main.c:1040", "startup_pitch_tolerance")
STOP_PITCH_DEG = Value(45.0, "refloat src/main.c:1065", "|pitch|>45 and |roll|>45 = dismount")

# --- footpad sensors ----------------------------------------------------------
# adc_left/adc_right in Refloat's REALTIME_DATA (doc/commands/REALTIME_DATA.md:52)
# compared against fault_adc1/fault_adc2 thresholds.
FOOTPAD_ADC_AT_STAND = Value(0.05, "estimate", "V with no foot")
FOOTPAD_ADC_FULL = Value(3.0, "estimate", "V at full standing load")
FAULT_ADC1 = Value(0.8, "refloat settings.xml default", "foot-present threshold")
FAULT_ADC2 = Value(1.6, "refloat settings.xml default", "both-feet threshold")

# --- battery ------------------------------------------------------------------
# The pack is the rider's hardware, so the cell count has to come from the board (NEXT 350). It is NOT a
# Refloat settings.xml parameter — the firmware reads it as VESC's own `si_battery_cells`
# (motor_data.c:80) — which is why searching settings.xml for "battery_cells" finds nothing
# and why this lives with the physical constants rather than in the firmware vocabulary.
# Not cosmetic: motor_data.c:81-88 multiplies the LV/HV tiltback thresholds by this whenever
# they are stored below 10 V, and Refloat's defaults (3.0 / 4.3) ARE per-cell, so the cell
# count alone decides the voltage at which the board leans back. 12S is the ADV500's usual
# pack; tagged estimate until the board confirms it.
BATTERY_CELLS = Value(12, "estimate", "VESC si_battery_cells; NEXT 350 asks the rider")

# --- ground and solver --------------------------------------------------------
MU_SLIDING = Value(1.1, "estimate", "tyre on dry asphalt")
# The world tilt used by NEXT 361's traction probe. A scenario knob, not hardware: every published
# number in this repo is produced with it at 0, and `python -m sim.spec --dump` shows it so nobody
# mistakes a 6-degree hill for the board's own geometry. Swept with spec.set() like any other.
SLOPE_DEG = Value(0.0, "scenario", "world tilt for the traction probe; 0 for every published run")
MU_ROLL = Value(0.012, "estimate", "rolling friction coefficient")
MU_TORSIONAL = Value(0.004, "estimate", "")
STEP_S = Value(0.001, "derived", "1 kHz: contact + a 40Hz-ish current loop")
CONTROL_HZ = Value(100.0, "estimate", "refloat reports loop_frequency in telemetry")

# --- sim behaviour switches ---------------------------------------------------
RIDER_RIGID = True          # v1: rider fused with the deck (single pendulum)
ALLOW_LATERAL = True        # free joint; lean-steering is a later addition


def g(name: str) -> float:
    """Float value of a module-level constant."""
    val = globals()[name]
    return float(val.v) if isinstance(val, Value) else float(val)


def set(name: str, value: float) -> float:
    """Override a constant for one experiment; return the previous value.

    Anything that calls g() at use time sees the change, and mjcf() reads spec at build
    time, so a sweep that changes WHEEL_RADIUS genuinely rebuilds the plant. Pass the
    returned value back to set() to put it back. Numbers are `src="override"` so a
    report can never confuse a swept value with a spec value.
    """
    val = globals()[name]
    if isinstance(val, Value):
        prev = float(val.v)
        # Value is frozen ON PURPOSE (a tuned constant must not drift mid-run); the
        # override replaces the module entry rather than mutating the instance in place.
        # Every consumer reads through g()/spec.NAME at use time (nothing from-imports a
        # constant -- checked 2026-11-15), so rebinding is the whole story.
        globals()[name] = Value(float(value), "override", val.note)
    else:
        prev = float(val)
        globals()[name] = Value(float(value), "override", "set at runtime")
    return prev


def table() -> list[tuple[str, float, str, str]]:
    rows = []
    for k, v in sorted(globals().items()):
        if k.isupper() and isinstance(v, Value):
            rows.append((k, v.v, v.src, v.note))
    return rows


def check() -> list[str]:
    """Sanity checks that catch a mistyped number before it becomes a weird sim."""
    out = []
    m_total = g("DECK_MASS") + g("RIDER_MASS") + g("WHEEL_MASS")
    hold = m_total * 9.81 * g("COM_HEIGHT")
    tau = g("TORQUE_MAX")
    # Largest lean the motor can still hold upright, and the accel at full torque.
    ratio = tau / hold
    if ratio >= 1.0:
        out.append("TORQUE_MAX holds the board at any lean: unrealistic for a hub board")
    else:
        hold_deg = math.degrees(math.asin(ratio))
        if not 1.0 <= hold_deg <= 8.0:
            out.append(
                "balance authority is %.1f deg lean at %.0f N*m - suspicious" % (hold_deg, tau)
            )
    accel = tau / (m_total * g("WHEEL_RADIUS"))
    if not 1.0 <= accel <= 8.0:
        out.append("full-throttle accel %.1f m/s^2 (%.1f g) is suspicious" % (accel, accel / 9.81))
    if g("WHEEL_RADIUS") <= 0 or g("COM_HEIGHT") <= 0:
        out.append("negative geometry")
    return out


def _main() -> int:
    import sys

    bad = check()
    if "--check" in sys.argv:
        for b in bad:
            print("FAIL:", b)
        print("ok" if not bad else "%d problems" % len(bad))
        return 1 if bad else 0

    m_total = g("DECK_MASS") + g("RIDER_MASS") + g("WHEEL_MASS")
    tau = g("TORQUE_MAX")
    hold_deg = math.degrees(math.asin(min(1.0, tau / (m_total * 9.81 * g("COM_HEIGHT")))))
    print("%-24s %10s  %-34s %s" % ("param", "value", "source", "note"))
    for k, v, src, note in table():
        print("%-24s %10.4f  %-34s %s" % (k, v, src, note))
    print()
    print("all-up mass            %.1f kg" % m_total)
    print("max holdable lean      %.2f deg  (motor torque vs gravity at %.2f m CoM)" % (hold_deg, g("COM_HEIGHT")))
    print("full-throttle accel    %.2f m/s^2  (%.2f g)" % (tau / (m_total * g("WHEEL_RADIUS")), tau / (m_total * 9.81)))
    for b in bad:
        print("WARNING:", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

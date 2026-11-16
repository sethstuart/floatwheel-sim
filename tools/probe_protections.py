"""NEXT 349 evidence: drive each ported protection into its own limit and PROVE it fires.

Usage:
    python tools/probe_protections.py                 # one run per behaviour, sized to fire
    python tools/probe_protections.py --sweep         ... and sweep each behaviour's own knob
    python tools/probe_protections.py --only turn     # just one

Why this exists rather than a line in `python -m sim.control`: a scenario named "turn"
proves nothing on its own. The assertion has to be that the behaviour it is named after
moved the setpoint by a measurable amount, that it did NOT fire in the run where the
feature is off, and that MODE_HANDTEST silences the modifiers but keeps the hard tiltbacks.
Each SCENARIOS row therefore carries the parameter that arms its behaviour, the key that
keynotes it, and the minimum degrees that counts as having fired. Run by --sweep and by
`python -m sim.protections`'s pins; this one adds the closed-loop plant to it.

Exits non-zero if any behaviour failed to fire, or if anything fired with the features
compiled out. That makes the table a gate, not a printout.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import spec  # noqa: E402
from sim.control import Params, run_case  # noqa: E402
from sim.protections import MIRRORED_BEHAVIOURS  # noqa: E402

# behaviour -> (scenario, arming params, the report key that must move, min degrees)
SCENARIOS = {
    "speed_tiltback": (
        "speed_ramp",
        # 1.0 m/s: the launch demand reaches ~2.5 m/s in 10 s, so this crosses with room
        # to spare and the tiltback - not a fall - decides how the run ends.
        {"tiltback_speed": 1.0, "tiltback_duty_angle": 1.5, "tiltback_duty_speed": 3.0,
         "tiltback_return_speed": 3.0},
        "tiltback_deg", 0.5,
    ),
    "duty_tiltback": (
        "duty_run",
        {"tiltback_duty": 0.7, "tiltback_duty_angle": 1.5, "tiltback_duty_speed": 3.0,
         "tiltback_return_speed": 3.0},
        "tiltback_deg", 0.5,
    ),
    "lv_tiltback": (
        "brownout",
        {"tiltback_lv": 30.0, "tiltback_lv_angle": 1.5, "tiltback_hv": 60.0,
         "tiltback_hv_angle": 1.0, "tiltback_hv_speed": 3.0, "tiltback_return_speed": 3.0},
        "tiltback_deg", 0.5,
    ),
    "hv_tiltback": (
        "brownout",
        # Armed in PER-CELL volts on purpose, and this row is new for NEXT 356: HV is the
        # branch that sat permanently on for a whole release, because the shipped per-cell
        # defaults (3.0 / 4.3) were being compared against pack volts without motor_data.c:81-88
        # scaling them. The lv row above arms in absolute volts instead, so the two rows
        # together cover both sides of the "under 10 V means per-cell" test — 3.4 * 12 cells
        # = 40.8 V, under the scenario's healthy 42.0 V rail, and the >=10 branch is covered
        # by the lv row's tiltback_hv=60.0 never being multiplied.
        {"tiltback_hv": 3.4, "si_battery_cells": 12, "tiltback_hv_angle": 1.5,
         "tiltback_hv_speed": 3.0, "tiltback_lv": 30.0, "tiltback_lv_angle": 1.0,
         "tiltback_return_speed": 3.0},
        "tiltback_deg", 0.5,
    ),
    "torque_tilt": (
        "torque_hold",
        {"torquetilt_strength": 0.3, "torquetilt_start_current": 5.0,
         "torquetilt_angle_limit": 1.5, "torquetilt_strength_regen": 0.3},
        "torquetilt_deg", 0.5,
    ),
    "noseangling": (
        "launch",
        {"tiltback_constant": 1.0, "tiltback_constant_erpm": 800.0, "noseangling_speed": 5.0},
        "noseangling_deg", 0.5,
    ),
    "turn_tilt": (
        "turn",
        {"turntilt_strength": 30.0, "turntilt_start_angle": 5.0, "turntilt_start_erpm": 2000.0,
         "turntilt_angle_limit": 1.5, "turntilt_erpm_boost": 200.0,
         "turntilt_erpm_boost_end": 5000.0, "turntilt_yaw_aggregate": 90.0},
        "turntilt_deg", 0.3,
    ),
    "brake_tilt": (
        "hard_brake",
        # Threshold has a history. When first written this row peaked at 0.13 deg and I
        # documented that as the feature's true transient nature. WRONG, and wrong in a way
        # worth remembering: the brake filter was configured at CONTROL_HZ (100 Hz) and stepped
        # at the plant's 1 kHz timestep, so its alphas were 10x off and the tilt never got
        # anywhere before the gate closed (NEXT 354 found it, and it measures ~3.7 deg now).
        # The lesson is not "brake tilt is small" — it is that a mistuned filter reads like a
        # dead feature, and a threshold lowered to match a dead feature hides the bug.
        {"braketilt_strength": 10.0, "braketilt_lingering": 2.0},
        "braketilt_deg", 1.0,
    ),
    "atr": (
        # ATR leans toward closing the gap between the current it pulls and the acceleration
        # that current buys (atr.c:74+), so nothing flat can arm it: on level ground the
        # launch accelerates on plan and accel_diff sits near zero (that is what the open-loop
        # cruise pin at protections.py's 7c pins — silent at the holding current). What arms
        # it is the 6° slope from slip_check's hill rung, where the wheel works and the board
        # does not get the speed the torque asks for. The arming mirrors that pin's atr_kw so
        # the closed-loop row and the open-loop pin differ only by the plant; the slope is the
        # row's 5th element, a spec constant, not a param.
        "launch",
        {"atr_strength_up": 2.0, "atr_strength_down": 1.0, "atr_threshold_up": 1.5,
         "atr_threshold_down": 1.5, "atr_amps_accel_ratio": 9.0, "atr_amps_decel_ratio": 8.0,
         "atr_angle_limit": 8.0, "atr_speed_boost": 0.0},
        "atr_deg", 0.5,
        {"SLOPE_DEG": 6.0},
    ),
}

# What MODE_HANDTEST is supposed to leave alone, and what it must silence
# (main.c:1484-1495). braketilt_strength is NOT zeroed there, so brake tilt may survive.
# atr_deg belongs in this tuple because MODE_HANDTEST zeroes atr_strength_up/down themselves
# (protections.py HANDTEST_DISABLED) — the open-loop pin proves it on one Protections() call;
# here it is the closed-loop proof, so an ATR lean surviving the handtest means the zeroing
# has been bypassed somewhere.
HANDTEST_SILENT = ("torquetilt_deg", "turntilt_deg", "noseangling_deg", "atr_deg")


def one(name, sweep=False):
    row = SCENARIOS[name]
    scenario, arm, key, min_deg = row[:4]
    # A row's optional 5th element: sim/spec.py constants the scenario needs from the world
    # rather than from the params — the ATR row rides the same 6° slope as slip_check's hill
    # rung, and SLOPE_DEG is a plant constant, not a firmware parameter. Saved and restored
    # the way tools/tune_sweep.py restores anything it sweeps through spec.set.
    spec_over = row[4] if len(row) > 4 else {}
    rows = []
    knobs = {"speed_tiltback": ("tiltback_duty_angle", [1.0, 1.5, 2.5]),
             "duty_tiltback": ("tiltback_duty_angle", [1.0, 1.5, 2.5]),
             "lv_tiltback": ("tiltback_lv_angle", [1.0, 1.5, 2.5]),
             "hv_tiltback": ("tiltback_hv_angle", [1.0, 1.5, 2.5]),
             "torque_tilt": ("torquetilt_angle_limit", [0.8, 1.5, 2.5]),
             "noseangling": ("tiltback_constant", [0.5, 1.0, 2.0]),
             "turn_tilt": ("turntilt_strength", [15.0, 30.0, 60.0]),
             "brake_tilt": ("braketilt_strength", [5.0, 10.0, 15.0]),
             # Not `atr_strength_up`, though that is the knob VESC Tool calls "ATR Uphill
             # Strength", and not by preference: the first version of this row swept the
             # strength over 0.5/1.0/2.0 and every rung peaked at the same 2.52-2.53 deg.
             # ATR is a feedback — it leans until the accel gap closes — so the strength sets
             # how HARD it reaches the lean the hill asks for, not WHERE it settles, and a
             # swept gain cannot fail to fire when the demand is what fires it. What does
             # discriminate is the demand itself, so this sweeps the hill (an uppercase knob
             # is a spec constant, the way slip_check sweeps MU_SLIDING and SLOPE_DEG).
             "atr": ("SLOPE_DEG", [3.0, 6.0, 9.0])}[name]
    for knob_val in (knobs[1] if sweep else [knobs[1][1]]):
        over = dict(arm)
        spec_here = dict(spec_over)
        if knobs[0].isupper():
            spec_here[knobs[0]] = knob_val
        else:
            over[knobs[0]] = knob_val
        saved = {k: spec.set(k, v) for k, v in spec_here.items()}
        try:
            on = run_case(10.0, scenario, Params(**over), protections=True)
            ht = run_case(10.0, scenario, Params(**over), protections=True, handtest=True)
            off = run_case(10.0, scenario, Params(**over))
            got = (on.get("protections") or {}).get(key, 0.0)
            fired = got >= min_deg
            silent_ok = all(
                (ht.get("protections") or {}).get(k, 0.0) < 1e-6 for k in HANDTEST_SILENT
            )
            leak = "protections" in off
            which = ",".join(k for k in on.get("tiltback_seen", {}) if k != "-") or "-"
            print(
                "  %-14s %-12s %s=%-6.2f fired=%-3s fell=%-5s handtest_silent=%-3s "
                "leak=%-3s tiltback=%s ramp=%s"
                % (name, "%s=%g" % (knobs[0], knob_val), key, got, fired, on["fell"],
                   silent_ok, leak, which, on.get("tiltback_fired_at_s"))
            )
            rows.append((name, fired, silent_ok, leak))
        finally:
            for k, v in saved.items():
                spec.set(k, v)
    return rows


def booster_term_check():
    """`main.c:698` fidelity — the line this gate did not used to test, and the port did not used to honour.

        float booster_proportional = d->setpoint - d->brake_tilt.setpoint.value - d->imu.pitch;

    The PID (line 696) gets the setpoint WITH brake tilt in it; the booster gets it with brake tilt
    taken BACK OUT, so the launch assist keeps pushing against a deck the brake tilt is pulling out
    from under the rider. My port fed both from the same `self.setpoint` for two rounds, which costs
    exactly `braketilt_deg` of booster error: measured on this gate at **3.39°** on the `hard_brake`
    scenario's peak-brake tick — `booster_prop` -12.96 where the skipped term gave -9.57 — applied
    through BOOSTER_KP 4.0 in the one situation where the booster is load-bearing, braking from
    speed. The pins' 0.80° value (offset 2.0 / factor -2.5, protections.py:1046) is the same term at
    a tamer demand.

    The assertion is on the controller's own recorded inputs, not on a torque number, so it cannot
    pass for the wrong reason: if someone simplifies the term back to `setpoint - pitch`, the
    arithmetic below stops holding while brake tilt is genuinely non-zero, and this gate goes red.
    """
    scenario, arm, _key, _min_deg = SCENARIOS["brake_tilt"]
    r = run_case(10.0, scenario, Params(**arm), protections=True)
    a = r.get("booster_audit")
    if not a:
        print("  %-14s FAIL the %s scenario recorded no brake-tilt tick to check against"
              % ("booster_term", scenario))
        return [("booster_term", False, "%s recorded no brake-tilt tick" % scenario)]

    want = a["setpoint"] - a["braketilt_deg"] - a["pitch_used"]
    naive = want + a["braketilt_deg"]              # what the term was before it was ported
    exact = abs(a["booster_prop"] - want) < 1e-12
    lived = abs(a["braketilt_deg"]) > 0.2
    # `braking` is reported, not asserted: brake tilt LINGERS (brake_tilt.c:45), so the largest tick
    # can legitimately arrive after the current has come back through zero.
    print("  %-14s braketilt=%+.2f booster_prop=%+.2f would-be=%+.2f delta=%.2f exact=%-3s "
          "braking=%-3s live=%-3s"
          % ("booster_term", a["braketilt_deg"], a["booster_prop"], naive,
             abs(naive - want), exact, a["current"] < 0.0, lived))
    if not exact:
        return [("booster_term", False,
                 "booster saw %+.3f on the %s scenario, main.c:698 wants %+.3f"
                 % (a["booster_prop"], scenario, want))]
    if not lived:
        return [("booster_term", False,
                 "brake tilt peaked at only %.3f deg, so the term is untested (scenario %s)"
                 % (abs(a["braketilt_deg"]), scenario))]
    return [("booster_term", True, "")]


def _brake_duty_check():
    """Same hard acceleration, same |duty|, sign flipped — does the detector tell a launch from a brake?

    Regression pin on the duty convention in sim/control.py. The first version of the derived duty took
    the MAGNITUDE of torque, which told the detector every brake was full throttle: the flag latched, the
    hold timer refreshed for the whole brake (main.c:502-503), and brake_tilt — which main.c:69 says must
    wind down during a slip — never reached its own threshold.

    Worth naming the deviation this guards rather than hiding it: motor_data.c:144 takes fabs() of a
    MEASURED duty and lets the accel/erpm sign test (main.c:486) do the sign work, so on hardware a brake
    does arrive with duty at 1.0. Mine is a torque proxy, not a measurement, so the sign has to be carried
    here instead — and NEXT 363 is the row that replaces the proxy with a back-EMF duty model and makes
    the whole question moot.
    """
    from sim.protections import Protections

    kw = dict(braketilt_strength=10.0, braketilt_lingering=2.0, atr_amps_decel_ratio=40.0)
    seen = []
    for duty in (1.0, -1.0):
        p = Protections(Params(**kw))
        e = 0.0
        for _ in range(400):                        # 0.4 s at 1 kHz: inside the 8 Hz SMA window
            e += 60.0                               # 60000 erpm/s, past main.c:485's 10000
            p.update(0.001, erpm=e, forward_speed=e / 1000.0, torque_nm=28.0 * (1.0 if duty > 0 else -1.0),
                     balance_offset=0.0, braking=duty < 0, duty=duty, battery_volt=42.0)
            if p.wheelslip:
                break
        seen.append(p.wheelslip)
    return (seen[0] and not seen[1],
            "forward duty latched=%s, braking duty latched=%s (want True/False)" % tuple(seen))


def slip_check():
    """NEXT 361: can a slide HAPPEN in this plant, and does the port notice unprompted?

    Nothing injects the flag, no scripted acceleration, no hand-written duty: the only levers are
    the tyre's friction coefficient and the world tilt, both swept through `spec.set` the way
    tools/tune_sweep.py sweeps anything else, and both restored in the `finally`. The balance
    controller stays in the loop, which is the point — a policy riding this board is what has to
    be able to find its traction limit, and if it cannot, this row has to say so out loud.

    Three runs of the same `launch` demand:
      grip   mu 1.1 (the spec's dry asphalt), level      -> the reference
      slide  mu 0.04 (ice, wet paint),      level        -> the tyre cannot deliver the force the
                                                            torque asks for; the wheel surface runs
                                                            away from the deck and the port's own
                                                            detector has to latch
      hill   mu 1.1,                       6 deg slope  -> the wheel is ALSO not reaching the
                                                            speed the torque asks for, which is
                                                            ATR's business, but nothing is sliding
    """
    from sim import spec

    rows = []
    armed = {"braketilt_strength": 10.0, "atr_amps_decel_ratio": 40.0}
    saved = {}

    def sweep(name, value):
        saved.setdefault(name, spec.set(name, value))

    try:
        sweep("MU_SLIDING", 1.1)
        sweep("SLOPE_DEG", 0.0)
        grip = run_case(6.0, "launch", Params(**armed), protections=True)
        # A grip ladder rather than one armed mu, because the first version of this probe learned
        # the hard way that "low mu" and "fell over" are the same event if you push too hard: at
        # mu 0.04 the board cannot stay up at all, the wheel ends up in free air at 215 rad/s, and
        # the detector is CORRECT not to call that a slide. So the question the row actually asks —
        # is there a surface where the wheel runs away while the rider is still riding, and does the
        # port notice? — gets answered by sweeping for it and printing every rung.
        ladder = []
        for mu in (0.04, 0.08, 0.12, 0.16, 0.20, 0.25):
            spec.set("MU_SLIDING", mu)
            ladder.append((mu, run_case(6.0, "launch", Params(**armed), protections=True)))
        # The same search one gear higher, and NEXT 363 is why it is needed: with a MEASURED duty the
        # detector cannot arm below a few thousand erpm at all (duty is back-EMF dominated, and
        # main.c:487 wants duty > 0.3), so a slide now has to be found where the wheel is actually
        # turning fast. `torque_hold` is the sustained -3 deg demand the torque-tilt rows use, which is
        # the closest thing this repo has to flat out.
        hard = []
        for mu in (0.08, 0.10, 0.12, 0.14, 0.16):
            spec.set("MU_SLIDING", mu)
            hard.append((mu, run_case(8.0, "torque_hold", Params(**armed), protections=True)))
        # A third way to reach a slide, and the only one that can clear the detector's own arming gate:
        # get the board up to speed on the good surface FIRST, then pull the grip out from under it. A
        # slide from a standing start never gets the wheel — or the back-EMF — high enough for duty to
        # pass 0.3, so if an at-speed slide is also unreachable, that is a fact about the detector's
        # arming band, not about whether this plant can slide.
        slick = []
        # Switch points are late on purpose: this plant accelerates at g*sin(1.5 deg) ~= 0.26 m/s^2 under
        # the launch demand, so 3 s buys 0.8 m/s and nothing. The wheel needs ~9000 erpm (~4 m/s of surface
        # speed) before the bridge duty is worth main.c:487's 0.3 — that is ~15 s of run-up. (The first
        # version switched at 3-5 s and every slick rung came back byte-identical to the gripped reference,
        # which is what told me the runs were too short rather than the contact too grippy.)
        for t_sw, mu in ((14.0, 0.12), (16.0, 0.16), (18.0, 0.10)):
            spec.set("MU_SLIDING", 1.1)                # the surface it starts on
            rep = run_case(26.0, "stomp_slick", Params(**armed), protections=True,
                           slick_after=(t_sw, mu))
            rep["tag"] = "slick@%.1f mu=%.2f" % (t_sw, mu)
            slick.append((mu, rep))
        spec.set("MU_SLIDING", 1.1)
        spec.set("SLOPE_DEG", 6.0)
        hill = run_case(6.0, "launch", Params(**armed), protections=True)
    finally:
        for k, v in saved.items():
            spec.set(k, v)

    def fmt(tag, rep):
        a = rep.get("slip_audit") or {}
        return "%s gap=%+.2f m/s wheel=%+.1f rad/s deck=%+.2f m/s slip_seen=%s atr_diff=%+.2f fell=%s" % (
            tag, a.get("gap_mps", float("nan")), a.get("wheel_rad_s", float("nan")),
            a.get("deck_mps", float("nan")), rep.get("slip_seen"),
            a.get("atr_accel_diff", float("nan")), rep["fell"])

    # The ladder is the evidence; the summary rows are the gate.
    for tag, mu, rep in ([("launch", m, r) for m, r in ladder]
                         + [("flatout", m, r) for m, r in hard]
                         + [("slick", m, r) for m, r in slick]):
        a = rep.get("slip_audit") or {}
        print("  %-18s %s duty=%+.2f" % (rep.get("tag") or "%s mu=%.2f" % (tag, mu), fmt("", rep),
                                         a.get("duty", float("nan"))))
    # What the ladder actually says, printed rather than smoothed over: below ~0.14 this board
    # cannot stay up at all, and above it the wheel never leaves the deck. Between the two there is
    # no "riding slide" yet — losing traction and falling are one event here, because a rigid
    # 1-D contact has no compliant tyre to keep loading while the wheel slides under the rider, and
    # by the time the wheel has spun up the deck has already pitched out (the run ends at 45 deg).
    # The wheel speeds above (199-280 rad/s) match sim/model.py's own free-air figure (~261 rad/s
    # at full torque), which is how I know those gaps are airborne spin, not traction slip.
    rode_and_slid = [(tag, mu, rep) for tag, cases in (("launch", ladder), ("flatout", hard),
                                                        ("slick", slick))
                     for mu, rep in cases
                     if not rep["fell"] and rep["slip_audit"]
                     and rep["slip_audit"]["gap_mps"] > 0.5 and rep["slip_seen"]]
    airborne_spin = [(mu, rep) for mu, rep in ladder
                     if rep["fell"] and rep["slip_audit"]
                     and rep["slip_audit"]["gap_mps"] > 5.0]
    gripped = [(mu, rep) for mu, rep in ladder if mu >= 0.20]

    def _rs(triples):
        return ", ".join("%s mu=%.2f gap=%.2f m/s duty=%.2f"
                         % (t, m, r["slip_audit"]["gap_mps"], r["slip_audit"]["duty"])
                         for t, m, r in triples) or "none"

    print("  %-14s %s" % (
        "riding_slide",
        ("reachable -- " + _rs(rode_and_slid)) if rode_and_slid else
        ("NOT reachable yet: no rung slid while riding; every big gap was a fall (%s)"
         % ", ".join("mu=%.2f gap=%.1f" % (m, r["slip_audit"]["gap_mps"])
                     for m, r in airborne_spin) or "none")))
    # NEXT 363's other half: a hard brake on the SPEC's own dry asphalt. Printed as `gripped_latch`
    # since 362 and gated from this round -- the row is in `checks` below, so the detail is printed by
    # the gate loop rather than here.
    hb = run_case(8.0, "hard_brake", Params(**armed), protections=True)
    hba = hb["slip_audit"] or {}
    checks = [
        ("no_slip_ref", not grip["slip_seen"] and grip["slip_audit"]["gap_mps"] < 0.15,
         fmt("grip ", grip)),
        ("hill_not_slide",
         (not hill["slip_seen"]) and hill["slip_audit"]["gap_mps"] < 0.15
         and abs(hill["slip_audit"]["atr_accel_diff"]) > 0.25, fmt("hill ", hill)),
        # The contact really can let go: some rung has the wheel surface metres per second away
        # from the deck, with nobody told the plant to slip.
        ("slip_reachable", bool(airborne_spin),
         "biggest gap %.2f m/s at mu=%.2f, against %.3f m/s on the spec's dry asphalt" % (
             max((r["slip_audit"]["gap_mps"] for _, r in airborne_spin), default=0.0),
             min((m for m, _ in airborne_spin), default=float("nan")),
             grip["slip_audit"]["gap_mps"])),
        # The claim NEXT 362 was filed for, and the one 361 could not make: a run that reaches its end
        # with the wheel outrunning the deck and the flag latched by the detector alone. What kept that
        # impossible was not a missing tyre — it was that nothing ever told the detector what duty the
        # bridge was at (main.c:487 wants duty > 0.3), so the flag could only ever fire in a scenario
        # that injected one.
        ("riding_slide_caught", bool(rode_and_slid),
         _rs(rode_and_slid) + " (runs reached their end, nothing injected)"),
        # And where there IS grip, nothing latches — the false-positive test that actually means
        # something. The row this replaces demanded ZERO latches across the whole ladder, which was
        # only a sane demand while duty was hidden from the detector: a board falling over with its
        # wheel accelerating hard at high duty is exactly what main.c:483 calls wheelslip, airborne or
        # not, and the firmware latches on it too. The sim agreeing with the firmware there is the
        # port being right, not broken.
        ("grip_no_false_positive", not any(r["slip_seen"] for _, r in gripped),
         "%d of %d gripped rungs (mu >= 0.20) latched; a falling rung latching is firmware-correct" % (
             sum(1 for _, r in gripped if r["slip_seen"]), len(gripped))),
        # NEXT 363's measure, promoted from a printed line: now that the duty the detector is given is a
        # voltage ratio rather than a demand, a hard brake on the spec's own dry asphalt must NOT read as
        # a slide. It did, until this row's first version: wheel +24.9 rad/s, nothing slipping, flag up.
        ("gripped_latch", not hb["slip_seen"],
         "hard_brake at mu 1.1: latched=%s, gap %.2f m/s, wheel %+.1f rad/s, duty %.2f (want False --"
         " duty is back-EMF and IR now, not torque)" % (
             hb["slip_seen"], hba.get("gap_mps", float("nan")), hba.get("wheel_rad_s", float("nan")),
             hba.get("duty", float("nan")))),
        # Regression pin on the duty convention (sim/control.py) — see _brake_duty_check for why the
        # sign is load-bearing here and where it deviates from the firmware.
        ("brake_is_not_a_slide",) + _brake_duty_check(),
    ]
    for name, ok, detail in checks:
        print("  %-14s %s" % (name, detail))
        if not ok:
            rows.append((name, False, detail.split(";", 1)[0]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="sweep each behaviour's own knob")
    ap.add_argument("--only", choices=sorted(SCENARIOS), help="run just one behaviour")
    a = ap.parse_args()

    names = [a.only] if a.only else list(SCENARIOS)
    # NEXT 349's Measure, printed rather than remembered: how many firmware behaviours the
    # port mirrors (see sim/protections.py's MIRRORED_BEHAVIOURS). The rows below are how
    # many of them are driven into their own limit in the plant by this gate.
    print("behaviours mirrored: %d (%d driven as rows below)"
          % (len(MIRRORED_BEHAVIOURS), len(SCENARIOS)))
    print("protections driven into their limits (%s):" % ("sweep" if a.sweep else "armed values"))
    bad = []
    for n in names:
        for name, fired, silent_ok, leak in one(n, sweep=a.sweep):
            if not fired:
                bad.append("%s never fired" % name)
            if not silent_ok:
                bad.append("%s survived MODE_HANDTEST" % name)
            if leak:
                bad.append("%s fired with protections compiled out" % name)
    # Not a behaviour firing, a behaviour being *obeyed*: the booster has to be fed what the
    # firmware feeds it while brake tilt is live. A protection that fires correctly and is then
    # ignored by the loop it is supposed to lean on is still a wrong sim, and every behaviour check
    # above would still print ok.
    print("  ---")
    for name, ok, why in booster_term_check():
        if not ok:
            bad.append("%s: %s" % (name, why))
    print("  ---")
    for name, ok, why in slip_check():
        if not ok:
            bad.append("%s: %s" % (name, why))
    if bad:
        for b in bad:
            print("FAIL %s" % b)
        return 1
    print("protections driven ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

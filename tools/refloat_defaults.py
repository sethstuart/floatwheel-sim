"""Read the ride-control parameter defaults out of Refloat's settings.xml.

Usage:
    python -m tools.refloat_defaults [--src PATH] [--json] [param ...]

    --src PATH   refloat checkout root (default: ../../lcm-idle-work/refloat)
    --json       emit a JSON object instead of a table
    param ...    restrict to these C field names (default: the ride-control set below)

Why: the sim's tuning surface should carry the numbers the rider actually sees in VESC
Tool, so that a gain learned in sim means the same thing on the board. settings.xml
is the firmware's source of truth for names, order and defaults.

Order of a parameter block is tag, longName, type, transmittable, description (huge
HTML), then valDouble/valInt - so the value is matched across the description.
"""

from __future__ import annotations

import argparse
import json
import os
import re

# Two levels up, not one: the refloat checkout sits beside GitHub/, at
# Documents/lcm-idle-work/refloat, which is what this file's own docstring says. With a single
# ".." the tool resolved to GitHub/lcm-idle-work and reported "no settings.xml", so anyone who
# ran it without --src concluded the checkout was missing and reached for a guess instead.
DEFAULT_SRC = os.path.join("..", "..", "lcm-idle-work", "refloat")

# The parameters the balance loop, throttle path and safety limits actually use.
# Names verified against the 90 tags in settings.xml (they are C field names, so
# guessing from VESC Tool's display labels gets them wrong: "Angle P" is `kp`,
# "Rate P" is `kp2`, and there is no `softstart_*` at all).
RIDE_PARAMS = [
    # balance PID (pid.c)
    "kp",
    "ki",
    "ki_limit",
    "kp2",
    "kp_brake",
    "kp2_brake",
    # IMU filter (balance_filter.c / imu.c)
    "mahony_kp",
    "mahony_kp_roll",
    # booster (booster.c)
    "booster_angle",
    "booster_ramp",
    "booster_current",
    "brkbooster_angle",
    "brkbooster_ramp",
    "brkbooster_current",
    # ATR - accel torque ramp (atr.c)
    "atr_threshold_up",
    "atr_threshold_down",
    "atr_strength_up",
    "atr_strength_down",
    "atr_angle_limit",
    "atr_filter",
    "atr_speed_boost",
    "atr_amps_accel_ratio",
    "atr_amps_decel_ratio",
    # ATR's own smoothing (filters/smooth_setpoint.c:39-62, configured at atr.c:60-71).
    # settings.xml spells these tags with dots, which the BLOCK regex used to reject, so
    # they were invisible to this tool and absent from sim/firmware_params.py.
    # Lines cited from src/conf/settings.xml @ a2b55d9.
    "atr.filter.time_constant",            # :224
    "atr.filter.on_speed_time_constant",   # :256
    "atr.filter.off_speed_time_constant",  # :280
    "atr.filter.on_speed_limit",           # :2173
    "atr.filter.off_speed_limit",          # :2195
    "atr.transition_boost",                # :2217
    # torque tilt's smoothing (configured in torque_tilt.c; used by smooth_setpoint)
    "torque_tilt.filter.time_constant",           # :2261
    "torque_tilt.filter.on_speed_time_constant",  # :2289
    "torque_tilt.filter.off_speed_time_constant",  # :2311
    "torque_tilt.filter.on_speed_limit",   # :1932
    "torque_tilt.filter.off_speed_limit",  # :1954
    # turn tilt's smoothing (turn_tilt.c:47-59; the speed tc is half this value)
    "turn_tilt.filter.time_constant",      # :2333
    # lean-based setpoint shifts
    "torquetilt_strength",
    "torquetilt_start_current",
    "torquetilt_angle_limit",
    "torquetilt_strength_regen",
    "turntilt_strength",
    "turntilt_start_angle",
    "turntilt_start_erpm",
    "turntilt_angle_limit",
    "turntilt_erpm_boost",
    "turntilt_erpm_boost_end",
    "turntilt_yaw_aggregate",
    "braketilt_strength",
    "braketilt_lingering",
    "inputtilt_angle_limit",
    "inputtilt_deadband",
    "inputtilt_remote_type",
    "noseangling_speed",
    # start / stop gating (main.c, state.c)
    "startup_pitch_tolerance",
    "startup_roll_tolerance",
    "startup_speed",
    "startup_simplestart_enabled",
    "startup_pushstart_enabled",
    "startup_dirtylandings_enabled",
    "fault_pitch",
    "fault_roll",
    "fault_delay_pitch",
    "fault_delay_roll",
    "fault_adc1",
    "fault_adc2",
    "fault_adc_half_erpm",
    "fault_is_dual_switch",
    "fault_moving_fault_disabled",
    "fault_reversestop_enabled",
    "fault_darkride_enabled",
    "reverse_stop_angle",
    # torque / speed limits
    "brake_current",
    "tiltback_duty",
    "tiltback_duty_angle",
    "tiltback_duty_speed",
    "tiltback_lv",
    "tiltback_lv_angle",
    "tiltback_lv_speed",
    "tiltback_hv",
    "tiltback_hv_angle",
    "tiltback_hv_speed",
    "tiltback_constant",
    "tiltback_constant_erpm",
    "tiltback_variable",
    "tiltback_variable_erpm",
    "tiltback_variable_max",
    "tiltback_speed",
    "tiltback_return_speed",
    "parking_brake_mode",
]

# Matched tag pairs, so a block that has no value tag (separators, type 0/3 UI-only
# entries) cannot reach forward and steal the *next* parameter's value. A plain
# forward scan did exactly that and silently dropped `kp` from the dump.
# The `\.` in the character classes is not cosmetic: the nested-config parameters
# (atr.filter.*, torque_tilt.filter.*, turn_tilt.filter.*) are tagged with dots, and
# the old `\w+` silently refused to match them at all - which is how a whole set of
# defaults went missing from sim/firmware_params.py while the tool reported success.
BLOCK = re.compile(
    r"<(?P<tag>[\w.]+)>\s*<longName>(?P<long>.*?)</longName>(?P<body>.*?)</(?P=tag)>",
    re.DOTALL,
)
VALUE = re.compile(r"<(?P<vt>valDouble|valInt|valBool)>(?P<val>.*?)</(?P=vt)>", re.DOTALL)


def parse(xml_text: str) -> dict:
    out = {}
    for m in BLOCK.finditer(xml_text):
        vm = VALUE.search(m.group("body"))
        if not vm:
            continue  # UI-only parameter: no serialized value
        raw = vm.group("val").strip()
        vt = vm.group("vt")
        try:
            val = float(raw) if vt != "valBool" else bool(int(raw))
        except ValueError:
            continue
        out.setdefault(m.group("tag"), {"value": val, "longName": m.group("long").strip()})
    return out


HEADER = '''"""Ride-control parameter defaults read out of Refloat's settings.xml.

GENERATED - do not hand edit. Regenerate with:
    python -m tools.refloat_defaults --src <refloat> --emit-module sim/firmware_params.py

source: {src} @ {commit}
These are the values VESC Tool shows on a fresh config, so a gain learned in the sim
means the same thing on the board. Override per experiment; do not edit this file.
"""

DEFAULTS = {json_body}

LONG_NAMES = {long_body}
'''


def emit_module(path: str, picked: dict, src: str, commit: str) -> None:
    import json as _json

    body = _json.dumps({k: v["value"] for k, v in picked.items()}, indent=4)
    longs = _json.dumps({k: v["longName"] for k, v in picked.items()}, indent=4)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER.format(src=src, commit=commit, json_body=body, long_body=longs))
    print("wrote %s (%d params)" % (path, len(picked)))


def git_commit(src: str) -> str:
    try:
        import subprocess

        return (
            subprocess.run(
                ["git", "-C", src, "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--emit-module", metavar="PATH")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    path = os.path.join(args.src, "src", "conf", "settings.xml")
    if not os.path.isfile(path):
        print("no settings.xml at %s" % path)
        return 2
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        defaults = parse(f.read())

    names = args.names or [n for n in RIDE_PARAMS if n in defaults]
    missing = [n for n in (args.names or []) if n not in defaults]
    picked = {n: defaults[n] for n in names}
    if args.emit_module:
        emit_module(args.emit_module, picked, os.path.abspath(args.src), git_commit(args.src))
    if args.json:
        print(json.dumps({k: v["value"] for k, v in picked.items()}, indent=2))
    else:
        for k, v in picked.items():
            print("%-26s %-10s  %s" % (k, v["value"], v["longName"]))
        if missing:
            print("not found in settings.xml: %s" % ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

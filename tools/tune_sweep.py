"""Gain sweeps on Refloat's own parameter names, so a result types straight into VESC Tool.

Usage:
    # the balance controller, on the firmware's gains:
    python -m tools.tune_sweep --what pid --grid kp=10,20,35 kp2=0.3,0.6,1.0 --seeds 3

    # how a trained policy degrades as the rider turns knobs that can actually be turned:
    python -m tools.tune_sweep --what robust --policy runs/ppo_balance_torque \
        --grid TORQUE_MAX=18,28,40 mahony_kp=1,2,4

    # what ATR is worth on a 6° hill: the same cells twice, once with the feature live and
    # once with --atr off, reading the peak lean off the `atr` head (NEXT 349):
    python -m tools.tune_sweep --what pid --scenario launch --protections --grid SLOPE_DEG=6.0
    python -m tools.tune_sweep --what pid --scenario launch --protections --atr off --grid SLOPE_DEG=6.0

Keys are resolved as firmware parameters first (settings.xml names) and then as
sim/spec.py constants (WHEEL_RADIUS, TORQUE_MAX, COM_HEIGHT, ...), so one flag covers
both kinds of knob. Everything is measured with the same harness as the training
report - sim.control.run_case for the PID, tools.train_ppo.evaluate_policy for the
policy - because a sweep that measures something different than the training report is
just decoration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

from sim import control as C
from sim import spec


def parse_grid(pairs: list[str]) -> dict:
    grid = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit("--grid wants name=v1,v2 (got %r)" % p)
        k, vs = p.split("=", 1)
        grid[k.strip()] = [float(v) for v in vs.split(",") if v.strip()]
    return grid


def resolve(key: str, known_params: set) -> str:
    if key in known_params:
        return "param"
    if key.isupper() and hasattr(spec, key):
        return "spec"
    raise SystemExit(
        "unknown knob %r: not a settings.xml parameter and not a sim/spec.py constant" % key
    )


def apply_setting(setting: dict, known_params: set):
    """Return the list of (setter, prev) needed to undo it."""
    undo = []
    for k, v in setting.items():
        if resolve(k, known_params) == "param":
            undo.append(("param", k, None))
        else:
            undo.append(("spec", k, spec.set(k, v)))
    return undo


def undo_setting(undo):
    for kind, k, prev in undo:
        if kind == "spec":
            spec.set(k, prev)


def settings(grid: dict):
    keys = list(grid)
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="pid", choices=["pid", "robust"])
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scenario", default="hold",
                    choices=["hold", "launch", "speed_ramp", "duty_run", "torque_hold",
                             "turn", "brownout", "hard_brake"],
                    help="the last five drive into a firmware protection (NEXT 349) and "
                         "only mean anything with --protections")
    ap.add_argument("--protections", action="store_true",
                    help="run the ported setpoint protections (sim/protections.py) underneath "
                         "the controller and report each behaviour's peak angle per cell")
    ap.add_argument("--handtest", action="store_true",
                    help="with --protections, MODE_HANDTEST: modifiers silenced, hard "
                         "tiltbacks alive (main.c:1481-1497)")
    ap.add_argument("--atr", default="on", choices=["on", "off"],
                    help="with --protections, `off` zeroes atr_strength_up/down in the Params — "
                         "the same two entries MODE_HANDTEST zeroes (main.c:1484-1495), but "
                         "without the mode's other entries or forced values, so an on/off pair "
                         "prices ATR alone. `off` still runs the scenario; it just runs one "
                         "nobody leaned. Needs a scenario that arms ATR, e.g. --scenario launch "
                         "with --grid SLOPE_DEG=6.0 (NEXT 349)")
    ap.add_argument("--policy", default="runs/ppo_balance_torque")
    ap.add_argument("--act-on", default="auto", choices=["auto", "torque", "setpoint"])
    ap.add_argument("--noise-deg", type=float, default=0.0,
                    help="IMU angle noise sigma (deg) injected at eval (NEXT 351); "
                         "rate channels get 20x, per sim/env.py. Recorded in the JSON.")
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    if not grid:
        grid = {"kp": [8.0, 20.0, 35.0, 50.0], "kp2": [0.0, 0.3, 0.6, 1.2]}
        print("no --grid given, using a kp x kp2 default grid")

    known = set(C.Params().v.keys())
    for k in grid:
        resolve(k, known)

    t0 = time.time()
    rows = []
    if args.what == "pid":
        print("sweeping the firmware balance controller, scenario=%s, %d seeds each" % (args.scenario, args.seeds))
        heads = ("fall%", "max|pitch|", "recover_s", "speed_mps", "mean|A|")
        if args.protections:
            heads += ("tiltback", "noseang", "torquetilt", "turntilt", "braketilt", "atr")
        for setting in settings(grid):
            acc = {h: [] for h in heads}
            for s in range(args.seeds):
                undo = apply_setting(setting, known)
                try:
                    p = C.Params(**setting)
                    if args.atr == "off":
                        # The mode knob the port spec asked for: main.c:1484-1495 zeroes these
                        # two under MODE_HANDTEST, and this zeroes just them. Zeroed last,
                        # after the grid, so `off` means off even in a cell that sets
                        # atr_strength_up — there its own head reads 0.00 by definition, which
                        # is exactly what `off` is for.
                        p["atr_strength_up"] = p["atr_strength_down"] = 0.0
                    r = C.run_case(10.0, args.scenario, p, seed=s,
                                   protections=args.protections, handtest=args.handtest)
                finally:
                    undo_setting(undo)
                acc["fall%"].append(100.0 if r["fell"] else 0.0)
                acc["max|pitch|"].append(r["max_pitch_deg"])
                rec = r.get("push_recovery_s", "never")
                acc["recover_s"].append(99.0 if rec == "never" else float(rec))
                acc["speed_mps"].append(abs(r["final_speed_mps"]))
                acc["mean|A|"].append(r["mean_abs_current_A"])
                if args.protections:
                    # Peak angle per behaviour, so a grid cell says which protection was
                    # doing the work - fall% alone cannot tell a tiltback from a bad gain.
                    deg = r.get("protections") or {}
                    for _h, _k in (("tiltback", "tiltback_deg"), ("noseang", "noseangling_deg"),
                                   ("torquetilt", "torquetilt_deg"), ("turntilt", "turntilt_deg"),
                                   ("braketilt", "braketilt_deg"), ("atr", "atr_deg")):
                        acc[_h].append(deg.get(_k, 0.0))
            mean = {h: sum(v) / len(v) for h, v in acc.items()}
            rows.append({"setting": setting, **mean})
            print(
                "%-46s " % json.dumps(setting, sort_keys=True)
                + "  ".join("%s %6.2f" % (h, mean[h]) for h in heads)
            )
    else:
        import numpy as np  # noqa: F401  (imported here so --what pid needs no torch)
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from sim.env import FloatwheelEnv
        from tools.train_ppo import evaluate_policy

        mp = os.path.join(args.policy, "model.zip")
        npx = os.path.join(args.policy, "vecnormalize.pkl")
        if not os.path.isfile(mp):
            raise SystemExit("no trained policy at %s (run tools.train_ppo first)" % mp)
        model = PPO.load(mp, device="cpu")
        # The policy's action shape has to match the one it was TRAINED on: a setpoint
        # policy fed through the torque map is a different controller entirely and the
        # numbers mean nothing. auto reads it back from the run's report.json (tag scene_acton).
        act_on = args.act_on
        if act_on == "auto":
            rj = os.path.join(args.policy, "report.json")
            tag = json.load(open(rj)).get("tag", "") if os.path.isfile(rj) else ""
            act_on = tag.split("_")[-1] if tag.split("_")[-1] in ("torque", "setpoint") else "torque"
            print("act_on=%s (auto, from %s)" % (act_on, "report tag " + tag if tag else "default"))
        # VecNormalize.load wants a vec_env to attach to; give it a throwaway one rather
        # than passing None, which loads but then fails on the first normalize_obs.
        venv = DummyVecEnv([lambda: FloatwheelEnv(scene="balance", act_on=act_on)])
        norm = VecNormalize.load(npx, venv) if os.path.isfile(npx) else None
        print("sweeping %s under a trained policy (%s), fleet protections=%s"
              % (args.policy, mp, args.protections))
        # NEXT 367: `--protections` used to reach only the PID branch above, so `--what policy
        # --protections` printed a protections sweep while stepping the fleet exactly zero times. A flag
        # that is accepted on one branch and ignored on the other is worse than having no flag, because
        # the command line looks like it asked for the safety behaviours.
        heads = ("return", "fall%", "max|pitch|", "mean|A|")
        # heads are PRINT names; evaluate_policy returns the report.json names. The old
        # code indexed the result with the print names and died KeyError on the first
        # row -- the third proof the robust branch had never run end to end (NEXT 346).
        rkey = {"return": "return", "fall%": "fall_rate",
                "max|pitch|": "max_pitch_deg", "mean|A|": "mean_abs_current_A"}
        for setting in settings(grid):
            undo = apply_setting(setting, known)
            try:
                fw = {k: v for k, v in setting.items() if resolve(k, known) == "param"}
                res = evaluate_policy(
                    model, norm, act_on=act_on, episodes=args.eval_episodes,
                    pushes=4.0, overrides=fw, noise_deg=args.noise_deg,
                    protections=args.protections,
                )
            finally:
                undo_setting(undo)
            rows.append({"setting": setting, **{h: res[rkey[h]] for h in heads}})
            print(
                "%-46s " % json.dumps(setting, sort_keys=True)
                + "  ".join("%s %7.2f" % (h, res[rkey[h]]) for h in heads)
            )

    out = args.out or os.path.join("runs", "sweep_%s_%s.json" % (args.what, args.scenario))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"what": args.what, "scenario": args.scenario, "grid": grid,
                   "noise_deg": args.noise_deg, "protections": args.protections,
                   "handtest": args.handtest, "atr": args.atr, "rows": rows}, f, indent=2, sort_keys=True)
    print("%d settings in %.0f s -> %s" % (len(rows), time.time() - t0, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

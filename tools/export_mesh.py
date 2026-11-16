"""Export the board's geometry to JSON for the Blender renderer.

Usage:
    python -m tools.export_mesh [--out out/geometry.json]

Split like this because Blender lives on the Windows side and the sim venv lives in
WSL: this side knows the physics spec, Blender side consumes plain numbers. Keeping
the contract JSON also means the render and the collision model can never drift -
both come from sim/spec.py, and a mesh imported into MuJoCo later would come from
here too.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from sim import model as M
from sim import spec


def geometry(pitch_deg: float = 0.0) -> dict:
    R = spec.g("WHEEL_RADIUS")
    W = spec.g("WHEEL_WIDTH")
    return {
        "wheel": {
            "radius": R,
            "width": W,
            "hub_radius": round(0.42 * R, 4),
            "rim_radius": round(0.86 * R, 4),
            "spokes": 12,
            "source": "spec.py WHEEL_RADIUS/WHEEL_WIDTH (tagged estimate)",
        },
        "deck": {
            "length": M.DECK_LEN,
            "width": M.DECK_W,
            "thickness": M.DECK_T,
            "z_above_axle": round(M.deck_z(), 4),
            "clearance_above_tyre": round(M.deck_z() - M.DECK_T / 2 - R, 4),
        },
        "footpads": {
            "x_offsets": [-0.155, 0.155],
            "length": 0.13,
            "width": round(M.DECK_W - 0.03, 4),
        },
        "led": {"colour": [0.10, 0.85, 1.00, 1.0], "width": 0.006},
        "rider": {
            "height": 1.78,
            "stance_width": 0.16,
            "mass": spec.g("RIDER_MASS"),
            "note": "physics rider is a box fused to the deck; this is for scale only",
        },
        "stance": {"pitch_deg": pitch_deg, "ride_height_m": round(R + M.deck_z(), 4)},
        "sim": {
            "torque_max_Nm": spec.g("TORQUE_MAX"),
            "torque_per_current": 1.5 * 15 * 0.027,
            "holdable_lean_deg": round(
                __import__("math").degrees(
                    __import__("math").asin(
                        min(
                            1.0,
                            spec.g("TORQUE_MAX")
                            / (
                                (spec.g("DECK_MASS") + spec.g("RIDER_MASS"))
                                * 9.81
                                * spec.g("COM_HEIGHT")
                            ),
                        )
                    )
                ),
                2,
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("out", "geometry.json"))
    ap.add_argument("--pitch", type=float, default=0.0)
    args = ap.parse_args()

    g = geometry(args.pitch)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(g, f, indent=2, sort_keys=True)
    print("wrote %s" % args.out)
    print("  wheel  r=%.3f m  w=%.3f m  ride height %.3f m" % (
        g["wheel"]["radius"], g["wheel"]["width"], g["stance"]["ride_height_m"]))
    print("  deck   %.2f x %.2f m, %.3f m above the axle" % (
        g["deck"]["length"], g["deck"]["width"], g["deck"]["z_above_axle"]))
    print("  holdable lean %.2f deg at %.0f N*m" % (
        g["sim"]["holdable_lean_deg"], g["sim"]["torque_max_Nm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

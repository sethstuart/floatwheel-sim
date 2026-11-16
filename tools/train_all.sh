#!/usr/bin/env bash
# Train the whole Floatwheel policy set in one go, into runs/, with a report each.
#
# Usage (from the repo root, inside WSL):
#     bash tools/train_all.sh                 # the three standard runs
#     bash tools/train_all.sh torque          # just one act-on target
#     PY=/path/to/python bash tools/train_all.sh ride
#
# Why a script: launching these through pwsh -> wsl -> bash with nested quotes has
# bitten me three times now (a \$VAR inside a double-quoted bash -lc string expands to
# nothing and every run dies as " -m: command not found" while the pipeline still
# reports exit 0). A file has no quoting problem to get wrong.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/floatwheel/bin/python}"
[ -x "$PY" ] || { echo "no python at $PY (set PY=...)"; exit 1; }
echo "using $PY"

run() {  # run <label> <extra args...>
  echo "=== $1 ==="
  "$PY" -m tools.train_ppo "${@:2}" || { echo "FAILED: $1"; return 1; }
}

want="${1:-all}"

if [ "$want" = "all" ] || [ "$want" = "torque" ]; then
  # pure balance: the policy IS the balance controller, no PID under it
  run torque --act-on torque --timesteps 400000 --out runs/ppo_balance_torque
fi
if [ "$want" = "all" ] || [ "$want" = "setpoint" ]; then
  # the useful shape: PPO steers the setpoint, the firmware PID still closes the loop
  run setpoint --act-on setpoint --timesteps 200000 --out runs/ppo_balance_setpoint
fi
if [ "$want" = "all" ] || [ "$want" = "ride" ]; then
  # ride quality: track a speed target without letting the board get away from you
  run ride --scene ride --act-on setpoint --timesteps 200000 --out runs/ppo_ride_setpoint
fi

echo "=== reports ==="
"$PY" - <<'PYEOF'
import glob, json
for p in sorted(glob.glob("runs/*/report.json")):
    d = json.load(open(p))
    for k in ("learned", "learned_calm", "pid_baseline"):
        r = d.get(k) or {}
        print("%-26s %-14s ret %8.1f  fell %5.1f%%  max|pitch| %6.2f  mean|A| %6.2f"
              % (p.split("/")[1], k, r.get("return", 0), 100 * r.get("fall_rate", 0),
                 r.get("max_pitch_deg", 0), r.get("mean_abs_current_A", 0)))
PYEOF

#!/usr/bin/env bash
# NEXT 351: the mahony_kp curve with IMU noise on, per policy — the tool behind the
# TUNING.md "provisional until noise" verdict. Run from anywhere; writes one JSON per
# (policy, noise) pair under runs/.
#
# Usage:  bash tools/noise_curve.sh [NOISE_LEVELS...]        # default: 0.5 1.0
#         POLICIES="setpoint" KP=1,2,4 EPISODES=6 bash tools/noise_curve.sh 0.5
#
# The grid fixes TORQUE_MAX at the spec value (never swept here) and walks mahony_kp,
# because the DoD is a CURVE against the filter gain, not a cross-product table.
# Cell comparisons are paired: every sweep starts from the env's seed 999, so all cells
# see the same noise realisation — see tools/tune_sweep.py.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/floatwheel/bin/python}"
POLICIES="${POLICIES:-setpoint torque}"
KP="${KP:-0.5,1,2,4,8}"
EPISODES="${EPISODES:-8}"
if [ $# -gt 0 ]; then LEVELS=("$@"); else LEVELS=(0.5 1.0); fi
for pol in $POLICIES; do
  for nz in ${LEVELS[@]}; do
    echo "== $pol @ noise $nz =="
    "$PY" -m tools.tune_sweep --what robust --policy "runs/ppo_balance_$pol" \
      --grid "mahony_kp=$KP" --noise-deg "$nz" --eval-episodes "$EPISODES" \
      --out "runs/sweep_robust351_${pol}_n${nz}.json" 2>&1 \
      | grep -Ev 'UserWarning|warnings.warn|^[[:space:]]*$'
  done
done
echo NOISE-CURVE-DONE

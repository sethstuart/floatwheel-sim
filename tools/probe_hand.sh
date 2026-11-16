#!/usr/bin/env bash
# Probe the hand-written torque controller's sign and gains, to tell an uncontrollable
# env apart from an under-trained policy.
#
# Usage:
#     bash tools/probe_hand.sh                      # sweep kp with the default kd
#     bash tools/probe_hand.sh "-0.7 -1.4 -2.8"     # your own kp list
#     KP="-1.0" KD="-0.01" bash tools/probe_hand.sh
#
# Why: if a two-line P+D controller built from the physics cannot hold the board up
# through the same action mapping PPO uses, the env's torque path is wrong, and training
# longer is wasted compute. This answers that in seconds.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/floatwheel/bin/python}"

# The kd list defaults to the NEGATIVE gains on purpose. Through this model's sign
# convention the rate term has to oppose lean-rate the same way the P term opposes lean,
# so kp and kd must share a sign; with opposite signs the "damping" fights the spring and
# bigger kd falls FASTER (measured: kd 0.007 -> 606 steps, kd 0.10 -> 50 steps). A default
# of positive gains made the bare command look like a broken plant, which is the worst
# possible thing for a script whose whole job is "is the env broken or is training weak".
for kp in ${1:--1.4 -2.8 -5.6}; do
  for kd in ${KDS:--0.002 -0.005 -0.01 -0.02 -0.05}; do
    printf 'kp=%-6s kd=%-6s map=%-7s ' "$kp" "$kd" "${MAP:-linear}"
    "$PY" -m tools.random_rollout --act-on torque --episodes 4 --torque-map "${MAP:-linear}" \
        --hand-kp "$kp" --hand-kd "$kd" 2>&1 | grep hand_torque || echo "no line"
  done
done

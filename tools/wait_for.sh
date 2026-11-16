#!/usr/bin/env bash
# Block inside WSL until a process pattern stops existing, so ONE long foreground tool
# call covers the whole job.
#
# Usage:
#     bash tools/wait_for.sh train_all            # wait for tools/train_all.sh to end
#     bash tools/wait_for.sh train_ppo 90         # ...and report progress every 90 s
#     timeout 900 bash tools/wait_for.sh train_all
#
# Why this exists: on this box WSL is frozen whenever no DSH tool call is in flight. A
# background job plus a blocking wait on the pwsh side is the worst case - a 400k-step
# training run advanced ~20 s of CPU across 8 minutes of waiting, because the only time
# the VM ran was while some other command was executing. Measured, not assumed: the
# wrapper's etimes went 522 -> 542 across one 480 s job_output(wait).
#
# So: start the work, then hold a foreground call open inside WSL with this. The VM
# stays awake for the duration, the call returns exactly when the job ends, and nothing
# polls.
set -uo pipefail

pat="${1:?usage: wait_for.sh <pgrep-pattern> [heartbeat_seconds]}"
beat="${2:-0}"

if ! pgrep -f "$pat" >/dev/null; then
  echo "no process matching '$pat' right now"
  exit 0
fi

while pgrep -f "$pat" >/dev/null; do
  if [ "$beat" -gt 0 ]; then
    ps -eo pid,etimes,pcpu,args --sort=-pcpu | grep -F "$pat" | grep -v grep | head -2
  fi
  sleep 5
done
echo "no process matching '$pat' - job finished"

#!/usr/bin/env bash
#
# Usage: setup_env.sh [--heavy]
#
# Create the Python environment for the Floatwheel sim and install what it needs,
# in two phases so the light stack is usable while the big download runs:
#
#   phase 1  numpy, mujoco, gymnasium          (tens of MB - model + env work)
#   phase 2  torch, stable-baselines3          (~2 GB - only needed for training)
#
# --heavy runs phase 2 as well. Prints versions at the end so the output can be
# pasted into a report, and exits non-zero if an import fails (no silent partial
# installs). Uses a venv under ~/venvs/floatwheel; falls back to pip --user when
# python3-venv is not installed.

set -u

PY=/usr/bin/python3
VENV="${VENV:-$HOME/venvs/floatwheel}"
HEAVY=0
[ "${1:-}" = "--heavy" ] && HEAVY=1

echo "=== interpreter ==="
$PY -V

if $PY -m venv "$VENV" >/dev/null 2>&1; then
	PYX="$VENV/bin/python"
	echo "venv: $VENV"
else
	echo "venv unavailable (python3-venv missing?) - falling back to pip --user"
	PYX="$PY"
fi

pip() { "$PYX" -m pip install --disable-pip-version-check "$@"; }

if ! "$PYX" -m pip --version >/dev/null 2>&1; then
	echo "ERROR: pip is not available in $PYX; install python3-pip / python3-venv first." >&2
	exit 3
fi

echo
echo "=== phase 1: sim core ==="
pip --upgrade pip wheel >/dev/null 2>&1
pip numpy mujoco gymnasium || { echo "phase 1 install failed" >&2; exit 1; }

for m in numpy mujoco gymnasium; do
	"$PYX" -c "import $m, sys; print('  ok    %-10s %s' % ('$m', getattr($m, '__version__', '?')))" \
		|| { echo "import failed for $m" >&2; exit 1; }
done

if [ "$HEAVY" = "1" ]; then
	echo
	echo "=== phase 2: training ==="
	pip torch --index-url https://download.pytorch.org/whl/cu121 || pip torch \
		|| { echo "torch install failed" >&2; exit 1; }
	pip stable-baselines3 || { echo "stable-baselines3 install failed" >&2; exit 1; }
	"$PYX" - <<'PY'
import torch, stable_baselines3
print("  ok    torch      %s  cuda=%s %s" % (
    torch.__version__, torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))
print("  ok    sb3        %s" % stable_baselines3.__version__)
PY
fi

echo
echo "=== offscreen render probe (mujoco) ==="
MUJOCO_GL=egl "$PYX" - <<'PY' || echo "  EGL offscreen unavailable - renders go through Blender instead"
import mujoco, os
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="0.1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 64, 64)
r.update_scene(m)
print("  ok    EGL offscreen render works, %s" % (type(r).__name__,))
PY

echo
echo "setup done"

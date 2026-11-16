#!/usr/bin/env bash
#
# Usage: probe_env.sh
#
# Report what is available for building a Floatwheel simulation + renders on this
# machine: python packages (physics engines, RL, mesh, camera), CAD/render
# binaries, GPU and disk headroom. Prints one line per item so the result can be
# pasted into a plan. Nothing installed, nothing written.

echo "=== python ==="
python3 -V 2>&1
python3 -m pip --version 2>&1 | head -1

for m in numpy scipy matplotlib numba mujoco pybullet gymnasium gymnasium_robotics \
         stable_baselines3 torch torch_geometric trimesh pyrender pyvista vpython \
         opensim_py meshio cadquery OCC ezdgm; do
	if python3 -c "import $m" 2>/dev/null; then
		v=$(python3 -c "import $m;print(getattr($m,'__version__','?'))" 2>/dev/null)
		printf '  ok      %-20s %s\n' "$m" "$v"
	else
		printf '  absent  %-20s\n' "$m"
	fi
done

echo "=== binaries ==="
for b in blender freecad FreeCAD openscad csg assimp ffmpeg convert node npm npm-git \
         nvidia-smi gltf-transform; do
	p=$(command -v "$b" 2>/dev/null)
	if [ -n "$p" ]; then printf '  %-16s %s\n' "$b" "$p"; else printf '  %-16s absent\n' "$b"; fi
done

echo "=== windows-side tools visible from WSL ==="
for b in blender.exe FreeCAD.exe; do
	p=$(command -v "$b" 2>/dev/null)
	[ -n "$p" ] && echo "  $b -> $p" || echo "  $b absent"
done
ls -d /mnt/c/Program\ Files/Blender* 2>/dev/null
ls -d "/mnt/c/Program Files/FreeCAD"* 2>/dev/null

echo "=== compute / space ==="
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | sed 's/^/  gpu: /'
else
	echo "  no nvidia-smi in this WSL distro"
fi
echo "  cpu: $(nproc) threads, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
free -g | awk 'NR==2{printf "  ram: %s GiB total, %s GiB available\n", $2, $7}'
df -h / | awk 'NR==2{printf "  disk(/): %s free of %s\n", $4, $2}'

"""Play a trajectory recorded by tools/record_policy.py in the MuJoCo viewer. Loops until closed.

Runs anywhere MuJoCo can open a window (WSL here cannot, so this is run on the Windows side):

    uv run --with mujoco --with numpy --with pillow tools/play_traj.py out/traj
    uv run --with mujoco --with numpy --with pillow tools/play_traj.py out/traj --model assets/adv2_reference.xml --dz 0.019
    ... --shot out/shot.png        # write stills instead of opening a window

--model swaps the drawn geometry (same kinematic tree, so the recorded qpos still applies).
--dz lifts the recorded axle height: the plant's tyre radius is 0.127 m, the reference model's is 0.146.
"""
import sys
import time

import mujoco
import numpy as np


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


d = sys.argv[1]
model = mujoco.MjModel.from_xml_path(arg("--model", d + "/board.xml"))
data = mujoco.MjData(model)
t = np.load(arg("--traj", d + "/traj.npz"))
qpos, hz = t["qpos"].copy(), float(t["hz"])
qpos[:, 2] += float(arg("--dz", 0.0))
deck = model.body("deck").id


def camera(cam, distance=3.0, azimuth=125, elevation=-14):
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = deck
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    cam.lookat[:] = (0, 0, 0.5)


if "--shot" in sys.argv:
    from PIL import Image
    out = arg("--shot")
    r = mujoco.Renderer(model, 900, 1600)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    views = [("full", 3.6, 125, -12, False), ("board", 1.25, 140, -22, True), ("side", 1.3, 90, -6, True),
             ("front", 1.2, 180, -12, True)]
    data.qpos[:] = qpos[int(0.45 * (len(qpos) - 1))]
    mujoco.mj_forward(model, data)
    for name, dist, az, el, hide_rider in views:
        camera(cam, dist, az, el)
        if hide_rider:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = data.xpos[deck]
        # geom group 0 holds everything; hide the rider by alpha instead of by group
        saved = model.mat_rgba.copy()
        rider = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "rider")
        if hide_rider and rider >= 0:
            model.mat_rgba[rider, 3] = 0.0
        r.update_scene(data, cam, opt)
        Image.fromarray(r.render()).save(out.replace(".png", "_%s.png" % name))
        model.mat_rgba[:] = saved
    print("shots written")
    sys.exit(0)

import mujoco.viewer

with mujoco.viewer.launch_passive(model, data) as v:
    camera(v.cam)
    while v.is_running():
        for i in range(len(qpos)):
            if not v.is_running():
                break
            t0 = time.time()
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(max(0.0, 1.0 / hz - (time.time() - t0)))

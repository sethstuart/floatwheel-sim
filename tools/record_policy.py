"""Record a trained policy riding, headless: writes the plant's MJCF and a qpos trajectory.

    python -m tools.record_policy out/traj [run_dir] [scene] [act_on] [episodes]

Defaults: runs/ppo_ride_setpoint ride setpoint 4. Play the result back with tools/play_traj.py.
"""
import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from sim import model as M
from sim.env import FloatwheelEnv

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
run = sys.argv[2] if len(sys.argv) > 2 else "runs/ppo_ride_setpoint"
scene = sys.argv[3] if len(sys.argv) > 3 else "ride"
act_on = sys.argv[4] if len(sys.argv) > 4 else "setpoint"
episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 4

env = FloatwheelEnv(scene=scene, act_on=act_on, push_mean_s=4.0, seed=999)
policy = PPO.load(run + "/model.zip", device="cpu")
qpos, pitch, speed, ep_of = [], [], [], []
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        qpos.append(env.data.qpos.copy())
        pitch.append(float(np.degrees(M.pitch_of(env.model, env.data))))
        speed.append(float(M.forward_speed(env.model, env.data)))
        ep_of.append(ep)
        done = term or trunc
    print("episode %d: %d frames so far, terminated=%s" % (ep, len(qpos), term))

open(out + "/board.xml", "w").write(M.mjcf())
np.savez(out + "/traj.npz", qpos=np.array(qpos), pitch=np.array(pitch), speed=np.array(speed),
         ep=np.array(ep_of), hz=float(env.control_hz))
json.dump({"run": run, "scene": scene, "act_on": act_on, "frames": len(qpos),
           "max_abs_pitch_deg": float(np.max(np.abs(pitch))), "max_speed": float(np.max(np.abs(speed)))},
          open(out + "/traj.json", "w"))
print(open(out + "/traj.json").read())

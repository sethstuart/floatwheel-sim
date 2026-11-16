# floatwheel-sim

A 3D model and MuJoCo simulation of a Floatwheel ADV2 (VESC controller ID `ADV500`), built so that a balance
policy can be trained here and the gains mean the same thing on the real board.

Status: **training works** — plant, firmware-mirroring controller, Gymnasium env, PPO
policies, gain sweeps, headless Blender renders. What the policies and sweeps actually
produce is in [docs/TUNING.md](docs/TUNING.md).

## Layout

| file | what it is |
| --- | --- |
| `sim/spec.py` | every physical number in one place, each tagged `measured` / `refloat` / `derived` / **`estimate`** |
| `sim/model.py` | MJCF builder + accessors + `--selftest` (six physics checks that must pass) |
| `sim/control.py` | the balance/throttle controller, mirroring Refloat's `pid_control()` |
| `sim/firmware_params.py` | **generated**: Refloat's own parameter defaults, from `settings.xml` |
| `sim/env.py` | Gymnasium env, scenes `balance` / `ride`, observations named after Refloat telemetry |
| `tools/probe_env.sh` | what this box actually has (GPU, RAM, python packages) — drove the stack choice |
| `tools/setup_env.sh` | create `~/venvs/floatwheel` (mujoco, gymnasium, torch, stable-baselines3) |
| `tools/diag.py` | one-run dump: joint inertias, contact pairs, torque vs wheel speed |
| `tools/refloat_defaults.py` | read parameter defaults out of Refloat's `settings.xml`; `--emit-module` |
| `tools/random_rollout.py` | falsification test: random policy must fall, PID must not |
| `tools/export_mesh.py` | geometry JSON for the renderer, straight from `spec.py` |
| `tools/blender_scene.py` | headless Blender: builds the visual board and renders it |
| `tools/train_ppo.py` | PPO: `evaluate_policy()` is the shared measuring stick, importable |
| `tools/train_all.sh` | train the three policy runs in one go (`bash tools/train_all.sh [torque]`) |
| `tools/tune_sweep.py` | gain sweeps on Refloat's own parameter names → VESC Tool numbers |
| `docs/TUNING.md` | what the sweeps and the trained policies found, with the numbers |
| `tools/record_policy.py` | roll a trained policy headless and save the trajectory |
| `tools/play_traj.py` | replay a saved trajectory in the MuJoCo viewer, or write stills |
| `assets/adv2_reference.xml` | a onewheel-shaped reference model (wheel through the deck, rails, footpads) |

## Train it

```bash
bash tools/train_all.sh                       # all three policies, then a summary
~/venvs/floatwheel/bin/python -m tools.train_ppo --act-on setpoint --timesteps 200000
~/venvs/floatwheel/bin/python -m tools.tune_sweep --what pid --grid kp=8,20,35 kp2=0.3,0.6,1.2
~/venvs/floatwheel/bin/python -m tools.tune_sweep --what robust --policy runs/ppo_balance_setpoint \
    --grid TORQUE_MAX=18,28,40 mahony_kp=1,2,4
```

`--grid` keys are settings.xml parameter names first, `sim/spec.py` constants second, so
one flag sweeps both the controller and the plant.

## Watch a policy

```bash
# where the sim venv lives (headless is fine)
~/venvs/floatwheel/bin/python -m tools.record_policy out/traj runs/ppo_ride_setpoint ride setpoint 4
# anywhere MuJoCo can open a window
uv run --with mujoco --with numpy --with pillow tools/play_traj.py out/traj --model assets/adv2_reference.xml --dz 0.019
```

**The plant's geometry is still a placeholder**: `sim/model.py` puts a plank on top of a narrow
tyre, which is not how the board is built. `assets/adv2_reference.xml` has the right topology
(wheel through the deck, rails either side, footpads fore and aft) but approximate dimensions, and
is only used for drawing so far. Moving the plant onto real ADV2 dimensions is the next job, and the
numbers in `docs/TUNING.md` will need re-measuring when that lands.

## Run it

```bash
# WSL, python 3.10, venv created by setup_env.sh
~/venvs/floatwheel/bin/python -m sim.model --selftest        # must print "selftest ok"
~/venvs/floatwheel/bin/python -m sim.control                  # closed loop, gain comparison
~/venvs/floatwheel/bin/python -m sim.env                      # env smoke test
~/venvs/floatwheel/bin/python -m tools.random_rollout         # random vs PID
~/venvs/floatwheel/bin/python -m tools.diag --fixture         # drivetrain on a stand
```

Render (Blender lives on the Windows side, the sim venv in WSL — the JSON is the bridge):

```powershell
# run from the repo root, three times with the view you want: beauty / pushed / side
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" -b --factory-startup `
  --python tools/blender_scene.py -- --geometry out/geometry.json `
  --out out/render_beauty.png --view beauty --engine cycles --samples 96 --res 1400x875
```

`--view side` is the end-on shot, and it is the one that proves the topology: one wheel,
one deck, nothing else touching the ground. `--view pushed --pitch 6` is the board
mid-lean. `--engine auto` takes Cycles on the GPU and falls back to EEVEE, printing
which one it used (`device=GPU` in the `RENDERED` line). MuJoCo's own offscreen renderer
does not work here at all — see Environment notes.

## What the board actually is

Single wheel, rider standing over it, balanced by wheel torque — an inverted pendulum,
**not** a four-wheel skateboard. Read off the firmware rather than guessed:
`refloat/src/main.c:695-734` turns `setpoint - balance_pitch` into motor **current**
(plus a rate-P term and a booster), it only starts when `|balance_pitch|` is inside
`startup_pitch_tolerance`, and it dismounts at 45°. That is a self-balancing machine.

Consequences the sim encodes:

* **Motor torque goes on the wheel↔deck hinge**, not on an internal rotor. The stator is
  bolted to the deck, so a rotor body inside the wheel would make the torque internal —
  no net torque on the wheel-deck pair, and the board would not balance. The rotor's
  inertia still matters, and enters as joint `armature`.
* **The platform must sit above the tyre line.** My first deck box cut 69.5 mm into the
  tyre; the wheel rubbed the deck in every run and would not turn at full torque
  (`tyre <-> platform dist=-0.0695`). Hence `deck_z()`.
* **Holdable lean is only ~2°.** 28 N·m of hub torque against 787 N·m of gravity at
  0.95 m CoM gives `asin(28/787) = 2.04°`. Any "lean" number bigger than that is the
  booster and the setpoint, not steady-state authority.

## Sign conventions (read this before touching a gain)

`sim/model.py:pitch_of` is **positive leaning BACKWARD**, matching Refloat's
`balance_pitch`, and `pitch_rate` is the derivative of that same quantity (checked, not
assumed). With "forward is positive" the firmware's own law `error = setpoint - pitch`
drives the board *into* the fall — it drew a saturated 40 A and fell over in every gain
set while looking perfectly reasonable. The convention lives in the sensor accessor so
`sim/control.py` can stay textually faithful to `pid.c`.

## What is measured, not assumed

* Clamped (on-a-stand) spin-up at full 28 N·m reaches **261 rad/s** in 0.5 s against a
  first-order prediction of 262 — the drivetrain, damping and armature are consistent.
* An unactuated tipped board is gone in 1.5 s (**−133°**): the pendulum is real.
* Full throttle with no controller pitches the deck back past −20° in **0.22 s** — the
  hub reaction pitching the rider off, which is what Refloat's booster exists to fix.
* `kp2 = 0` (no rate damping) **falls over** under load while the firmware defaults hold
  0.4-0.9°. The test discriminates, so it can be used to judge a tune.
* A random torque policy falls 100% of episodes in ~0.8 s; the PID holds all of them.
* Off the ground the same spin-up gives 261 rad/s; pressed against the floor it gives 0 —
  ground friction, which is why `--fixture` hangs the wheel clear like a real stand.

## Numbers that still need measuring

Everything tagged `estimate` in `sim/spec.py`. The ones that actually move behaviour:

| parameter | current value | why it matters |
| --- | --- | --- |
| `WHEEL_RADIUS` | 0.127 m (7.00" tyre) | every lever arm and speed figure |
| `WHEEL_MASS` / `ARMATURE_INERTIA` | 4.5 kg / 0.012 kg·m² | spin-up rate, how the booster feels |
| `TORQUE_MAX` | 28 N·m | sets the 2° holdable lean — the single most important number here |
| `TORQUE_PER_CURRENT` | 0.62 N·m/A | firmware's own compat constant is 0.6075, so close |
| `RIDER_MASS` / `COM_HEIGHT` | 78 kg / 0.95 m | gravity torque, and so everything about stiffness |
| `DECK_MASS` | 6.5 kg | minor |
| `CONTROL_HZ` | 100 Hz | Refloat's loop runs in the IMU callback; `loop_frequency` on your board settles it |
| `MU_SLIDING` | 1.1 | launch traction, wheelie behaviour |
| `FAULT_ADC1/2` | 2.0 V | matches your `fault_adc1/2 = 2` ✓ footpad gating |

The tags make replacing them a search, not a hunt.

## Faithful vs approximated

Faithful (mirrored from source, line references in `sim/control.py`): the PID (P, I with
the hidden `LOOP_HERTZ_COMPAT = 720` scale, rate-P instead of D, `ki_limit`), the
asymmetric braking gain scales, the booster's deadband/ramp/speed-stiffening,
torque→current through `TORQUE_CONSTANT_COMPAT`, the softstart ramp and current clamps.

Approximations, named so they do not quietly become truth: `balance_pitch` comes from a
complementary filter rather than the Mahony filter; `pitch_rate` is treated as deg/s;
ATR, the torque/turn/brake tilts and the state machine are simplified ports.

## Environment notes

* 4090 with working CUDA in WSL, 15 GiB WSL RAM — training will fit comfortably.
* **MuJoCo's EGL offscreen renderer is dead on this box** (`eglQueryString` is None), so
  all renders go through headless Blender 4.5.
* Never open a window: everything here runs `-b` / `-NoNewWindow` / background jobs.

## License and credits

GPL-3.0, see [LICENSE](LICENSE). The controller and protections in `sim/` are ported from
[Refloat](https://github.com/lukash/refloat) (GPL-3.0), and `sim/firmware_params.py` is generated
from its `settings.xml`. This is an independent research project, not affiliated with Floatwheel,
and nothing here has been validated on hardware.

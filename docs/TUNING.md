# Tuning the Floatwheel: what the sim says

Two things live here: a gain sweep of the balance controller on the firmware's own
parameter names, and what PPO learned about riding the board. Numbers come from runs on
this box; the command that produced each block is under it, so nothing here is folklore.

## 1. The gain sweep (`kp` × `kp2`)

    python -m tools.tune_sweep --what pid --grid kp=8,20,35,50 kp2=0.0,0.3,0.6,1.2 \
        --seeds 3 --scenario hold

Hold-upright, one 300-700 N shove mid-episode, averaged over 3 seeds. `recover_s` is
timed from the peak of the reaction, not from the instant of the shove (an impulse
changes rate before it changes angle, so timing from the shove measured 0.0 s for
everything and looked like a working metric).

| kp | kp2 | fall% | max&#124;pitch&#124; ° | recover_s | speed m/s | mean&#124;A&#124; |
|---:|----:|------:|--------:|----------:|----------:|-------:|
| 8 | 0.0 | **100** | 45.0 | 0.22 | 0.89 | 11.2 |
| 8 | 0.3 | 0 | 1.81 | 0.30 | 0.42 | 0.88 |
| 8 | 0.6 | 0 | 1.38 | 0.29 | 0.42 | 0.56 |
| 8 | 1.2 | 0 | 0.93 | 0.28 | 0.42 | 0.46 |
| 20 | 0.0 | **33** | 31.0 | 0.16 | 0.46 | 24.2 |
| 20 | 0.3 | 0 | 1.16 | 0.16 | 0.42 | 1.09 |
| 20 | 0.6 | 0 | 0.97 | 0.15 | 0.42 | 0.66 |
| 20 | 1.2 | 0 | 0.73 | 0.13 | 0.42 | 0.50 |
| 35 | 0.0 | 0 | 13.3 | 0.12 | 1.16 | 28.9 |
| 35 | 0.3 | 0 | 0.90 | 0.11 | 0.42 | 1.42 |
| 35 | 0.6 | 0 | 0.79 | 0.10 | 0.42 | 0.78 |
| 35 | 1.2 | 0 | 0.62 | 0.08 | 0.42 | 0.54 |
| 50 | 0.0 | 0 | 9.20 | 0.09 | 0.64 | 30.0 |
| 50 | 0.3 | 0 | 0.77 | 0.09 | 0.42 | 1.80 |
| 50 | 0.6 | 0 | 0.68 | 0.08 | 0.42 | 0.90 |
| 50 | 1.2 | 0 | **0.56** | **0.06** | 0.42 | 0.59 |

What it says:

* **`kp2` is the stability, `kp` is the stiffness.** With `kp2 = 0` — the firmware's
  rate-P term switched off — the board falls over at `kp = 8` and fails a third of the
  time at `kp = 20`, while drawing 11-30 A doing so. Every setting with `kp2 >= 0.3`
  holds. Refloat uses rate-P instead of a D term ("works better with high Mahony KP",
  `pid.c:51-89`), and this is what that trade looks like: the rate term is not
  refinement, it is the thing keeping you on the board.
* **Recovery time is monotone in both gains**: 0.30 s at `kp 8 / kp2 0.3` down to 0.06 s
  at `kp 50 / kp2 1.2`. Same for max angle under the shove (1.81° → 0.56°).
* **The cost is amps, and it is steep.** At `kp2 = 1.2`, going from `kp 8` to `kp 50`
  multiplies mean |current| by 1.3; going to `kp 50` at `kp2 = 0` costs **30 A** —
  60× the economical setting, for a board that is already holding. Heat and battery for
  a feel difference you can measure but probably not feel.
* **The firmware's own defaults (`kp 20`, `kp2 0.6`) sit at a sensible place**: 0.97°,
  0.15 s, 0.66 A. Stiffer (`kp 35`, `kp2 1.2`) buys 0.62°/0.08 s at 0.54 A — no reason
  not to, on this model.
* `speed_mps` is 0.42 in almost every row because the board creeps: with a level
  setpoint and an integral term the wheel finds a small steady speed. The two outliers
  (0.89, 1.16) are rows where it fell over and was still accelerating when it hit 45°.

Two things this sweep cannot tell you: what it *feels* like under a real rider (the
model's rider is a rigid 78 kg mass at 0.95 m), and anything about tiltbacks, which are
not in the hold scenario.

## 2. What PPO learned

`tools/train_ppo.py`, three runs, on the same env: PPO with an MLP `[64, 64]`, 100 Hz
control, 1 action. The `setpoint` runs sit **on top of** the firmware PID (the policy
moves the setpoint, the PID still closes the loop — which is the shape worth having,
because what you trained stays comparable to what Refloat already does on the board).
The `torque` run has no PID under it: the network *is* the balance controller.

Measured on this box, 2026-11-15, from `runs/*/report.json` (12 eval episodes each, shoves
every ~4 s; `pid_baseline` is the same env with a zero action, i.e. the firmware PID alone):

| run | policy shape | return | fall% | max&#124;pitch&#124; ° | mean&#124;A&#124; |
| --- | --- | ---: | ---: | ---: | ---: |
| `ppo_balance_setpoint` | PPO moves the setpoint, firmware PID closes the loop | 1343.3 | 0 | 1.27 | 2.57 |
| `ppo_balance_setpoint` | same, no shoves | 1344.1 | 0 | 0.37 | 1.18 |
| — | **firmware PID alone** (the bar) | **1348.6** | **0** | **1.10** | **2.45** |
| `ppo_ride_setpoint` | PPO chases a speed target, PID under it | **1547.7** | 0 | 0.98 | 2.18 |
| `ppo_ride_setpoint` | same, no shoves | 1582.5 | 0 | 0.54 | 1.61 |
| — | firmware PID on the ride scene | 1527.5 | 0 | 1.12 | 1.57 |
| `ppo_balance_torque` | **PPO is the balance controller** — no PID | 1346.6 | 0 | 1.15 | **1.83** |
| — | hand PD through the same action map (kp −1.4, kd −0.02) | 1344.6 | 0 | 1.5 | — |

Read it honestly:

* **The setpoint shape works.** A trained policy steering the firmware's own controller
  matches the PID on balance (1343 vs 1349) and *beats* it on the ride scene (1548 vs 1528)
  while holding a tighter angle (0.98° vs 1.12°). That is the useful result, and it is the
  shape that transfers to hardware: what it changed is the setpoint, a thing VESC Tool
  already has a concept of.
* **The pure-torque shape works now (NEXT 347/348, 400k steps, R78).** It did not work
  before, and the reason was a bug, not the physics: `behaviour_clone` regressed the wrong
  element of `policy(obs)` — the latent, not the action — so the warm start silently did
  nothing and the policy fell in 100% at 45.1°. With the clone fixed to fit the emitted
  action mean (corr 0.9997 to the hand PD) and the device pinned to CPU (measured 2299
  steps/s at 174 s for 400k), the torque policy now holds all 12 eval episodes at 1.15°,
  mean |current| 1.83 A — the tightest angle on the board and a third less current than the
  PID. It does not yet beat the PID's 1.10° return-for-return, but it is no longer "does not
  pay"; on this plant the learned-torque shape works, and the hand PD's 1001-step hold at
  1.5° through the same action map is what told us it always should have.

### Things that were true before the numbers were in, and stayed true

* **A policy that thrashes is not a policy.** The first version had no action-rate
  penalty and the full ±28 N·m range behind one action. It fell in 100% of episodes at
  `std 1.01`: PPO never stopped exploring because thrashing paid as much as anything.
  Adding `0.08 (u_t − u_{t−1})²` and scaling the action so full stick is 8.4 N·m took
  `std` to 0.34 and the return up 8× in the same number of steps.
* **The horizon, not the network, was the wall.** At a 100 Hz control rate `gamma =
  0.99` is a one-second horizon. Recovery from a shove happens over 0.5-1.5 s, so a
  0.99 policy literally cannot be credited for catching itself. `gamma = 0.998`
  (≈5 s) plus longer rollouts is the change that mattered most.
* **Training is cheap here**: 200k-400k steps in 3-6 minutes on one 4090 box, ~1200
  steps/s single-threaded per env set. Enough runs to matter, not enough to need a
  cluster.
* **Judge with mean |current|, not just return.** A policy that freezes dead upright in
  the middle of the tolerance band scores well and rides terribly. `evaluate_policy`
  reports amps beside return for exactly that reason.

## 3. Robustness of the learned policy

    python -m tools.tune_sweep --what robust --policy runs/ppo_balance_setpoint \
        --grid TORQUE_MAX=18,28,40 mahony_kp=1,2,4

This asks the question that matters before you trust a trained policy: it was trained at
one torque limit and one filter constant. When you turn those knobs on the real board —
which you can, from VESC Tool — does the behaviour survive, or did it memorise the plant?

Not folklore — run 2026-11-15 (NEXT 346), 8 eval episodes per setting with shoves, from
`runs/sweep_robust_setpoint.json`. Getting here took three fixes, because this branch had
never once completed: `spec.set()` tried to mutate a frozen `Value` (crash), the branch
hardcoded `act_on="torque"` (it would have fed a *setpoint* policy through the torque map —
wrong controller, plausible-looking numbers), and the rows were indexed by their print heads
instead of `evaluate_policy`'s keys (`KeyError: 'fall%'`).

| TORQUE_MAX | mahony_kp | return | fall% | max&#124;pitch&#124; ° | mean&#124;A&#124; |
|---:|---:|---:|---:|---:|---:|
| 18 | 1 | 1340.8 | **0** | 1.73 | 3.60 |
| 18 | 2 | 1341.1 | **0** | 1.70 | 3.02 |
| 18 | 4 | 1341.2 | **0** | 1.69 | 2.71 |
| 28 (spec) | 1 | 1343.3 | 0 | 1.30 | 2.93 |
| 28 (spec) | 2 | 1343.4 | 0 | 1.29 | 2.57 |
| 28 (spec) | 4 | 1343.5 | 0 | 1.29 | 2.28 |
| 40 | 1 | 1343.3 | 0 | 1.30 | 2.93 |
| 40 | 2 | 1343.4 | 0 | 1.29 | 2.57 |
| 40 | 4 | 1343.5 | 0 | 1.29 | 2.28 |

What it says:

* **The policy did not memorise the plant.** Every knob combination holds: 0 falls across
  all nine. Turning the torque limit down 36% (28→18 N·m) costs a 0.4° lean margin and —
  tellingly — *more* current, because the policy keeps pushing against a tighter ceiling.
* **It does not rely on headroom it shouldn't.** `TORQUE_MAX=40` is identical to 28 to the
  decimal: the policy never commands near the limit, so raising it changes nothing. A
  policy that *had* memorised 28 would show a jump at 40.
* **`mahony_kp` behaves like filter cleanliness, not a cliff.** 1→4 trims mean current
  ~18% (2.93→2.28 A) with the same holding angle: a cleaner angle estimate costs fewer
  corrections. Monotone and harmless at every value tested.
* **Cross-check:** the `28 / 2` row reproduces the PPO table in §2 (1343.4 vs 1343.3, 2.57 A
  exactly) — sweep and training report measure the same thing the same way, which was the
  point of one harness.

The honest answer to "does the trained policy survive you turning the knobs" is now **yes,
for the two knobs this grid turns** — and untested for the rest, including the two
things a sweep cannot see from here: what it *feels* like under a real rider (§1's rigid
78 kg rider stands), and tiltbacks, which `sim/control.py` does not model at all — NEXT 349
owns porting them so a future sweep can drive into each limit.

### The same grid on the pure-torque policy — it fails differently

`runs/sweep_robust_torque.json`, same command with `--policy runs/ppo_balance_torque`
(8 episodes per cell, `act_on=torque` picked up from the run's own tag, 28 s):

| TORQUE_MAX | mahony_kp | return | fall% | max&#124;pitch&#124; ° | mean&#124;A&#124; |
|---:|---:|---:|---:|---:|---:|
| 18 | 1 | 1338.7 | **0** | 1.94 | 2.22 |
| 18 | 2 | 1339.4 | **0** | 1.92 | 2.15 |
| 18 | 4 | 1339.9 | **0** | 1.88 | 2.11 |
| 28 (spec) | 1 | 1345.9 | 0 | 1.24 | 1.97 |
| 28 (spec) | 2 | 1346.4 | 0 | 1.20 | 1.90 |
| 28 (spec) | 4 | 1346.7 | 0 | 1.18 | 1.88 |
| 40 | 1 | 1347.8 | 0 | 0.94 | 1.88 |
| 40 | 2 | 1348.2 | 0 | 0.91 | 1.82 |
| 40 | 4 | 1348.4 | 0 | 0.89 | 1.80 |

Again 0% falls across all nine. But where the setpoint policy was *flat* above its training
value, this one is not: 28 and 40 differ to the touch (1.20° → 0.91°), and 18 costs it a
whole degree more lean than it was trained to need.

* **The torque limit is a tuning knob for this shape, not a safety margin.** It was trained
  to spend 28 N·m, so it spends whatever the plant actually allows: give it 40 and it
  balances tighter, withhold to 18 and it holds looser. On hardware that means a learned
  torque controller has to be re-checked — not just "still working" — every time the torque
  limit moves, because moving it moves the controller's gain. The setpoint shape has no
  such coupling, which is a point in its favour beyond transferability.
* **Do not read "1.94° at 18 N·m" as "it can hold 1.94°".** The static argument puts the
  steady lean available at 18 N·m near 1.2° (the 28 N·m figure is 2.04°). These episodes
  are impulses — the board swings *through* 1.9° and catches itself; nobody asked it to park
  there. Standing still at a lean is the tiltback test, i.e. NEXT 349.
* **And the `mahony_kp` caution above, now paid (NEXT 351) — and it resolved LOUD.**
  `tools/noise_curve.sh` re-ran the mahony_kp walk (0.5→8) at fixed `TORQUE_MAX`, for both
  policies, at two defensible IMU noise levels (σ 0.5° rider-normal, 1.0° rough-road;
  `runs/sweep_robust351_*.json`). Two findings, one of them bigger than the question asked:

  **1. The curve is flat, and structurally so.** Torque-policy mean |A| at σ=0.5°: 11.96 →
  11.97 across a sixteen-fold filter-gain range. The setpoint policy does show a real
  minimum — 26.93 A at kp 4, rising again at kp 8 — but it is ~5% deep and buried under the
  second finding. The flatness is a sim fact, not a physics fact: `sim/env.py` injects the
  noise *into the observation vector, after the Mahony filter has already run*, so the
  injected noise never meets the gain it was being priced against. This grid cannot see the
  cost side at any value. Fix filed as **NEXT 352** (sensor noise upstream of the filter,
  obs read downstream).

  **2. Neither saved policy was ever trained to see noise, and it shows on the amp gauge.**
  Same weights, same shoves, noise added at eval: torque-policy mean |current| goes
  **1.90 → 11.96 A** (σ 0.5°) → 20.3 A (σ 1.0°); the setpoint policy **2.57 → 24–28 A** —
  a 6–10× bill for a board that still does not fall (0% throughout, max |pitch| 1.5–2.9°).
  That is precisely the thrash this doc condemns under "judge with mean |current|": a policy
  burning amps to react to inputs it never saw in training. On hardware that is heat and a
  dead battery, and it says the training recipe, not the filter gain, is what to fix first —
  `FloatwheelEnv(noise_deg=…)` exists and the saved policies never used it.

  So §3's advice changes: "raise `mahony_kp`, save 18–22% current" is **withdrawn as
  hardware advice** — it is what a noise-free sim prints, nothing more. Until NEXT 352 puts
  filter noise where a filter can actually fight it, and the policies are retrained with
  noise in the training envs, no amp figure here survives contact with a real IMU.
  Curves caveat: cells are paired by construction (every sweep from env seed 999, one noise
  realisation), so cell-to-cell comparisons are clean and absolute values are one draw.

## 4. The protections are in the sim now — and one number from the first version of this section had to be taken back

`sim/control.py` used to mirror three things (PID, booster, current clamps). With NEXT 349 it
mirrors eleven more, all of them *setpoint* modifiers, in `sim/protections.py`: the duty, HV, LV
and over-speed tiltbacks, constant and variable noseangling, torque tilt, turn tilt, brake
tilt, ATR, and `MODE_HANDTEST`'s disable list. They are **off by default**, so the tables above
describe the same controller with or without them. The count is printed by
`python -m tools.probe_protections.py` (`behaviours mirrored: 11`) off
`sim/protections.py`'s `MIRRORED_BEHAVIOURS`, and this sentence said *ten* until the ATR row
closed the row — which is precisely why the number lives in code now. Full source references, the pin list and
both sweeps live in `docs/TILTBACK_PORT.md`; this section is only what a person tuning a real
board should take from it.

Read this section after NEXT 354, not before. Porting the firmware's setpoint filter exposed a
mistune that had been in the plant from the start — filter alphas configured at `CONTROL_HZ`
(100 Hz, a tagged `estimate`) but stepped at MuJoCo's 1 kHz timestep — and it changed three
things worth stating here rather than burying in a changelog: two findings I had published in
this section are **retracted** (below), the §1 firmware-PID numbers moved modestly (defaults
hold 0.92 → 0.98° and 0.64 → 0.71 A, recovery 0.15 s unchanged, `kp2=0` launch falls at 2.6 s
rather than 5.9 s — the ordering and every fall% are unchanged, current columns in
`runs/sweep_pid_hold.json`), and the learned-policy tables in §2/§3 **did not move at all**,
because a pure-torque policy never passes through the PID's booster or current EMA.

* **Read a tiltback angle against your cruising speed, not against a standing board.** The
  authority line this doc opens with (`28 N·m vs 787 N·m of gravity → can hold 2.04°`) is a
  *stationary* fact. In motion the plant sustains far more lean than that: the torque-hold
  scenario rides at 0% falls with max |pitch| 7.6–8.3° while accelerating, because
  acceleration tilts effective gravity and the wheel can lean under the rider. So a
  `torquetilt_angle_limit` of 6–8°, which looks lethal next to the 2.04° line, is not
  automatically nonsense — and conversely a tiltback that looks mild at 1° can matter,
  depending where the board is when it fires. Judge these knobs in a scenario with speed in it.
* **RETRACTED — the over-speed tiltback does not fall the rider at 1.0 m/s.** The bullet that
  used to sit here said `tiltback_speed=1.0` drops the rider 100% of the time while 1.5 and
  2.0 drop 0%, and it was labelled a question rather than advice. It was worse than a question:
  it was an artefact. NEXT 354 ported the firmware's real setpoint filter and the whole cell
  came back at **0% falls at every threshold**, with the feature simply trimming cruising speed
  2.47 → 1.11 m/s (`tiltback_duty_angle=1.5`, 3 seeds). The cause was that the protection
  filters were configured at `CONTROL_HZ` (100 Hz, a tagged `estimate`) while the plant steps
  them at MuJoCo's 1 kHz timestep — ten times too little smoothing per second, thrashing the
  setpoint. `tiltback_speed` is a normal knob: set it where you want the board to start easing
  off, not where a bug said to fear it. Full write-up and both re-measured tables:
  `docs/TILTBACK_PORT.md`, section "NEXT 354".
* **RETRACTED — brake tilt is not a 0.13° nothing.** The earlier claim ("a ~26 ms transient,
  hardened demand does not move it") was the same bug viewed from the other side: a filter
  that never reached its target before the gate closed. With the rates derived from the real
  step it peaks near **3.7°** in the plant, which matches what the firmware's own comment
  promises — a "delayed lingering nose lift" you can feel. `braketilt_lingering` is the knob
  for how long it lingers: it divides the release speed (`brake_tilt.c:45`), so raising it
  stretches the lift out rather than making it bigger.
* **Three firmware quirks worth knowing before you tune by feel.** Zeroing `turntilt_strength`
  (which is what VESC Tool's hand test does) *freezes* turn tilt wherever it was rather than
  releasing it — `turn_tilt.c:89-91` returns before the filter, so a board tested that way can
  hold a lean it appears to have switched off. Lean *with* the brake and brake tilt keeps its
  previous target instead of dropping (`brake_tilt.c:78` has no `else`). And torque tilt and
  brake tilt do not add: same sign, the larger one wins (`main.c:947-956`), so turning both up
  buys less than the two numbers suggest.
* **The two battery thresholds in VESC Tool are per-cell, not pack volts — and the sim now says
  so out loud.** Refloat ships `tiltback_lv=3.0` and `tiltback_hv=4.3`, and the firmware
  multiplies each by VESC's cell count **only when the number you typed is under 10**
  (`motor_data.c:81-88`). On a 12S pack that means 3.0 → 36.0 V and 4.3 → 51.6 V, which is the
  sensible reading. The trap is the boundary: type an absolute figure below 10 V — 9.0 on a 12S
  board, say — and the firmware reads it as per-cell, multiplies to 108 V, and your board leans
  back for the entire ride. Nothing in VESC Tool warns you; the number is either "small enough
  to be per-cell" or "big enough to be the pack", decided by one comparison. The sim had this
  backwards for a release (it compared 4.3 against pack volts, so a healthy 42 V rail sat in HV
  tiltback forever) and the pins now pin both sides of the `< 10` rule, including the case where
  the same 30 V pack is *low* voltage on a 12S board and *high* voltage on a 6S one.
* **How to check any of this yourself:** `python -m sim.protections` (63 pins, including the
  filter's alpha, its on/off asymmetry and its zero-crossing rule) and
  `python -m tools.probe_protections.py` (each behaviour driven to its limit, `--sweep` for
  its own knob) both exit non-zero on failure; `tools/tune_sweep.py --protections --scenario
  speed_ramp|duty_run|torque_hold|turn|brownout|hard_brake` gains a peak-angle column per
  behaviour, which is how you tell a tiltback from a bad gain when fall% moves.

## Fleet arm-states, measured on this box (NEXT 367, 2026-11-16)

Same policy, same seed, same 20k-step budget, four envs, judged on 12 eval episodes by
`tools.train_ppo.evaluate_policy` — the only difference is whether `sim/protections.py` is stepped and
what it is armed with. Produced with the knobs this row added: `--protections`, `--arm K=V` (repeatable),
and `--what policy --protections` in `tools/tune_sweep.py`. The armed runs use the over-speed tiltback at
`tiltback_speed=2.0`, `tiltback_speed_angle=15.0`; strengths ship at 0.0, so an unarmed `--protections`
run steps a fleet that cannot move the board (the tool now says so).

| act_on | scene | fleet | fall% | max\|pitch\| | mean\|A\| | return |
| --- | --- | --- | --- | --- | --- | --- |
| torque | balance | never stepped | 0 | 1.42° | 2.60 | 1344.6 |
| torque | balance | armed, over-speed 2.0/15° | 0 | 1.42° | 2.60 | 1344.6 |
| torque | ride | never stepped | 0 | 1.33° | 2.77 | 1583.1 |
| torque | ride | armed, over-speed 2.0/15° | 0 | 1.33° | 2.77 | 1583.1 |
| setpoint | balance | never stepped | 0 | 1.33° | 3.18 | 1332.7 |
| setpoint | balance | armed, over-speed 2.0/15° | 0 | **3.39°** | **22.78** | 1265.8 |
| setpoint | balance, no shoves | never stepped | 0 | 0.54° | 1.73 | 1334.6 |
| setpoint | balance, no shoves | armed | 0 | 0.38° | 1.21 | 1343.6 |
| PID baseline (zero action) | balance | never stepped | 0 | 1.11° | 1.90 | 1349.2 |
| PID baseline (zero action) | balance | armed | 0 | **3.87°** | **22.49** | 1263.8 |

**What the top four rows mean: nothing changed, because a pure-torque policy cannot meet the fleet.** In
`act_on="torque"` the policy *is* the balance controller, `BalanceController.update()` is never called,
and the entire protection fleet lives inside it — so arming it changes no byte of the result in either
scene, at any arm value. `report.json` now carries `protections_inert_in_torque: true` and the tool prints
a warning, because that equality is exactly the kind of result that gets written up as "the policy is
robust to tiltbacks". It isn't; it never saw one. This is the NEXT 348 shape, so its bar is currently
measured on a board whose safety fleet is out of the loop *structurally*, not just by a flag.

**What the bottom six mean: in the mode the firmware actually runs (setpoint), the tiltback is not
cosmetic.** Seven times the current (3.18 → 22.78 A) and two and a half times the pitch (1.33° → 3.39°)
for the same trained policy at the same seed, and the PID baseline moves the same way (1.90 → 22.49 A),
which is the sanity check: the policy and the firmware react to the same arm in the same direction. With
shoves off (the calm rows) the arm costs *less* current, not more — the cost is specifically the riding
regime where the board crosses the arm speed, not the standing one.

**Nobody falls in any of these**, so the 0%-falls bar of NEXT 348 does not discriminate between a board
that fights back and one that does not. The number that does is mean|A| — which is the column NEXT 347's
whole "rides at 0.24 A, i.e. no torque" bug was about. A tiltback that a policy answers with 7× the
current is doing something; a policy that never notices it is being scored on a different board.

Caveat that stays until NEXT 350: `TORQUE_MAX`, `WHEEL_RADIUS`, `RIDER_MASS`/`COM_HEIGHT`, `MU_SLIDING`
and the erpm/duty constants are estimates, so *which* arm value is realistic is not known — the rows
above are about the ratio between boards, not about the real board.

## NEXT 348's bar on both boards, at a real budget (measured 2026-11-16, 100k steps, setpoint)

`python -m tools.train_ppo --timesteps 100000 --envs 4 --act-on setpoint [--protections --arm
tiltback_speed=2.0 --arm tiltback_speed_angle=15]`, judged on 12 eval episodes. The bar NEXT 348 set
against the hand PD: **0% falls, max|pitch| under 2°, mean|A| under 5**.

| board | fall% | max\|pitch\| | mean\|A\| | return | bar |
| --- | --- | --- | --- | --- | --- |
| fleet never stepped | 0 | **1.12°** | **1.63** | 1349.2 | **met** on all three |
| armed, over-speed tiltback 2.0 m/s / 15° | 0 | **2.44°** | **18.99** | 1297.6 | **missed** on two of three |
| fleet never stepped, no shoves | 0 | 0.26° | 0.17 | 1351.1 | met |
| armed, no shoves | 0 | 0.34° | 0.96 | 1346.5 | met |

The policy does not fall either way, so fall rate is not the discriminator — current is, by an order of
magnitude, and pitch crosses the 2° line. The calm rows say the same thing from the other side: with no
shoves the armed board still passes, so what costs is *riding into the arm*, not holding a standstill.

Verdict for the row: the pure-setpoint shape is **not retired** — it meets the hand PD's bar on the board
the firmware runs with nothing armed, and misses it on a board with an over-speed tiltback set to bite at
2 m/s. Whether that is a failure of the policy or of the arm value is NEXT 350's question: 2 m/s and 15°
are my guesses, not the real board, and `TORQUE_MAX`/`WHEEL_RADIUS`/`RIDER_MASS`/`COM_HEIGHT` are all still
`estimate` in `sim/spec.py`. A policy that answers a tiltback with 19 A is doing something; whether the
board it did it on is the one the rider rides is unanswerable until 350 lands. Any 348 number quoted from now
on has to name its board — which is why `report.json` carries `protections`/`armed`.

# NEXT 349 — port spec: tiltbacks, noseangling and ATR

Read off the Refloat source the real board actually runs (local clone
`refloat`, v1.3.1 @ `a2b55d9`), not off the VESC
Tool UI. Every line reference below was opened and read; nothing here is remembered.

## The finding that shapes the whole port

**All eight behaviours are setpoint modifiers.** None of them touch current or torque
directly; each computes an *angle* that becomes the balance controller's setpoint:

```c
d->setpoint_target = d->float_conf.tiltback_duty_angle;      // main.c:512
d->setpoint_target = -d->float_conf.tiltback_hv_angle;       // main.c:542
noseangling_target += d->float_conf.tiltback_constant * d->motor.erpm_sign;   // main.c:687
```

Two consequences, both load-bearing for this sim:

1. Port them into the **setpoint path** (`sim/control.py`) and the firmware PID *and* the
   setpoint-trained policy both inherit them for free — which is also the answer to
   "how do we test a learned policy against the board's safety behaviour": the safety
   behaviour lives below the policy, in code we already mirror.
2. The board tilts the rider by **moving the target the controller chases**, so a tiltback
   is indistinguishable from the rider leaning, as far as the balance loop is concerned.
   That is why tiltbacks feel like nothing until you are already off the pad: the loop is
   told to be there. A sim that instead applied a force during tiltback would get the
   *feel* wrong and would fool anyone checking with the eye test.

## What to port, with the exact source

| behaviour | source | what it computes | inputs the sim must supply |
| --- | --- | --- | --- |
| speed tiltback | `main.c:637-642` | `\|speed\| > tiltback_speed` → setpoint `±tiltback_duty_angle` (**reuses the *duty* angle, not a speed angle** — a real quirk, mirror it, do not "fix" it) | forward speed ✓ have it |
| duty tiltback | `main.c:510-514` | `duty > tiltback_duty` → `±tiltback_duty_angle` | duty cycle — derive from `\|erpm\|`/max, or inject |
| HV tiltback | `main.c:538-557` | volt rising past threshold, **500 ms debounce** ("500ms have passed or voltage is another volt higher, time for some tiltback") → `±tiltback_hv_angle` | battery voltage → **inject as a scenario input**, do not model a battery |
| LV tiltback | `main.c:561-635` | volt sagging → `±tiltback_lv_angle`, and note the deliberate asymmetry: *"Use the angle from Low-Voltage tiltback, but slower speed from High-Voltage tiltback"* (lines 561, 576, 593) | voltage, same as above |
| constant noseangling | `main.c:686-687` | above `tiltback_constant_erpm`, add `tiltback_constant × erpm_sign` | erpm ✓ have it |
| variable noseangling | `main.c:191-195, 682-684` | `tiltback_variable = variable/1000 × sign(max)`; ramps over `\|erpm\| − variable_erpm` up to `tiltback_variable_max_erpm` | erpm ✓ |
| torque tilt | `torque_tilt.c:60-67` | `(|torque| − start_current × TORQUE_CONSTANT_COMPAT) × strength`, clamped to `torquetilt_angle_limit`, sign of torque; **regen has its own strength** (`:60`) | motor current/torque ✓ have it (`controller.current`) |
| turn tilt | `turn_tilt.c:89-132` | needs yaw **rate** *and* yaw **aggregate**; gate on `turntilt_start_angle`/`start_erpm`, boost below `erpm_boost_end` (`:112-115`), aggregate up to double via `aggregate_damper` (`:119-124`), clamp to `angle_limit` (`:129`) | yaw rate exists (free joint dof 4); **a straight-running 1-D sim never develops yaw** → inject a yaw-rate profile |
| brake tilt | `brake_tilt.c:38-45` | `factor = −(0.5 + (20 − strength)/5)`, lingering scaled by `braketilt_lingering` against `off_speed_limit` | braking flag — from `motor.braking`, i.e. negative current demand |
| input tilt | `remote.c:70-149` | `target = input × inputtilt_angle_limit` with `inputtilt_deadband` and optional throttle inversion; remote types other than PPM are refused (`main.c:1251`) | **ported in NEXT 365** — `run_case(rider_input=…)` is the channel, so no stick has to exist for the
    stage to be exercised; it stays inert while `inputtilt_remote_type` is NONE, and a pin keeps that
    default honest |

Ramp rates matter as much as targets: every tiltback carries its own `*_speed` (deg/s) and
a separate `tiltback_return_speed` for coming back (`main.c:273-286` picks which speed
applies per state). Porting the target without the rate gives a board that snaps the rider
off the pads in one control period — a behaviour no real Floatwheel has.

## The coupling nobody would notice, and ATR is not "just a speed cap"

When ATR is active, the firmware **zeros the other protections**:

```c
d->float_conf.torquetilt_strength = 0;          // main.c:1489
d->float_conf.torquetilt_strength_regen = 0;    // main.c:1490
d->float_conf.turntilt_strength = 0;            // main.c:1493
d->float_conf.tiltback_constant = 0;            // main.c:1494
d->float_conf.tiltback_variable = 0;            // main.c:1495   (and again at 1941-1953)
```

So an ATR mode that only limits speed while leaving tiltbacks live is **wrong in a way that
no single-behaviour test would catch**. `sim/control.py` must implement ATR (`atr.c`:
`atr_init` :25, `atr_reset` :37, `atr_configure` :47, `atr_update` :74) as a *mode* that
suppresses the rest, and the test for it is "with ATR on, do torque tilt and tiltback stop
firing", not "does ATR slow the board down".

## How to test each one (the part that makes the sweep worth running)

Each behaviour needs a scenario that actually drives into it — `tools/tune_sweep.py`
grows a `--scenario` per limit, since `hold` and `launch` will never reach most of them:

* `speed_ramp` — accelerate to past `tiltback_speed`, assert the setpoint ramps at
  `tiltback_duty_speed` deg/s and the rider-angle target matches `tiltback_duty_angle`.
* `sustained_push` — hold current above `torquetilt_start_current × TORQUE_CONSTANT_COMPAT`
  and check the lean ramps to `torquetilt_angle_limit`, and that the **regen** strength is
  used when the motor is braking (`torque_tilt.c:60`).
* `turn` — inject a yaw-rate profile; assert turn tilt needs both the aggregate and the
  rate (`turn_tilt.c:104`: below `turntilt_start_angle` *or* below 30°/s, nothing happens),
  which is the detail a naive port always gets wrong.
* `brownout` — inject a falling `battery_volt` and assert the 500 ms debounce, plus the LV
  angle with HV ramp speed.
* `atr_on_off` — same scenario, ATR enabled and disabled; the assertion is the *absence* of
  the other behaviours.

Then the question the row actually asks — *which setting makes this board unrideable* —
becomes a grid over `tiltback_duty_angle`, `tiltback_*_speed`, `torquetilt_strength` and
`turntilt_strength`, scored on how much of the tolerance band the rider is asked to absorb
and how fast.

## ERRATUM (2026-11-15) — "ATR zeroes the other protections" is wrong

I opened the two call sites above before implementing them. **Neither is ATR.** Both are
*mode* entry points, and ATR gets zeroed along with everything else:

* `main.c:1481-1497` is inside `cmd_handtest()` — `d->state.mode = cfg[0] ? MODE_HANDTEST :
  MODE_NORMAL`, then the block zeros `ki`, `kp_brake`, `kp2_brake`, both booster angles,
  `torquetilt_strength(_regen)`, **`atr_strength_up`/`atr_strength_down`**, `turntilt_strength`,
  `tiltback_constant`, `tiltback_variable`. That is the firmware's "hand test" tuning mode.
* `main.c:1935-1953` is the same list inside `flywheel_start()` (`MODE_FLYWHEEL`), plus
  `brake_current = 0`, darkride/reversestop off.

So ATR is **not** a suppressor mode: `atr_update()` is called unconditionally in the running
state (`main.c:933`) and contributes one more setpoint term; it is switched off by its own
`atr_strength_up/down = 0`, and it is what *gets* switched off by handtest/flywheel. The port
must therefore implement:

1. ATR as a plain setpoint contributor (`atr.c:74-162`, gated by `atr_strength_up/down`).
2. A `mode` knob mirroring `cmd_handtest`/`flywheel_start` — *that* is the "which protections
   go to zero and in what order" behaviour, and its test is still "with handtest on, does
   torque tilt stop firing".
3. The test `atr_on_off` stays useful but asserts something different from what was written
   above: same scenario, ATR strengths non-zero vs zero, and the assertion is the *presence /
   absence of the ATR term alone* — other tilts must be unaffected either way.

Everything else in this spec verified out of the same clone (`refloat @ a2b55d9`): the
setpoint combination and its torque/ATR+brake aggregation at `main.c:912-956`, the per-state
ramp speeds at `main.c:270-290`, the duty/HV/LV/speed chain at `main.c:510-650`,
`rate_limitf` (`lib/utils.c`, saturating), `sign(x) = (x<0)?-1:1` (`lib/utils.h:99`),
`LOOP_HERTZ_COMPAT(_RECIP)` (`lib/utils.h:34-35`), `ema_calculate_alpha_time_constant(tc,f) =
ema_calculate_alpha(1/(2*pi*tc), f)` (`filters/ema.c:33-35`), and `motor.braking =
current < 0`, `erpm_sign = sign(erpm)`, 1 Hz duty EMA, 8 Hz acceleration SMA
(`motor_data.c:128-152`).

**Still missing before the port compiles into something honest:** the settings.xml *defaults*
for the filter fields these ports need — `atr.filter.{time_constant,on_speed_time_constant,
off_speed_time_constant,on_speed_limit,off_speed_limit}`, `atr.transition_boost`,
`torque_tilt.filter.*`, `turn_tilt.filter.time_constant`, and `battery_cells` (LV/HV
thresholds are per-cell below 10 V and get multiplied, `motor_data.c:79-91`). The
`<cDefine>` names are listed in `src/conf/settings.xml` (lines 224-303, 1932-1975,
2173-2238, 2261-2310); `tools/refloat_defaults.py` did not extract them, so
`sim/firmware_params.py` has no entry for them. Read them out of settings.xml's `<default>`
elements, don't guess.

## Scope decisions, stated so they are not mistakes later

* Battery voltage and duty are **inputs**, not models. Simulating a battery to test a
  voltage-triggered lean answers a question nobody asked.
* ~~`inputtilt` stays out: it needs a rider throttle channel this sim does not have.~~ **Ported in
  NEXT 365** — the channel is an injected scenario input (`run_case(rider_input=…)`), so the stage no
  longer needs a stick to exist, only someone to hold one.
* Sign conventions get their own test before anything else. These functions all multiply by
  `motor.erpm_sign` or negate an angle; in this model a flipped sign produces a tiltback
  that throws the rider the wrong way and looks completely plausible in a screenshot —
  the same trap the pitch-sign tests in `sim/model.py` exist to prevent.
* Nothing here is allowed to change torque/current paths: if a tiltback needs the torque
  loop to survive, the port is wrong.

## What got ported, and how it is proved (NEXT 349, `sim/protections.py`)

Eleven behaviours mirrored, up from the three the controller had (balance PID, booster,
current clamps). Each cites its source lines in the code, and each is *fired* by a closed
loop rather than asserted by reading:

`duty tiltback` · `HV tiltback` (500 ms rising-edge debounce) · `LV tiltback` (LV angle at
the HV ramp speed) · `over-speed tiltback` (reusing the duty angle) · `constant noseangling`
· `variable noseangling` · `torque tilt` (separate regen strength) · `turn tilt` (rate AND
aggregate) · `brake tilt` (three gates, lingering) · `ATR` (`accel_diff` = expected-minus-measured
acceleration over filtered current, `atr.c:74+`; the last to land, in `8698e89`, and the one this
file wrongly claimed needed a motor model) · `MODE_HANDTEST`'s disable list — with
what it spares.

Two commands, both gates rather than printouts (non-zero exit on failure):

```
python -m sim.protections                     # 27 pins: signs, ramps, gating, handtest coupling
python -m tools.probe_protections.py          # each behaviour driven into its limit in the plant
python -m tools.probe_protections.py --sweep   # ... and each behaviour's own knob
python -m tools.tune_sweep --what pid --scenario launch --protections --grid SLOPE_DEG=6.0
python -m tools.tune_sweep --what pid --scenario launch --protections --atr off --grid SLOPE_DEG=6.0
```

The last pair is this file's `atr_on_off` item: the same cells with the ATR term live and with
`atr_strength_up/down` zeroed (main.c:1484-1495), and the `atr` head is the assertion — the
presence and absence of that term alone, every other column comparable cell for cell. The
probe's `atr` row drives it into its limit on the 6° hill instead, and reports
`behaviours mirrored: 11` from `sim/protections.py`'s `MIRRORED_BEHAVIOURS` so the Measure is
counted, not remembered.

The pins are where the arithmetic is proved, because the plant cannot be trusted to prove
it: `turn tilt` needs both a rate and an aggregate (`turn_tilt.c:104`) or a board that only
goes straight never shows it; torque tilt must be silent below `torquetilt_start_current`
and use its **regen** strength on the other side; every tiltback must ramp (first tick
< 0.01°) and return at `tiltback_return_speed`. The sign pin is deliberately
convention-free — it asserts the offset *opposes the physical direction of travel*, not that
it matches an erpm sign, because this model's gear may flip that sign and a tiltback that
trusts the encoder convention trusts the wrong thing.

### The over-speed tiltback is the setting that kills the rider, and only when set tight

`python -m tools.tune_sweep --what pid --scenario speed_ramp --protections --grid
tiltback_speed=0,1.0,1.5,2.0 tiltback_duty_angle=1.5 --seeds 3` → `runs/sweep_protect_speed.json`
(firmware balance PID, rider holding a steady −1.5° demand):

| `tiltback_speed` (m/s) | fall% | max\|pitch\| | ends at m/s | peak tiltback |
| --- | --- | --- | --- | --- |
| 0 (off — the firmware default) | 0% | 10.52° | 2.62 | 0.00° |
| **1.0** | **100%** | **45.11°** | 0.45 | 1.50° |
| 1.5 | 0% | 10.20° | 0.89 | 1.50° |
| 2.0 | 0% | 9.60° | 0.83 | 1.50° |

Non-monotone, and the shape is the whole story: set the over-speed threshold **below the
speed the rider is asking to ride at** and the protection fights the rider every cycle — it
levels the board, the board drops under the threshold, the tiltback releases, the −1.5°
demand accelerates it back over, and each rate-limited transition pumps the pendulum until
it goes over. Set above the cruising demand and the same feature just trims 2.62 → 0.89 m/s
and nobody falls. Same knob, same code, opposite outcomes, 0.5 m/s apart — which is the
concrete answer to "which setting makes this board unrideable", and the kind of thing a
monotone sweep would have reported as "no effect".

Treat it as a **candidate**, not advice: the firmware damps every one of these transitions
with `smooth_setpoint_update`'s on/off time constants, and this port stands in for that with
a linear rate limiter. A limit cycle that a real filter would damp is exactly what that
substitution predicts. **NEXT 354** ports the filter (the missing settings.xml defaults are
listed in the ERRATUM above) and re-measures this cell; if 1.0 m/s still falls 100%, it is
firmware behaviour, if it stops, it was my stand-in. Either way it does not get quoted to
the rider as tuning advice until then — and `RIDER_MASS`/`COM_HEIGHT`/`TORQUE_MAX` are still
estimates (NEXT 350), so the threshold where it flips will move.

Torque tilt, by contrast, does what its knob says (`--scenario torque_hold --grid
torquetilt_angle_limit=0.5,1.5,2.5` → `runs/sweep_protect_torquetilt.json`): peak angle
tracks the limit exactly (0.50/1.50/2.50°) and the cost appears in the neighbouring columns —
max\|pitch\| 7.85 → 10.41°, mean\|A\| 11.31 → 14.51 A — at 0% falls.

### Two findings the plant gave and the source did not

* **A sustained 8–10° lean is survivable at speed while a static lean caps out near 2°.**
  The torque-authority line (`28 N*m vs 787 N*m gravity → 2.04°`) is a *standing-board*
  number; once the board is accelerating, the inertial term tilts effective gravity and the
  rider can sit far further forward. So any judgement of "that tiltback angle is too
  aggressive" has to come from a scenario with speed in it — judging one from a stationary
  board condemns behaviour that is completely normal at 5 m/s.
* **Brake tilt is a transient, not a state.** In closed loop its three gates
  (`current < 0`, `|erpm| > 2000`, proportional across zero) are all satisfied only for a
  ~26 ms window when the board overshoots its own brake setpoint, so it peaks around 0.13°.
  Making the brake demand harder does not raise it (+1.5° and +3.0° both peak at 0.13 —
  checked, and that insensitivity *is* the evidence). That matches the firmware's own
  description, a "delayed lingering nose lift" (`brake_tilt.c:74`). Anyone tempted to crank
  the scenario to make the number look important should read this paragraph first. **(RETRACTED, NEXT 354: this reading came from a filter
configured at the wrong frequency, and brake tilt peaks near 3.7° once its alpha is derived from
the real step rate. Read the "NEXT 354" section below first; the paragraph stays because the
wrong reading and the reason it looked right are the useful part.)**

### Not ported, named rather than faked

~~ATR's own setpoint contributions (needs the `accel_diff` estimate over filtered current,
`atr.c:74+`, i.e. a motor model this sim lacks)~~ — **ported in `8698e89`, which also refuted the
excuse I had written here.** `accel_diff` is expected acceleration (from the torque the sim
already applies) minus measured acceleration (`d(erpm)/dt`, which the plant already measures), so
nothing was missing except my reading of `atr.c:74+`; "a half-port would be worse than none" was
an excuse, not a constraint. Its pins cover the standstill gate (`atr.c:107-117`), the sign table
at `:119-124`, silence at holding current, and the same-sign aggregation against torque tilt.
~~the `smooth_setpoint` filter behind every ramp~~ — **that one is ported too** (NEXT 354, top of
this section; its constants came out of settings.xml' `<default>` elements in `6162ef9`, not out
of my head).

Still genuinely absent, and each one changes what may be claimed: the fault state machine, which is
why nothing here can say what the board would do at a *fault* rather than at a lean limit.

~~`inputtilt` (no rider throttle channel, though the three `inputtilt_*` parameters are already in the
vocabulary and would silently look implemented)~~ — **ported in NEXT 365** (`remote.c:85-156`, landing
at `main.c:922-923`). The worry written there was the correct one: the parameters were already in the
table with nothing reading them, which is exactly how a feature looks implemented while doing nothing.
The rider channel now exists as an injected scenario input — `run_case(rider_input=...)`, the same shape
`SLOPE_DEG` uses — the stick stays inert until `inputtilt_remote_type` says one is fitted (that flag *is*
the switch; there is no `inputtilt_enabled` in this build), and a pin holds the shipped default at
exactly 0.0 so nothing measured before the port moves. Still not here, and named as absent: the
PPM/UART acquisition with its `REMOTE_TIMEOUT` (`remote.c:70-89`), and `move_speed` /
`remote_get_move_torque` (`:100-112`, `:134-146`), which is a constant-speed **torque** path on a
different branch, not a setpoint one.

~~`wheelslip` (no slip model, so the winddown paths are dead code — `SmoothSetpoint.winddown` is
implemented and nothing calls it, which is worth knowing before trusting a winddown number)~~ —
**ported in NEXT 360, with the honest half of the claim kept.** The detector is `main.c:483-509` on
the signals the firmware itself reads (the acceleration filter, erpm and the injected `duty`), and all
four winddown call sites now have callers: `torque_tilt.c:54`, `atr.c:77`, `brake_tilt.c:69`,
`turn_tilt.c:93`. What is *still* not here is the cause of a slide a rider could actually ride through —
and the sentence that used to explain why ("this plant's wheel-deck contact is rigid, so a slide is
announced by a scenario rather than emerging from a friction model") was wrong. NEXT 361 checked it:
`sim/model.py` gives the deck, the tyre and the floor Coulomb friction off `MU_SLIDING`, so the contact is
friction-limited rather than rigid, and the wheel genuinely runs away from the deck — 26 to 37 m/s of
wheel-versus-deck speed gap once µ drops below ~0.14. What it claimed about *why* has gone the same way:
those gaps belonged to runs that fell not because rigid geoms give the tyre nowhere to slide, but because
nothing had ever told the detector what duty the bridge was at (NEXT 362's missing input, NEXT 363's
measured one). The big gaps are still fall-only, for an honest reason — below µ≈0.14 the board cannot stay
up at all. And as of NEXT 363 the detector **has** latched under a rider who stayed up: a stomp onto a µ 0.16
patch at +4.34 m/s, the wheel away to +47.3 rad/s, gap 1.66 m/s, duty 0.46, flag up, run finished upright. So
in a report `wheelslip` may now be read as "the surface gave way under a rider" — still not "somebody asked
for a slide", because nothing injects it any more.
`SmoothSetpoint.winddown` went from zero callers to four, with eight pins and a field in the report.

`brake_tilt`'s downhill damper is no longer pinned at 1 either — it reads ATR's `accel_diff` and
the steep-downhill disable at `brake_tilt.c:87-90` does fire, pinned both ways: identical
braking lifts the nose -0.800° on the flat and 0.000° on a steep downhill. That behaviour existed
in the firmware all along; it was simply unreachable until ATR was ported, which is a good
argument for naming what you have not built instead of stubbing it.

## NEXT 354 — the filter is ported, two claims above are retracted, and the cause was not the filter

`sim/protections.py` now contains `SmoothSetpoint`, a line-by-line port of
`filters/smooth_setpoint.c:25-109` plus its alpha from `filters/ema.c:24-34`, used by torque
tilt, turn tilt and brake tilt exactly where the firmware uses it (`torque_tilt.c:69`,
`turn_tilt.c:139`, `brake_tilt.c:96`). Each is configured as its own `*_configure` does:
torque tilt from `torque_tilt.filter.*`; turn tilt with the speed time constants at half the
main one and all four rate limits hard-coded to 20 deg/s; brake tilt borrowing **ATR's**
filter wholesale with its release speed `atr.filter.off_speed_limit / braketilt_lingering`
(`brake_tilt.c:45` — that division is the entire function of the `lingering` parameter).

The first thing the port settled was a suspicion I had wrong in the other direction: the
**tiltbacks and noseangling do not use this filter at all**. They ramp through `rate_limitf`
(`main.c:914-918`, `main.c:690-692`) — a saturating linear limiter — so the `_ramp_toward`
this file used to call an "approximation of smooth_setpoint" was already the faithful mirror.
The docstrings now name which is which so nobody "upgrades" the wrong one.

Three behaviours turned up that a plausible port gets wrong, each now pinned:

* `turn_tilt.c:89-91` returns **before** the filter update when `turntilt_strength == 0`, so
  zeroing the strength — which is what `MODE_HANDTEST` does — **freezes** the tilt wherever it
  was instead of releasing it. The obvious port ramps it to zero, looks cleaner in a table,
  and deletes a tilt the real board is still holding.
* `brake_tilt.c:78` has no `else`: lean *with* the brake and the target stays at its previous
  value, still ramping. Returning 0.0 there releases the nose early — the one part of a brake
  the rider feels — and it is what my 349 port did.
* `main.c:947-956`: torque tilt and (ATR + brake tilt) do not add. Same sign, the larger
  magnitude wins; opposite signs, they sum. My 349 port summed them, double-counting a lift
  the board would only apply once.

### RETRACTED: "the over-speed tiltback falls the rider 100% at 1.0 m/s"

That result, published above on 2026-11-15, was an artefact. Porting the real filter surfaced
it: the filters were configured at `CONTROL_HZ` — **100 Hz, a tagged `estimate`** — while the
plant steps the controller at MuJoCo's `model.opt.timestep`, **1 kHz**. An EMA alpha is
per-step, so every one of them smoothed ten times less per second than its time constant asks
for. Deriving the rate from the `dt` actually handed to `update()` — which is what the firmware
does with `main_freq_tracker`, measuring the main loop and reconfiguring (`main.c:154-163`,
`785`) — and re-running the identical cell:

| `tiltback_speed` (m/s) | fall% | max\|pitch\| | ends at m/s | mean\|A\| | peak tiltback |
| --- | --- | --- | --- | --- | --- |
| 0 (off) | 0% | 3.24° | 2.47 | 5.53 | 0.00° |
| 1.0 | 0% | 3.24° | 1.11 | 2.78 | 1.50° |
| 1.5 | 0% | 3.24° | 1.49 | 3.70 | 1.50° |
| 2.0 | 0% | 3.24° | 2.02 | 4.72 | 1.50° |

No threshold drops the rider; the feature trims speed (2.47 → 1.11 m/s) and the board stands.
The non-monotone knife-edge I described was the mistuned filter thrashing the setpoint, not a
protection fighting a rider. `tiltback_speed=1.0` is **not** a dangerous setting, and the
earlier version of this file should be read as wrong on that point.

**RETRACTED as well:** "brake tilt is a ~26 ms transient peaking 0.13°, and hardening the
demand does not move it." Same cause — a filter that never got anywhere before its gate
closed, which I then rationalised as the feature's nature and *lowered the gate's threshold to
match*. It peaks near **3.7°** today. That half is the lesson worth keeping: a mistuned filter
reads exactly like a dead feature, and a threshold tuned down to suit a dead feature is how the
bug survives review. `tools/probe_protections.py` carries the story at the row that was wrong.

Torque tilt with the filter correct (`--scenario torque_hold --grid
torquetilt_angle_limit=0.5,1.5,2.5`, 3 seeds): peaks 0.33 / 1.10 / 1.96° against limits of
0.5 / 1.5 / 2.5°. It tracks the knob but no longer reaches it inside a 10 s episode, because
the 0.2 s time constant is real and the rider's demand reverses before the tilt finishes.
Cost side: max\|pitch\| 7.59 → 8.32°, mean\|A\| 11.18 → 11.85 A, 0% falls.

**Scope of the fix, measured rather than assumed:** the protections moved enormously (above);
the firmware PID tables moved modestly (firmware defaults hold 0.92 → **0.98°**, 0.64 →
**0.71 A**, recovery 0.15 s unchanged; the `kp2=0` launch case now falls at **2.6 s** instead
of 5.9 s — the table's shape is unchanged: every `kp2=0` cell falls, every other cell holds);
and the **learned-policy tables did not move at all** (1.94 / 1.20 / 0.91° across `TORQUE_MAX`
18/28/40, exactly as published), because the pure-torque policy never passes through the PID's
booster or current EMA. That asymmetry is a useful fact about the model in its own right: the
filter rates live in the PID path, so a policy that bypasses the PID does not inherit their
errors — which is also why the saved policies stay valid without retraining.

## NEXT 356 — the voltage thresholds are per-cell, and comparing them raw made a healthy pack "over-voltage"

`motor_data.c:79-91` is four lines that decide when a Floatwheel starts leaning you back at low
and high battery voltage:

```c
uint8_t battery_cells = VESC_IF->get_cfg_int(CFG_PARAM_si_battery_cells);
if (battery_cells > 0) {
    if (lv_threshold < 10) lv_threshold *= battery_cells;
    if (hv_threshold < 10) hv_threshold *= battery_cells;
}
```

Two things had been wrong with my port of it. The thresholds themselves are the ones
`main.c:1169-1171` passes in — `float_conf.tiltback_lv` and `tiltback_hv` — and their shipped
defaults are **3.0 and 4.3**, which settings.xml's own help text calls per-cell values
("Set to a per-cell value (recommended 4.3V, which is now default)"; on pre-6.05 firmware the
same field takes the total instead, which is exactly why the comparison is guarded by `< 10`).
So comparing those numbers against a pack voltage, as `sim/protections.py` did until now, means
**any real battery satisfies `>= 4.3` on the first tick**: the sim held a permanent 8° HV
pushback on a 42 V board. It stayed invisible because every test that touched voltage armed
absolute numbers (30.0, 42.0, 60.0), which the `< 10` guard correctly leaves alone, and because
the plant gate had rows for the duty, speed and LV tiltbacks and **no HV row at all** — the
branch that was actually broken. `tools/probe_protections.py` now has one, armed in per-cell
volts, so the multiply is exercised in closed loop rather than only in a unit pin.

Second correction, to this row's own ask: `battery_cells` is **not** a Refloat parameter, so
there is nothing to extract from settings.xml — the count comes from VESC's own config
(`CFG_PARAM_si_battery_cells`), which is why grepping the 90 tags for `cell` finds only the
BMS cell thresholds. It belongs with the hardware constants in `sim/spec.py`
(`BATTERY_CELLS`, tagged `estimate`, surfaced by `python -m sim.spec --dump`) and reaches the
parameter object the way the other hardware numbers do — `Params` carries it under the
firmware's name, `si_battery_cells` — so a pin can put two pack sizes side by side. Fixing the
ask also meant fixing `tools/refloat_defaults.py`, whose default `--src` was one directory too
shallow: it reported "no settings.xml", which is precisely the kind of tool failure that
invites a guessed constant.

The pins that make it un-shippable (`python -m sim.protections`, 63 assertions): a 12S pack at
42 V on the shipped defaults is in **no** voltage tiltback (this one measured the bug); **one
30 V pack is low voltage on a 12S board and high voltage on a 6S one**, same volts, opposite
protections; a threshold of 10 V or more is left alone rather than multiplied again; `si_battery_cells
== 0` uses the stored number as-is instead of multiplying by zero; and the scaled pair lands on
36.000 / 51.600 exactly, where `motor_data.c:82-86` says it should.

Not modelled, named: the firmware refreshes this pair at most twice a second
(`main.c:1168-1173`), so a config change takes up to 500 ms to take effect — the sim's
parameters never move mid-run, so there is nothing to gain from faking the cache. And
`BATTERY_CELLS` is an estimate: the ADV500's usual pack is 12S, and if the rider's is 13S or 15S
every LV/HV voltage in this repo moves by that ratio, which is why NEXT 350 asks for it in the
same breath as the masses. Until then, say "the LV/HV tiltbacks are correct *per cell*", not
"they fire at 36 V".

## NEXT 359 — the booster was fed the *braked* setpoint, and `main.c:698` exists to undo that

```c
    pid_update(&d->pid, d->setpoint, &d->motor, &d->imu, &d->float_conf, dt);                     // :696
    float booster_proportional = d->setpoint - d->brake_tilt.setpoint.value - d->imu.pitch;       // :698
```

Two lines apart, and the difference between them is the entire behaviour of a board while it is
being braked. By line 696 `d->setpoint` already carries brake tilt along with every other offset, so
the PID is asked to hold the *tilted* attitude — that is the effect the rider feels. Line 698 then
hands the booster the target with brake tilt **subtracted back out**, so the launch assist keeps
pushing against a deck the brake tilt is pulling out from under the rider instead of leaning along
with it. My port fed both from one `self.setpoint`, which is the difference of exactly
`brake_tilt.setpoint.value`, in the one place the booster is load-bearing.

This sat in `sim/control.py` as a comment that named the line and then talked itself past it ("brake
tilt is 0 until braking"), which is the same mistake as the ATR excuse in `sim/control.py`'s old
header: a reason that only holds during the boring part of the manoeuvre is not a reason.

What it is worth, measured on `hard_brake`'s peak-brake tick (`python -m tools.probe_protections`,
the `booster_term` row): brake tilt **+3.39°**, booster error **-12.96°** where the skipped term gave
**-9.57°**, through `BOOSTER_KP` 4.0. Not a rounding error dressed as fidelity — it is the booster
being asked for a different number, 3.4° of it, at the moment the board is decelerating from speed.

What proves it, so the next person cannot quietly re-break it: the controller records
`booster_prop`/`pitch_used`, `run_case` keeps the largest-brake tick as `booster_audit`, and the gate
asserts the arithmetic against those recorded inputs — not against a torque figure, which could agree
for the wrong reason. Protections stay off by default and the report has no `braketilt_deg` when they
are off, so `python -m sim.control` re-ran byte-identical (0.98°/0.71 A/0.15 s hold; 3.27°/5.55
A/2.49 m/s launch; `kp2=0` falls at 2.57 s) and no saved policy is owed a retrain.

Two things this row does **not** claim: that 3.39° is the right figure for an ADV500 (NEXT 350's
unknowns, same caveat as every protections number here), and that the `brake_tilt` behaviour is
*tuned* — it still ends `fell=True` at the armed `braketilt_strength=10`, verified identical with this
change stashed, so it predates the row and is an armed extreme rather than a recommendation.

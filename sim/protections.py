"""The firmware's control protections: the behaviours that move the SETPOINT.

Usage:
    python -m sim.protections          # the pins: signs, ramps, gating, handtest coupling

Why this is a separate module: in the firmware none of these touch current or torque.
Each computes an *angle* that becomes the balance loop's target —

    d->setpoint_target = d->float_conf.tiltback_duty_angle;              // main.c:512
    d->setpoint_target = -d->float_conf.tiltback_hv_angle;                // main.c:542
    noseangling_target += d->float_conf.tiltback_constant * erpm_sign;   // main.c:687

— so they belong underneath the controller, not inside it. That placement is also the
answer to "how is a learned policy tested against the board's safety behaviour": the safety
behaviour lives below the policy, in code this sim already mirrors, so a policy that beats
the PID still gets tilted off the pad like any other rider.

Ported from (local clone of the firmware the real board runs, Refloat v1.3.1 @ a2b55d9):
  src/main.c:510-642            tiltback chain (duty / HV / LV / speed) + debounces
  src/main.c:191-195, 682-687   constant and variable noseangling
  src/torque_tilt.c:51-69       torque tilt
  src/turn_tilt.c:62-140        turn tilt (yaw rate AND aggregate)
  src/brake_tilt.c:37-97        brake tilt with lingering (and its ATR downhill damper)
  src/atr.c:25-163              ATR: expected-vs-measured acceleration, and its own filter
  src/motor_data.c:122, :147    the 8 Hz SMA of d(erpm)/dt that ATR measures with
  src/filters/sma.c:26-102      that SMA (window size derived from the cutoff, not chosen)
  src/main.c:1481-1497, 1935-1953  MODE_HANDTEST: the one that disables all the others

Named approximations, so they do not quietly become truth:
  * The firmware ramps every one of these through `smooth_setpoint_update` with per-feature
    on/off time constants and speed limits (e.g. turn_tilt.c:47-59). Those filter constants
    are not in the 90-name settings.xml vocabulary this sim carries, so each ramp here is a
    rate limiter at a documented deg/s. Consequence: the *final* angle is right, the
    *feel* of the onset is approximate. Anything that judges onset feel needs the filter
    constants from conf, not from this file.
  * Ramp shape, as of the two lanes that touched this file: torque tilt, turn tilt, brake tilt
    and ATR run through `SmoothSetpoint` (the firmware's own filter, NEXT 354). The tiltback
    chain and noseangling still ramp linearly, and that one IS faithful — they go through
    `rate_limitf`, not `smooth_setpoint_update` (main.c:914-918, main.c:690-692).
  * `wheelslip` (main.c:483-509) is ported and its four winddown paths have callers, pinned
    against torque_tilt.c:54-57, turn_tilt.c:93-96, brake_tilt.c:69-72, atr.c:77-81. What the
    plant cannot yet do is RIDING through one: the contact is rigid Coulomb, so when the tyre
    gives way the deck pitches out before the wheel can spin up while still touching the
    ground. Every large wheel-versus-deck speed gap this plant produces (26-37 m/s at mu
    0.04-0.12) belongs to a run that FELL, and the detector is correct not to latch on
    airborne spin. See tools/probe_protections.py's grip_ladder rows and NEXT 362 (tyre
    compliance). Nothing injects the flag: it is computed from the erpm and duty each run
    already carries.
  * ATR (atr.c) is ported: it compares the current it is pulling against the acceleration that
    current should be buying, and leans toward closing the gap (hill, headwind, load). Its one
    borrowed input is `motor->acceleration`, the 8 Hz SMA of d(erpm)/dt (motor_data.c:122, :147),
    which this module builds from the erpm it is handed each tick. What it therefore cannot see
    is the *filtered* current the firmware's motor->torque carries (motor_data.c:150-151, a
    biquad at `atr_filter` Hz): this model's torque is already a smoothed quantity, so ATR's
    input is smooth-by-substitution rather than smooth-by-mirror. Say "ATR's torque input is
    the model's own", not "the firmware's current filter".

Inputs the 1-D sim cannot produce are injected: `battery_volt` and `duty` (a battery and a
duty cycle are scenario inputs, not things to model in order to test a voltage-triggered
lean), and `yaw_rate_dps` (a board that only goes straight never develops yaw). When one is
None the behaviour that needs it stays inactive and says so, rather than assuming a value.
"""

from __future__ import annotations

import math

from . import spec

LOOP_HERTZ_COMPAT = 720.0
LOOP_HERTZ_COMPAT_RECIP = 1.0 / LOOP_HERTZ_COMPAT
TORQUE_CONSTANT_COMPAT = 1.5 * 15 * 0.027  # src/lib/utils.h:34-35
YAW_RATE_LIMIT_DPS = 72.0                  # turn_tilt.c:73 clamp
YAW_AGGREGATE_MIN_DPS = 30.0               # turn_tilt.c:81, :104

# NEXT 349's Measure, as data rather than prose: the firmware control-protection behaviours
# this module mirrors into the controller path, on top of the three sim/control.py always had
# (balance PID, booster, current clamps). docs/TILTBACK_PORT.md's "What got ported" section
# is the same list with its citations; this tuple is the copy a gate can count, so "how many
# are mirrored" is read off the code instead of re-remembered — which is the failure that
# made an earlier count wrong: it was counted from a docstring, over a behaviour whose name
# had been truncated out of it. Count it with `python tools/probe_protections.py`.
MIRRORED_BEHAVIOURS = (
    "duty tiltback",            # main.c:510 (the chain is 510-642, this file's class cites each rung)
    "HV tiltback",              # main.c:538-542, debounced 500 ms
    "LV tiltback",              # main.c:561/576/593, per-cell below 10 V (motor_data.c:79-91)
    "over-speed tiltback",      # main.c:637-642 (reuses the duty angle: quirk, mirrored)
    "constant noseangling",     # main.c:191-195, 682-687 (tiltback_constant)
    "variable noseangling",     # main.c:191-195, 682-687 (tiltback_variable/_max)
    "torque tilt",              # torque_tilt.c:51-69
    "turn tilt",                # turn_tilt.c:62-140
    "brake tilt",               # brake_tilt.c:37-97
    "ATR",                      # atr.c:25-163 (the last to land, 8698e89)
    "MODE_HANDTEST disable list",  # main.c:1481-1497 — with what it spares
)


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _sign(x):
    return 0.0 if x == 0 else (1.0 if x > 0 else -1.0)


def _ema_alpha(cutoff_hz, update_hz):
    omega = min(2.0 * math.pi * cutoff_hz / update_hz, 0.5)
    return omega - 0.5 * omega * omega


def _ema_alpha_time_constant(time_constant, update_hz):
    """ema.c:33-35 — alpha from a TIME CONSTANT rather than a cutoff, so the smoothing per
    *second* is the same however often update() is called. That is what lets this sim run the
    filter at its 100 Hz control loop where the firmware runs it at the ~500 Hz main loop
    (main.c:61, configured per measured frequency at main.c:154-163)."""
    if time_constant <= 0.0:
        # C computes 1/(0*f) as +inf, utils.c:28 clamps omega to 0.5, so alpha is 0.375.
        # Python raises ZeroDivisionError instead of producing inf, so spell the result out.
        return 0.375
    return _ema_alpha(1.0 / (2.0 * math.pi * time_constant), update_hz)


class SmoothSetpoint:
    """Faithful port of src/filters/smooth_setpoint.c:25-109 (Refloat v1.3.1 @ a2b55d9).

    This is what stands between a protection's target angle and the rider feeling it, and it
    is not a rate limiter — it is an EMA pair with *different* time constants for starting to
    move and stopping, plus a rate clamp whose limit depends on whether the angle is growing,
    shrinking or crossing zero. Three consequences worth knowing before trusting a table:

      * on/off asymmetry: `on_speed_*` limits apply while the value grows toward the target,
        `off_speed_*` while it returns (:96-102), and a zero-crossing gets `max()` of the two.
      * `step` is a filtered estimate of the increment (:83-91), so the ramp builds up over
        a few ticks instead of starting at full speed — an onset, not a step.
      * `alpha` is multiplied by 2.146 (:58) because it feeds a second-order arrangement.

    Used by torque tilt (torque_tilt.c:69), turn tilt (turn_tilt.c:139), brake tilt
    (brake_tilt.c:96), ATR (atr.c:162) and the remote (remote.c:156). NOT used by the
    tiltbacks or noseangling, which ramp linearly through `rate_limitf` (main.c:914-918 and
    main.c:690-692) — a distinction this module used to get wrong in the other direction.
    """

    def __init__(self, frequency=None):
        self.alpha = 0.0
        self.on_speed_alpha = 0.0
        self.off_speed_alpha = 0.0
        self.winddown_alpha = 0.0
        self.frequency = frequency
        self.reset()

    def configure(
        self,
        time_constant,
        on_speed_time_constant,
        off_speed_time_constant,
        winddown_time_constant,
        on_speed_up,
        off_speed_up,
        on_speed_down,
        off_speed_down,
        frequency=None,
    ):
        """smooth_setpoint.c:39-62, same argument order. Every caller passes a 0.2 s winddown."""
        self.on_speed_up = on_speed_up
        self.off_speed_up = off_speed_up
        self.on_speed_down = on_speed_down
        self.off_speed_down = off_speed_down
        if frequency is not None:
            self.frequency = frequency
        f = self.frequency
        self.alpha = 2.146 * _ema_alpha_time_constant(time_constant, f)
        self.on_speed_alpha = _ema_alpha_time_constant(on_speed_time_constant, f)
        self.off_speed_alpha = _ema_alpha_time_constant(off_speed_time_constant, f)
        self.winddown_alpha = _ema_alpha_time_constant(winddown_time_constant, f)

    def reset(self):
        self.is_winddown = False
        self.v1 = 0.0
        self.step = 0.0
        self.value = 0.0

    def winddown(self):
        """smooth_setpoint.c:107-109. Called only on wheelslip, which this plant can cause but not
        yet ride: the wheel runs away from the deck and the run ends in a fall (NEXT 361), so this
        path is exercised by the pins and the probe -- present so the port is complete, and pinned
        below so it stays correct."""
        self.is_winddown = True
        self.value *= 1.0 - self.winddown_alpha

    def update(self, target, forward, mult, dt):
        """smooth_setpoint.c:71-105. `forward` is motor->forward, the direction of travel."""
        if self.is_winddown:
            self.is_winddown = False
            self.v1 = self.value
            self.step = 0.0

        is_up = (self.value >= 0.0) == forward

        self.v1 += mult * self.alpha * (target - self.v1)
        delta = mult * self.alpha * (self.v1 - self.value)

        if abs(delta) > abs(self.step) or _sign(delta) != _sign(self.step):
            if _sign(self.value) == _sign(delta):
                self.step += mult * self.on_speed_alpha * (delta - self.step)
            else:
                self.step += mult * self.off_speed_alpha * (delta - self.step)
        else:
            self.step = delta

        on_speed = self.on_speed_up if is_up else self.on_speed_down
        off_speed = self.off_speed_up if is_up else self.off_speed_down
        if self.value * target < 0:
            speed_limit = max(on_speed, off_speed)          # crossing zero: the faster of the two
        elif abs(self.value) > abs(target):
            speed_limit = off_speed                         # shrinking
        else:
            speed_limit = on_speed                          # growing

        self.value += _sign(self.step) * min(abs(self.step), mult * speed_limit * dt)
        return self.value


# The parameters MODE_HANDTEST zeroes (main.c:1484-1495) plus the ones it sets to a
# neutral value. A dict, not a hard-coded list, so a sweep can see what a mode did.
HANDTEST_DISABLED = (
    "torquetilt_strength",
    "torquetilt_strength_regen",
    "atr_strength_up",
    "atr_strength_down",
    "turntilt_strength",
    "tiltback_constant",
    "tiltback_variable",
    "tiltback_variable_max",
)
HANDTEST_FORCED = {"ki": 0.0, "kp_brake": 1.0, "kp2_brake": 1.0,
                   "brkbooster_angle": 100.0, "booster_angle": 100.0}


def _fw_sign(x):
    """The firmware's own `sign(x)` macro, lib/utils.h:99: `((x) < 0) ? -1 : 1`.

    It returns +1 for zero, which is NOT the same function as `_sign` above (whose 0 maps to
    0). Both are correct, in different places: `_sign` is used for tiltback *direction*, where
    "no lean" must be 0, while atr.c:101 and :143 need the macro's semantics — a port that
    shares one helper between the two silently changes what ATR does at exactly zero torque,
    which is the most common state on a flat road.
    """
    return -1 if x < 0 else 1


class Sma:
    """Port of src/filters/sma.c — a sliding moving average over a derived number of samples.

    `n` is not a knob: sma_calculate_n (sma.c:26-29) derives it from the cutoff, so an 8 Hz
    cutoff at the firmware's ~500 Hz main loop is a ~27-sample window, while the same 8 Hz
    here at a 1 kHz plant is ~125. Both fall out of the same formula rather than being
    hand-picked, which is what keeps `motor->acceleration` the same physical quantity.
    """

    def __init__(self, cutoff_hz, update_hz):
        # sma.c:26-29, including the uint8_t truncation on return and the 255 clamp.
        fc = cutoff_hz / update_hz if update_hz > 0 else 0.0
        self.n = 255 if fc <= 0 else max(1, int(min(math.sqrt(0.196202 + fc * fc) / fc, 255.0)))
        self.value = 0.0
        self._buf = [0.0] * self.n
        self._idx = 0

    def reset(self):                                    # sma.c:65-72
        self.value = 0.0
        self._buf = [0.0] * self.n
        self._idx = 0

    def update(self, target):                           # sma.c:74-77
        self.value += (target - self._buf[self._idx]) / self.n
        self._buf[self._idx] = target
        if self._idx + 1 >= self.n:
            self._idx = 0
        else:
            self._idx += 1


class Atr:
    """Port of src/atr.c:25-163 (Refloat v1.3.1 @ a2b55d9) — Acceleration Tracking Response.

    What it actually is, in one sentence: ATR compares how the board IS accelerating against how
    it SHOULD be accelerating for the current it is pulling, and leans the rider toward the
    direction that would restore the difference. That single idea is the whole module:

        accel_diff = expected_acc(from torque) - measured_acc(from d(erpm)/dt)

    A hill or a headwind burns current without accelerating, so `accel_diff` grows, and ATR
    tilts the nose up the hill to keep the rider over the wheel. A flat-road cruise produces
    current that DOES accelerate, so accel_diff sits at zero and ATR stays out of it. That is
    why a load-based test and a speed-based test pin different things, and why a port that
    reads torque alone passes a naive test and is still wrong.

    Three things a re-derivation tends to lose, so they are pinned in `main()`:

      * the 15 Nm knee (atr.c:97-104): below it the torque->acceleration map is linear, above
        it a second, softer slope (1.3x the denominator). It is the difference between ATR
        saturating usefully and ATR fighting the rider at full lean.
      * the 8 A offset multiplied by `erpm_sign` (:84, :98, :102): the current needed just to
        hold speed, subtracted in the direction of travel. Without it ATR leans on every
        steady-state cruise.
      * the acceleration gate (:107-117): below 250 erpm `accel_diff` is pinned to 0 — ATR is
        a riding feature, not a standstill jitter. The three alphas above it (10/6/1 Hz) trade
        response for smoothness as speed rises.

    Its output is NOT added to the setpoint directly: main.c:950 sums `atr.setpoint.value +
    brake_tilt.setpoint.value` and combines that with torque tilt by max-of-magnitude on equal
    signs, add on opposite. See the aggregation in `Protections.update`.
    """

    def __init__(self, getparam, frequency):
        """`getparam` is `Protections._p`, so MODE_HANDTEST's zeroing of atr_strength_up/down
        (main.c:1484-1495) gates ATR through the same door as everything else here."""
        self._g = getparam

        # atr.c:48-53: the speed boost starts at 1/3000 and the configured boost buys more
        # erpm-per-unit, i.e. a bigger atr_speed_boost reaches its cap later.
        self.speed_boost_mult = 1.0 / 3000.0
        boost = abs(self._g("atr_speed_boost"))
        if boost > 0.4:
            self.speed_boost_mult = 1.0 / ((boost - 0.4) * 5000.0 + 3000.0)

        self.ad_alpha3 = _ema_alpha(10.0, frequency)   # atr.c:55
        self.ad_alpha2 = _ema_alpha(6.0, frequency)    # atr.c:56
        self.ad_alpha1 = _ema_alpha(1.0, frequency)    # atr.c:57
        self.transition_alpha = _ema_alpha(6.0, frequency)   # atr.c:59

        # atr.c:60-71: the same filter shape the tilts use, with the firmware's own 0.2 s
        # winddown, and — note the shape of the call — on_speed_limit passed TWICE, so ATR's
        # rate limit does not depend on direction. Only the TIME constants differ on/off.
        self.setpoint = SmoothSetpoint(frequency)
        self.setpoint.configure(
            self._g("atr.filter.time_constant"),
            self._g("atr.filter.on_speed_time_constant"),
            self._g("atr.filter.off_speed_time_constant"),
            0.2,
            self._g("atr.filter.on_speed_limit"),
            self._g("atr.filter.off_speed_limit"),
            self._g("atr.filter.on_speed_limit"),
            self._g("atr.filter.off_speed_limit"),
        )
        self.reset()

    def reset(self):
        """atr.c:37-45 (atr_reset), which atr_init also runs."""
        self.accel_diff = 0.0
        self.speed_boost = 0.0
        self.target = 0.0
        self.transition_value = 0.0    # ema value of transition_target (atr.c:42)
        self.transition_boost = 1.0
        self.setpoint.reset()

    def update(self, torque, erpm, abs_erpm, erpm_sign, braking, forward, acceleration, dt):
        """One atr_update (atr.c:74-162).

        `acceleration` is motor->acceleration.value, the 8 Hz SMA of d(erpm)/dt that
        motor_data.c:122/:147 produces; the caller owns that filter.
        """
        # atr.c:77-81 wheelslip: winddown + reset and return. The plant can make the wheel run away
        # from the deck, but only in a run that is already falling, so this branch is exercised by
        # the pins and by tools/probe_protections.py rather than by a riding scenario -- and it is
        # still correct the day tyre compliance lands (NEXT 362).
        abs_torque = abs(torque)
        torque_offset = 8 * TORQUE_CONSTANT_COMPAT      # atr.c:84 "hard-code to 8A"
        threshold = self._g("atr_threshold_down") if braking else self._g("atr_threshold_up")
        accel_factor = (
            self._g("atr_amps_decel_ratio") if braking else self._g("atr_amps_accel_ratio")
        ) * TORQUE_CONSTANT_COMPAT                      # atr.c:86-88
        accel_factor2 = accel_factor * 1.3              # atr.c:89

        # measured: how fast the wheel really is accelerating, in the firmware's own units
        # (erpm-per-loop-tick, clamped to +-5). atr.c:92
        measured_acc = _clamp(acceleration * LOOP_HERTZ_COMPAT_RECIP, -5.0, 5.0)

        # expected: what that current should be buying. atr.c:94-104.
        # The 15 here is compared against torque in N*m, so it is 15 N*m ~ 24.7 A on this
        # motor — NOT 15 A. torque_tilt.c:63 multiplies its threshold by TORQUE_CONSTANT_COMPAT
        # and atr.c does not. Quirk, mirrored, not "fixed" (same policy as the LV/HV ramp
        # asymmetry above): a board whose ATR knee is at 15 A is not the real board.
        if abs_torque < 15:
            expected_acc = (torque - erpm_sign * torque_offset) / accel_factor
        else:
            torque_sign = _fw_sign(torque)
            expected_acc = (torque_sign * 15 - erpm_sign * torque_offset) / accel_factor
            expected_acc += torque_sign * (abs_torque - 15) / accel_factor2

        new_accel_diff = expected_acc - measured_acc    # atr.c:106
        if abs_erpm > 250:                              # atr.c:107-117
            alpha = self.ad_alpha3
            if abs_erpm > 2000:
                alpha = self.ad_alpha1
            elif abs_erpm > 1000:
                alpha = self.ad_alpha2
            self.accel_diff += alpha * (new_accel_diff - self.accel_diff)
        else:
            self.accel_diff = 0.0

        # The sign table at atr.c:119-124, transcribed literally: forward AND accel_diff
        # agreeing picks the uphill strength, disagreeing picks the downhill one. accel_diff
        # > 0 means "you are not accelerating as hard as this current should", so moving
        # forward with a positive diff is climbing, and uphill/downhill is about the sign of
        # the shortfall — not the direction of travel.
        atr_strength = (
            self._g("atr_strength_up") if forward == (self.accel_diff > 0)
            else self._g("atr_strength_down")
        )

        # atr.c:127-136: above 3000 erpm while driving, crank the response by the speed boost.
        if abs_erpm > 3000 and not braking:
            speed_boost_mult = (abs_erpm - 3000.0) * self.speed_boost_mult
            self.speed_boost = min(1.0, speed_boost_mult) * self._g("atr_speed_boost")
            atr_strength += atr_strength * self.speed_boost
        else:
            self.speed_boost = 0.0

        # atr.c:139-146: deadband, then the limit. The subtraction (not a clamp to zero) is
        # what makes it a threshold rather than a cliff — the output still starts at zero when
        # it engages, so nothing pops.
        new_atr_target = atr_strength * self.accel_diff
        if abs(new_atr_target) < threshold:
            new_atr_target = 0.0
        else:
            new_atr_target -= _fw_sign(new_atr_target) * threshold
        limit = self._g("atr_angle_limit")
        self.target = _clamp(new_atr_target, -limit, limit)

        # atr.c:148-160: a 6 Hz EMA of the target, and the transition boost that only fires
        # when the filtered value and the (filtered) target are on OPPOSITE sides of zero and
        # more than a degree apart — the case where ATR has to cross its own output.
        self.transition_value += self.transition_alpha * (self.target - self.transition_value)
        transition_target = self.transition_value
        degrees_diff = abs(self.setpoint.value - transition_target) - 1.0
        if self.setpoint.value * transition_target < 0 and degrees_diff > 0.0:
            self.transition_boost = (
                1.0 + min(degrees_diff, 1.0) * (self._g("atr.transition_boost") - 1.0)
            )
        else:
            self.transition_boost = 1.0

        # atr.c:162: to the rider through the filter, mult = the transition boost.
        self.setpoint.update(self.target, forward, self.transition_boost, dt)


class InputTilt:
    """Port of src/remote.c's inputtilt path (Refloat v1.3.1 @ a2b55d9): the rider's lean-in trim.

    The stick value becomes a TARGET LEAN ANGLE the board is then asked to hold —

        remote.c:149   target = remote->input * config->inputtilt_angle_limit;
        remote.c:156   smooth_setpoint_update(&remote->setpoint, target, true, 1.0f, dt);

    and main.c:922-923 runs it BEFORE every other tilt and adds the result straight into `setpoint`,
    so the whole protection fleet sees the leaned-into posture as the rider's own request. That
    includes brake_tilt_update, which at :939 is handed `setpoint - imu.balance_pitch` with this term
    already inside it — which is why the controller feeds `balance_offset + input_tilt_out` to the
    brake term here rather than the bare offset.

    NOT ported, named instead of invented:
      * the acquisition (remote.c:70-89): PPM/UART stick reads, REMOTE_TIMEOUT, the `connected`
        flag. The caller hands the stick value over directly. `inputtilt_remote_type` 0 is
        INPUTTILT_NONE, which takes the :81-88 branch — so "no stick fitted" behaves exactly like
        the firmware and the stage is inert until somebody fits one.
      * move_speed / remote_get_move_torque (remote.c:100-112, :134-146): a constant-speed TORQUE
        feature on a different path, with no line in this plant yet.
      * `inputtilt_speed` was REMOVED upstream (main.c:1815-1816). The ramp is whatever the filter's
        hard-coded 100 deg/s limits give (remote.c:53-56), so nothing here offers a speed knob.
    """

    def __init__(self, getparam, frequency):
        self.getparam = getparam
        # remote.c:45-58 (remote_configure). The on/off speed time constants are a QUARTER of the
        # main one (:46) — turn_tilt does the same trick with a half — winddown is 0.2 s like every
        # other filter here, and all four rate limits are the literal 100.0 the firmware passes.
        self.filter = SmoothSetpoint(frequency)
        tc = float(self.getparam("remote.filter.time_constant"))
        self.filter.configure(tc, tc * 0.25, tc * 0.25, 0.2, 100.0, 100.0, 100.0, 100.0)
        self.input = 0.0
        self.target = 0.0

    def reset(self):
        self.filter.reset()                               # remote.c:39 (remote_reset)
        self.input = 0.0
        self.target = 0.0

    def update(self, value, dt, *, darkride=False):
        """remote.c:85-118 (deadband, invert) then :148-156 (target, smooth ramp), in that order."""
        if not self.getparam("inputtilt_remote_type"):    # INPUTTILT_NONE, remote.c:81-88
            self.input = 0.0
        else:
            db = float(self.getparam("inputtilt_deadband"))
            v = float(value or 0.0)
            a = abs(v)
            if a < db:
                v = 0.0
            else:
                # remote.c:92-97: the deadband is REMAPPED, not clipped. A stick at 0.16 with a 0.10
                # deadband reads 0.06/0.90 = 0.067 — not 0.16, not 0. A plain threshold would snap
                # a tenth of a degree of trim into the setpoint the moment the stick left the
                # deadband, which is the jump this remap exists to prevent.
                v = _sign(v) * (a - db) / max(1e-9, 1.0 - db)
            if self.getparam("inputtilt_invert_throttle"):
                v = -v                                    # remote.c:114-116
            self.input = v

        target = self.input * float(self.getparam("inputtilt_angle_limit"))   # remote.c:149
        if darkride:
            target = -target                              # remote.c:151-153
        self.target = target
        # remote.c:156: `true` and `1.0f` are literals in the firmware ("the `forward` argument
        # doesn't matter, as up and down speeds are the same"). Unlike ATR at atr.c:162, the remote
        # passes no transition boost, so `mult` stays 1.0 and the ramp is the limits' business.
        self.filter.update(target, True, 1.0, dt)
        return self.filter.value

    @property
    def value(self):
        return self.filter.value


class Protections:
    """One instance per board. Call update() once per control tick, after the balance
    filter has run; it returns the angle the balance loop should chase in ADDITION to
    whatever the rider or a policy asked for."""

    def __init__(self, params, control_hz=None, handtest=False):
        self.p = params
        self.hz = control_hz or spec.g("CONTROL_HZ")
        self.handtest = handtest
        self.a_yaw = _ema_alpha(25.0, self.hz)   # turn_tilt.c:42
        self._step_hz = None                     # the rate update() is ACTUALLY stepped at
        self._configure_filters()
        self.reset()

    def _track_step_frequency(self, dt):
        """Configure the filters at the frequency they are really stepped at.

        The firmware does exactly this: `main_freq_tracker` measures the main loop and
        re-runs every `*_configure` when it changes (main.c:154-163, 785). Configuring at a
        nominal rate and stepping at another is a silent mistune — the alpha is per-step, so
        a filter built for 100 Hz and stepped at 1 kHz smooths ten times less than the time
        constant asks for, and nobody sees it in a table until a scenario starts falling.

        This matters here because `CONTROL_HZ` is a tagged `estimate` (NEXT 350 is asking
        the rider for it) and the loop's real step is MuJoCo's `model.opt.timestep`. Those two
        disagree today: spec says 100 Hz, the plant steps at 1 kHz. Measuring instead of
        trusting the constant means the protections are correct either way, and the
        `estimate` stays the rider's question rather than becoming my hidden assumption.
        """
        step_hz = 1.0 / dt if dt > 0 else self.hz
        if self._step_hz is None or abs(step_hz - self._step_hz) > 1e-6:
            self._step_hz = step_hz
            self.hz = step_hz
            self.a_yaw = _ema_alpha(25.0, step_hz)   # turn_tilt.c:42, same treatment
            self._configure_filters()

    def _configure_filters(self):
        """Where each tilt gets its SmoothSetpoint, exactly as the firmware configures it.

        Configured once at construction, like the firmware's `*_configure` calls (which run
        on a measured-frequency change, main.c:154-163) — so sweeping `braketilt_lingering`
        means building a new Protections, which is what the sweep tools do per run. A filter
        re-tuned mid-ride is not a thing the board does.
        """
        # Raw config, not _p(): MODE_HANDTEST zeroes protection *strengths*, never a filter
        # constant, so reading the handtest-aware accessor here would invent a coupling.
        # torque_tilt.c:36-49 — own filter fields, winddown 0.2 s, same limits up and down
        self.torque_filter = SmoothSetpoint(self.hz)
        self.torque_filter.configure(
            self.p["torque_tilt.filter.time_constant"],
            self.p["torque_tilt.filter.on_speed_time_constant"],
            self.p["torque_tilt.filter.off_speed_time_constant"],
            0.2,
            self.p["torque_tilt.filter.on_speed_limit"],
            self.p["torque_tilt.filter.off_speed_limit"],
            self.p["torque_tilt.filter.on_speed_limit"],
            self.p["torque_tilt.filter.off_speed_limit"],
        )
        # turn_tilt.c:47-59 — speed time constants are half the main one, and all four rate
        # limits are hard-coded 20 deg/s (they are not configurable in the firmware either)
        turn_tc = self.p["turn_tilt.filter.time_constant"]
        self.turn_filter = SmoothSetpoint(self.hz)
        self.turn_filter.configure(turn_tc, turn_tc * 0.5, turn_tc * 0.5, 0.2,
                                   20.0, 20.0, 20.0, 20.0)
        # brake_tilt.c:37-59 — brake tilt borrows ATR's filter wholesale, and derives its
        # release speed from ATR's off_speed_limit divided by `braketilt_lingering`: the
        # parameter's whole job is to slow the nose back down (brake_tilt.c:45)
        off_speed = self.p["atr.filter.off_speed_limit"] / max(self.p["braketilt_lingering"], 1.0)
        self.brake_filter = SmoothSetpoint(self.hz)
        # atr.c:47-71 (atr_configure, called from main_freq_update_reconfigure at main.c:160)
        # and motor_data.c:122 (sma_configure(&m->acceleration, 8.0f, frequency)) — the 8 Hz
        # cutoff is a hard-coded constant in the firmware, not a setting.
        self.atr = Atr(self._p, self.hz)
        # NEXT 365: the rider's lean-in trim (remote.c). Built here rather than in __init__ because
        # a step-frequency change re-tunes every filter in this method — the remote's is no different
        # from ATR's in that respect (main.c:154-163 re-configures all of them together).
        self.input_tilt = InputTilt(self._p, self.hz)
        self.acc_sma = Sma(8.0, self.hz)
        self.brake_filter.configure(
            self.p["atr.filter.time_constant"],
            self.p["atr.filter.on_speed_time_constant"],
            self.p["atr.filter.off_speed_time_constant"],
            0.2,
            self.p["atr.filter.on_speed_limit"],
            off_speed,
            self.p["atr.filter.on_speed_limit"],
            off_speed,
        )

    def reset(self):
        self.tiltback_target = 0.0
        self.noseangling = 0.0
        self.torque_tilt_out = 0.0
        self.turn_tilt_out = 0.0
        self.brake_tilt_out = 0.0
        # main.c:483-509 `state.wheelslip`. Latching with a release hold, not instantaneous:
        # the firmware stays slipped for 0.2 s after the signal clears "to avoid any
        # overreactions" (main.c:501), and that hold is what keeps a board from chattering
        # its tilt modifiers down and back up mid-slide.
        self.wheelslip = False
        self._slip_age = 0.0
        # NEXT 365: the remote's trim is per-ride state. remote_reset (remote.c:38-43) zeroes the
        # input, the setpoint filter and the move PID; the move PID is not ported, so the first two
        # are what a fresh Protections must not carry over from the run before it.
        if getattr(self, "input_tilt", None) is not None:
            self.input_tilt.reset()
        self.input_tilt_out = 0.0
        self.brake_target = 0.0            # brake_tilt.c:78's stale target (see _brake_tilt)
        # ATR (atr.c:37-45). Built in _configure_filters, like the firmware builds it in
        # main_freq_update_reconfigure; guarded because __init__ reaches reset() before the
        # first configure pass exists. atr_out is what main.c:950 actually adds: the FILTERED
        # value, never atr.target.
        if getattr(self, "atr", None) is not None:
            self.atr.reset()
        self.atr_out = 0.0
        self.atr_accel_diff = 0.0
        # motor->last_erpm: motor_data.c:147 differences the RAW encoder reading, and the
        # first tick after a reset therefore measures from 0, exactly as a booting board does.
        self._last_erpm = 0.0
        for f in (getattr(self, "torque_filter", None),
                  getattr(self, "turn_filter", None),
                  getattr(self, "brake_filter", None)):
            if f is not None:              # first call from __init__ precedes configure
                f.reset()
        self.yaw_change = 0.0            # turn_tilt.c: yaw_change EMA value
        self.yaw_aggregate = 0.0         # turn_tilt.c:35
        self.batt_last = None            # "last voltage the rider saw"
        self.batt_rise_since = None      # start of the rising edge (500 ms debounce)
        self.batt_last_beep = None
        self.active = ""
        self.brake_factor = 0.0
        if self.p["braketilt_strength"] != 0:
            # brake_tilt.c:42, negative sign folded in "instead of adding it each
            # balance loop" - the comment is the reason the sign is not a bug.
            self.brake_factor = -(0.5 + (20 - self.p["braketilt_strength"]) / 5.0)

    def _p(self, name):
        """Effective parameter: MODE_HANDTEST overrides the firmware's own values."""
        if self.handtest and name in HANDTEST_DISABLED:
            return 0.0
        return self.p.get(name, 0.0)

    # --- tiltback chain (main.c:510-642) --------------------------------------
    def _lv_hv_thresholds(self):
        """Return (lv, hv) tiltback thresholds in ABSOLUTE pack volts (motor_data.c:79-91).

        Refloat stores both as PER-CELL volts — its own settings.xml help says so outright:
        "Set to a per-cell value (recommended 4.3V, which is now default)" — and the firmware
        only ever compares them against pack voltage after multiplying by VESC's
        `si_battery_cells`, and only when the stored number is below 10 V, so a tuner can
        still type an absolute figure and a board that reports 0 cells keeps the raw value.

        Skipping the multiply is not a rounding error. With the shipped defaults a 42 V pack
        satisfies `battery_volt >= 4.3` on the first tick, i.e. the sim holds a permanent 8°
        HV pushback on a healthy battery and everything tested downstream of the LV branch is
        silently measuring the wrong branch. `si_battery_cells` is the firmware's own name and
        overrides here so a pin can put two pack sizes side by side; the default is the
        hardware constant in `sim/spec.py`, which is the rider's to answer (NEXT 350). The
        firmware refreshes this pair at most twice a second (`main.c:1168-1173`); the sim's
        config never changes mid-run, so that cache is named rather than faked.
        """
        cells = int(self.p.get("si_battery_cells", spec.g("BATTERY_CELLS")))
        lv = self._p("tiltback_lv")
        hv = self._p("tiltback_hv")
        if cells > 0:
            if lv < 10.0:   # motor_data.c:82 — "under 10 V" is how it spots a per-cell value
                lv *= cells
            if hv < 10.0:   # motor_data.c:85
                hv *= cells
        return lv, hv

    def _tiltback_wanted(self, duty, battery_volt, forward_speed, dt, now):
        """Return (target_deg, ramp_deg_per_s, which) for the one tiltback in force.

        The if/else-if order in the firmware is load-bearing: duty beats HV beats LV beats
        speed, so a board that is both over-duty and over-speed tilts at the duty rate.
        """
        angle_duty = self._p("tiltback_duty_angle")
        speed_duty = self._p("tiltback_duty_speed")
        # main.c:510 duty
        if duty is not None and duty > self._p("tiltback_duty"):
            return angle_duty, speed_duty, "duty"
        # main.c:538 high voltage, debounced 500 ms ("500ms have passed or voltage is
        # another volt higher, time for some tiltback")
        if battery_volt is not None:
            lv_thr, hv_thr = self._lv_hv_thresholds()   # motor_data.c:79-91, per-cell -> pack
            if self.batt_last is not None and battery_volt > self.batt_last + 1.0:
                self.batt_rise_since = now
            if battery_volt >= hv_thr:
                if self.batt_rise_since is None or now - self.batt_rise_since >= 0.5:
                    return (
                        self._p("tiltback_hv_angle"),
                        self._p("tiltback_hv_speed"),
                        "hv",
                    )
            # main.c:561/576/593 low voltage. Note the deliberate asymmetry the firmware
            # spells out in comments three times: LV ANGLE at the HV RAMP SPEED. Mirroring
            # it is correct; "fixing" it would make the board safer than the rider's.
            if battery_volt <= lv_thr:
                return (
                    self._p("tiltback_lv_angle"),
                    self._p("tiltback_hv_speed"),
                    "lv",
                )
            self.batt_last = battery_volt
        # main.c:637-642 over speed. It reuses tiltback_duty_angle, NOT a speed-specific
        # angle - there is no tiltback_speed_angle in the firmware at all. Quirk: mirror it.
        if self._p("tiltback_speed") > 0.0 and abs(forward_speed) > self._p("tiltback_speed"):
            return angle_duty, speed_duty, "speed"
        return 0.0, self._p("tiltback_return_speed"), ""

    # --- noseangling (main.c:191-195, 682-687) --------------------------------
    def _noseangling_wanted(self, erpm):
        sign_erpm = _sign(erpm)
        var = self._p("tiltback_variable")
        var_max = self._p("tiltback_variable_max")
        target = 0.0
        if var != 0.0:
            # main.c:191-195: the per-erpm rate, and the erpm at which it saturates
            rate = var / 1000.0 * _sign(var_max)
            max_erpm = abs(var_max / var)
            erpm_term = _clamp(abs(erpm) - self._p("tiltback_variable_erpm"), 0.0, max_erpm)
            target += rate * erpm_term * sign_erpm
        if abs(erpm) > self._p("tiltback_constant_erpm"):
            target += self._p("tiltback_constant") * sign_erpm
        return target

    # --- torque tilt (torque_tilt.c:51-69) ------------------------------------
    def _torque_tilt_wanted(self, torque_nm):
        strength = (
            self._p("torquetilt_strength_regen")
            if torque_nm < 0
            else self._p("torquetilt_strength")
        )
        strength *= 1.0 / TORQUE_CONSTANT_COMPAT        # torque_tilt.c:61
        torque_base = max(
            abs(torque_nm) - self._p("torquetilt_start_current") * TORQUE_CONSTANT_COMPAT, 0.0
        )                                              # torque_tilt.c:63-65
        return min(torque_base * strength, self._p("torquetilt_angle_limit")) * _sign(torque_nm)

    # --- turn tilt (turn_tilt.c:62-140) ---------------------------------------
    def _turn_tilt(self, yaw_rate_dps, erpm, dt):
        if self._p("turntilt_strength") == 0:
            # turn_tilt.c:89-91 returns before smooth_setpoint_update, so the caller must
            # NOT run the filter: the output freezes at its last value. Returning 0.0 here
            # would be the plausible, wrong version — it would look like a clean release.
            return None
        if yaw_rate_dps is None:
            # No yaw sensor in this sim means no turn is being made, not that the feature is
            # off: target 0, so the filter ramps the tilt away instead of freezing it.
            return 0.0
        # turn_tilt_aggregate(): EMA of the clamped rate, aggregate only counts real turns
        self.yaw_change += self.a_yaw * (
            _clamp(yaw_rate_dps, -YAW_RATE_LIMIT_DPS, YAW_RATE_LIMIT_DPS) - self.yaw_change
        )
        if _sign(self.yaw_change) != _sign(self.yaw_aggregate):
            self.yaw_aggregate = 0.0                    # turn_tilt.c:76-78 direction flip
        if abs(self.yaw_change) > YAW_AGGREGATE_MIN_DPS:
            self.yaw_aggregate += yaw_rate_dps * dt     # turn_tilt.c:81-83
        abs_change = abs(self.yaw_change)
        abs_aggregate = abs(self.yaw_aggregate)
        # turn_tilt.c:104: BOTH a rate and an accumulated turn are required. This is the
        # detail a naive port drops, and it is why wiggling over cracks does not tilt you.
        if abs_aggregate < self._p("turntilt_start_angle") or abs_change < YAW_AGGREGATE_MIN_DPS:
            return 0.0
        target = abs_change * LOOP_HERTZ_COMPAT_RECIP * self._p("turntilt_strength")
        boost_per_erpm = (
            self._p("turntilt_erpm_boost") / 100.0 / max(self._p("turntilt_erpm_boost_end"), 1.0)
        )                                              # turn_tilt.c:44-45
        if abs(erpm) < self._p("turntilt_erpm_boost_end"):
            target *= 1.0 + abs(erpm) * boost_per_erpm  # turn_tilt.c:112-113
        else:
            target *= 1.0 + self._p("turntilt_erpm_boost") / 100.0   # turn_tilt.c:115
        damper = 0.5 if abs(erpm) < 2000 else 1.0      # turn_tilt.c:120-123
        target *= min(1.0 + damper * abs_aggregate / max(self._p("turntilt_yaw_aggregate"), 1.0), 2.0)
        target = _clamp(target, -self._p("turntilt_angle_limit"), self._p("turntilt_angle_limit"))
        if abs(erpm) < self._p("turntilt_start_erpm"):
            return 0.0                                  # turn_tilt.c:132-133
        return target * _sign(erpm)                     # turn_tilt.c:135

    # --- brake tilt (brake_tilt.c:37-97) --------------------------------------
    def _brake_tilt(self, erpm, balance_offset, braking, prev_target=0.0):
        """Return the brake tilt TARGET for this tick (the filter decides the angle).

        `braking` is the firmware's `motor->braking`, which motor_data.c:128-152 defines
        simply as `current < 0` - the caller passes that. It does NOT mean "going backwards"
        (`erpm < -500` is pid.c's *direction* test for the asymmetric brake gains, a
        different thing entirely), and the firmware itself does not trust the flag on its
        own: brake_tilt.c:76-78 says "negative currents alone don't necessarily constitute
        active braking, look at proportional", which is why the balance_offset sign test at
        :78 is what actually decides whether the rider is braking.

        `prev_target` exists because brake_tilt.c:78 has no else: when the outer gate passes
        but the rider is leaning WITH the brake, the firmware leaves `bt->target` where it was
        and the filter keeps ramping toward the stale value. Returning 0.0 there — the obvious
        reading — would release the nose early, which is the one part of a brake the rider
        feels.
        """
        if self.brake_factor >= 0:
            return 0.0                                  # brake_tilt.c:38-39 -> outer else :93
        if not (braking and abs(erpm) > 2000.0):
            return 0.0                                  # brake_tilt.c:75
        if _sign(balance_offset) == _sign(erpm):
            return prev_target                          # brake_tilt.c:78, no else
        # brake_tilt.c:79-90: braking on a downhill should NOT lift the nose as much, because
        # gravity is already doing the slowing and a lift there just pitches the rider into the
        # turn. The signal that says "downhill" is ATR's accel_diff — braking hard, wheel not
        # slowing as fast as the current says it should. It reads atr->accel_diff directly (not
        # the filtered angle), and ATR runs first (main.c:933 before :934), so this is the same
        # tick's estimate. Note it is gated on the SIGNED erpm at +-1000, not abs_erpm: the two
        # branches are forward-downhill and reverse-downhill, not "fast".
        downhill_damper = 1.0
        if (erpm > 1000.0 and self.atr.accel_diff < -1.0) or (
            erpm < -1000.0 and self.atr.accel_diff > 1.0
        ):
            downhill_damper += abs(self.atr.accel_diff) / 2.0      # brake_tilt.c:84
        target = balance_offset / self.brake_factor / downhill_damper   # brake_tilt.c:86
        if downhill_damper > 2.0:
            # brake_tilt.c:87-90, "steep downhills, we don't enable this feature at all!"
            target = 0.0
        return target

    def update(
        self,
        dt,
        *,
        erpm,
        forward_speed,
        torque_nm,
        balance_offset,
        yaw_rate_dps=None,
        battery_volt=None,
        duty=None,
        braking=None,
        now=0.0,
        rider_input=None,
        darkride=False,
    ):
        """Advance one tick. Returns the contributions (degrees) and the total the
        controller should add to its setpoint.

        `rider_input` is the remote stick in [-1, 1] (remote.c's `value`, before the deadband —
        the ported stage does that remap itself, so a caller that pre-clipped would double-count).
        `darkride` flips the trim's sign, because in the dark the board is meant to lean the other
        way (remote.c:151-153). Both default to nothing, and with `inputtilt_remote_type` 0 — the
        shipped default, no stick fitted — the whole stage stays at zero."""
        # Direction the rider is being pushed: a tiltback leans you BACKWARD while you are
        # going forward, so that leaning is what slows you. Expressed against the physical
        # direction of travel rather than the erpm sign, because this model's gear may flip
        # that sign and the rider's spine does not care about encoder conventions. The pin
        # in main() asserts exactly this, so a sign error here cannot pass quietly.
        self._track_step_frequency(dt)
        dir_sign = 1.0 if forward_speed >= 0 else -1.0
        # motor->forward, the direction of travel the filter asks for at :78 of its own file.
        forward = forward_speed >= 0

        # motor_data.c:147 + :122: ATR's measurement input, built here because it is the
        # motor's, not ATR's — d(erpm)/dt from the RAW encoder reading (not this model's
        # velocity state), smoothed by an 8 Hz sliding average. Smoothing matters: ATR prices
        # the gap between this and the current it is pulling, so an unsmoothed derivative turns
        # encoder quantisation into fake hill-lean.
        self.acc_sma.update((erpm - self._last_erpm) / dt if dt > 0 else 0.0)
        self._last_erpm = erpm

        # Tiltback and noseangling ramp through rate_limitf (main.c:914-918, main.c:690-692),
        # NOT through SmoothSetpoint — a linear limit at tiltback_*_speed / noseangling_speed
        # deg/s is the faithful mirror here, not a shortcut. Verified against the source in
        # NEXT 354; see the module docstring before "improving" either of these two.
        wanted, ramp, which = self._tiltback_wanted(duty, battery_volt, forward_speed, dt, now)
        wanted *= dir_sign
        rate = ramp if wanted != 0 else self._p("tiltback_return_speed")
        self.tiltback_target = _ramp_toward(self.tiltback_target, wanted, rate, dt)

        nn = self._noseangling_wanted(erpm)
        self.noseangling = _ramp_toward(
            self.noseangling, nn, self._p("noseangling_speed"), dt
        )

        # --- wheelslip (main.c:483-509) -------------------------------------------------
        # The firmware's detector, in order: the wheel is accelerating HARD in the direction
        # it is already going, at real duty, above walking speed. That is the contact patch
        # letting go (main.c:339 calls the confirmation "exorcist wiggle"), and four
        # behaviours then WIND DOWN rather than freeze or snap — torque_tilt.c:54, atr.c:77,
        # brake_tilt.c:69, turn_tilt.c:93. The one worth thinking about is ATR: a slipping
        # wheel is exactly "torque demanded, wheel not answering", so ATR would lean into a
        # loss of traction and the tilt modifiers would hold that lean for the whole slide.
        #
        # What is NOT ported, named rather than faked: this plant's wheel-deck contact is
        # rigid Coulomb, so a slide is possible but a RIDING one is not yet: NEXT 361's grip ladder
        # measured 26-37 m/s of wheel-versus-deck speed gap at mu 0.04-0.12 and every one of those
        # runs fell, which is the wheel running away from the ground, not from the rider. The flag
        # is earned
        # the same signals the firmware reads — an acceleration filter, erpm, and the injected
        # duty (injected the way yaw and battery voltage are, because a 1-D sim cannot
        # generate them) — instead of being declared by a scenario. The contact's own friction is what decides whether the wheel can push the deck at all (NEXT 361), and the scenario only chooses the surface. `self.acc_sma` stands in for
        # motor_data's acceleration EMA: the enter/exit thresholds (10000 / 7000) are an
        # order of magnitude above anything riding produces — a 28 N*m spin-up in free air is
        # ~77000 erpm/s, which sim/model.py's own selftest measures — so the filter choice
        # does not decide the verdict, and the pins below assert the margin rather than
        # assuming it.
        acc = self.acc_sma.value
        if (abs(acc) > 10000.0                             # main.c:485
                and _fw_sign(acc) == _fw_sign(erpm)        # main.c:486
                and duty is not None and duty > 0.3        # main.c:487
                and abs(erpm) > 2000.0):                   # main.c:489
            self.wheelslip = True
            self._slip_age = 0.0                           # main.c:492 timer_refresh
        elif self.wheelslip:
            # main.c:496-507, and the detail that makes it a hold rather than a lag: while duty
            # is at the ceiling the timer is REFRESHED, so a board that is still spinning does
            # not start counting down. Only after duty drops does the 0.2 s run, and the flag
            # clears at the end of it "to avoid any overreactions".
            if duty is not None and duty > 0.95:           # main.c:502-503 timer_refresh
                self._slip_age = 0.0
            else:
                self._slip_age += dt                       # main.c:504
                if self._slip_age >= 0.2:                  # main.c:504 timer_older(..., 0.2)
                    self.wheelslip = False
        slip = self.wheelslip

        # --- the three tilts run through the firmware's filter, not a rate limiter ----
        # torque_tilt.c:51-69: no early return anywhere, the filter always runs — except
        # under wheelslip (torque_tilt.c:52-56), which winds the value down and returns.
        if slip:
            self.torque_filter.winddown()                  # torque_tilt.c:54
        else:
            self.torque_filter.update(self._torque_tilt_wanted(torque_nm), forward, 1.0, dt)
        self.torque_tilt_out = self.torque_filter.value

        # atr.c:74-162, at main.c:933 — after torque tilt, BEFORE brake tilt, which reads its
        # raw accel_diff. `torque_nm` is the same signal the torque tilt consumed above: the
        # firmware's motor->torque is the filtered directional current divided by the speed
        # constant (motor_data.c:150-151), which is what this model's torque is.
        if slip:
            self.atr.setpoint.winddown()                   # atr.c:75-79
            # atr_update returns BEFORE its filter runs, so the ATR value is shaved by the
            # winddown alpha and its accel_diff goes stale (nothing recomputes it, which is
            # what the firmware leaves behind too). Note what does NOT stop: the acceleration
            # SMA is fed from the erpm history above this block (line 760), on every tick, so
            # the detector keeps a live acceleration signal through the slide. The firmware's
            # release does not use acceleration anyway (main.c:496-507 is duty + a 0.2 s hold).
        else:
            self.atr.update(
                torque=torque_nm,
                erpm=erpm,
                abs_erpm=abs(erpm),
                erpm_sign=_fw_sign(erpm),   # motor_data.c:129 uses the macro, so 0 erpm is +1
                braking=braking if braking is not None else torque_nm < 0,
                forward=forward,
                acceleration=self.acc_sma.value,
                dt=dt,
            )
        self.atr_out = self.atr.setpoint.value
        self.atr_accel_diff = self.atr.accel_diff

        # turn_tilt.c:89-91: `if (turntilt_strength == 0) return;` comes BEFORE the filter
        # update, so zeroing the strength FREEZES the output wherever it was rather than
        # ramping it to zero. That is what MODE_HANDTEST does to a board mid-turn, and it is
        # the opposite of what a naive port does — the naive version is also the version that
        # looks safe in a table.
        tur = self._turn_tilt(yaw_rate_dps, erpm, dt)
        if tur is None:
            # turn_tilt.c:89-91 returns BEFORE the wheelslip check at :93, so a frozen turn
            # tilt (strength 0, e.g. under MODE_HANDTEST) does not wind down during a slide
            # either. Two early returns, and their order is the behaviour; collapsing them is
            # the kind of tidy-up that quietly changes what a board does when it steps out.
            pass
        elif slip:
            self.turn_filter.winddown()                  # turn_tilt.c:93-96
        else:
            self.turn_filter.update(tur, forward, 1.0, dt)
        self.turn_tilt_out = self.turn_filter.value

        # main.c:922-923, and the order is the point: the remote's trim lands in `setpoint` BEFORE
        # any other tilt runs, so the `setpoint - imu.balance_pitch` that brake_tilt_update is handed
        # at :939 already contains it. Running it after, or leaving it out of the brake term, would
        # quietly un-trim the brake response every time the rider leans in.
        self.input_tilt_out = self.input_tilt.update(rider_input, dt, darkride=darkride)

        if braking is None:
            # Fallback for callers that do not report motor current: motor->braking is
            # simply `current < 0` (motor_data.c:128-152) and torque carries this model's
            # current sign. The controller passes its own current, which is the better signal.
            braking = torque_nm < 0
        # brake_tilt.c:75-94: if the outer gate passes but the rider is leaning WITH the
        # brake (line 78), there is no else — `bt->target` keeps its previous value and the
        # filter keeps ramping toward it. Only the outer gate failing sets target = 0.
        if slip:
            self.brake_filter.winddown()                 # brake_tilt.c:67-71
            # Returns before the target is computed, so `brake_target` KEEPS its value and the
            # tilt ramps back when the slide ends. That is brake_tilt.c:78's stale-target rule
            # arriving from the other direction, and it is why the target is not zeroed here.
        else:
            self.brake_target = self._brake_tilt(erpm, balance_offset + self.input_tilt_out,
                                                 braking, self.brake_target)
            self.brake_filter.update(self.brake_target, forward, 1.0, dt)
        self.brake_tilt_out = self.brake_filter.value

        # main.c:947-956, the aggregation nobody reads twice: torque tilt and (ATR + brake
        # tilt) do not simply add. Same sign, the larger magnitude wins; opposite signs, they
        # add. ATR's term is real now (atr.c, ported in NEXT 349), which is what makes the
        # max() branch meaningful: at zero it was an identity wearing a rule's clothes.
        ab_offset = self.atr_out + self.brake_tilt_out
        if _sign(ab_offset) == _sign(self.torque_tilt_out):
            torque_combined = _sign(ab_offset) * max(abs(ab_offset), abs(self.torque_tilt_out))
        else:
            torque_combined = ab_offset + self.torque_tilt_out

        total = (
            self.tiltback_target
            + self.noseangling
            + self.turn_tilt_out
            + torque_combined
            # main.c:923 `setpoint += d->remote.setpoint.value`. Its own term, not folded into the
            # torque aggregation below: the remote is the rider asking for a posture, and the
            # max()/add rule at :950-955 exists to resolve competing PROTECTIONS, not the rider.
            + self.input_tilt_out
        )
        self.active = which
        return {
            "setpoint_offset": total,
            "tiltback_deg": self.tiltback_target,
            "noseangling_deg": self.noseangling,
            "torquetilt_deg": self.torque_tilt_out,
            "turntilt_deg": self.turn_tilt_out,
            "braketilt_deg": self.brake_tilt_out,
            # Reported, not assumed: a probe has to be able to tell "no slip" from "slip, but the
            # behaviour that should react to it did not" (main.c:483-509).
            "wheelslip": 1.0 if self.wheelslip else 0.0,
            "atr_deg": self.atr_out,
            # Reported so a probe can tell "the rider never touched the stick" (0.0 with
            # inputtilt_remote_type 0) apart from "the trim was asked for and did nothing" — the
            # same distinction the wheelslip row above exists for (NEXT 365).
            "inputtilt_deg": self.input_tilt_out,
            "inputtilt_target_deg": self.input_tilt.target,
            "inputtilt_stick": self.input_tilt.input,
            "tiltback_which": which,
            "yaw_aggregate": self.yaw_aggregate,
            # Reported so a probe can pin the cause and not just the effect: accel_diff is the
            # one number that says WHY ATR did or did not lean.
            "atr_accel_diff": self.atr_accel_diff,
        }


def _ramp_toward(cur, target, speed_deg_per_s, dt):
    """`rate_limitf` (lib/utils.c:26) — the firmware's linear rate limiter, saturating at the
    step. Used by the tiltbacks (main.c:914-918) and noseangling (main.c:690-692), and by
    nothing else. The three *tilts* use SmoothSetpoint instead; do not route these through it,
    the difference is what NEXT 354 established."""
    if speed_deg_per_s <= 0:
        return target
    step = speed_deg_per_s * dt
    if abs(target - cur) <= step:
        return target
    return cur + math.copysign(step, target - cur)


# --- pins ---------------------------------------------------------------------
def _pins(verbose=True) -> list:
    """Assertions that would catch the mistakes this port is most likely to contain.

    Run by `python -m sim.protections`; a failure exits non-zero, because a protection
    that silently leans the wrong way is not a protection.
    """
    from .control import Params

    fails = []

    def check(name, ok, detail=""):
        if verbose:
            print("  %-58s %s%s" % (name, "ok" if ok else "FAIL", ("  " + detail) if detail else ""))
        if not ok:
            fails.append(name)

    dt = 1.0 / 1000.0

    # 1. Sign: over-speed tiltback must lean the rider OPPOSITE the direction of travel,
    #    i.e. the offset it adds has to be the one that decelerates. Convention-free: a
    #    flipped sign here throws the rider forward at speed and looks completely normal
    #    in a plot, which is the failure mode this whole file is most afraid of.
    pr = Protections(Params(tiltback_speed=3.0, tiltback_duty_angle=5.0,
                            tiltback_duty_speed=20.0, tiltback_return_speed=20.0))
    r = None
    for _ in range(400):
        r = pr.update(dt, erpm=9000.0, forward_speed=6.0, torque_nm=5.0, balance_offset=0.0)
    check("over-speed tiltback leans backward while moving forward",
          r["tiltback_deg"] > 1.0, "got %+.2f deg" % r["tiltback_deg"])
    check("...and the total offset opposes travel",
          r["setpoint_offset"] * 6.0 >= 0.0,
          "offset %+.2f while moving at %+.1f m/s" % (r["setpoint_offset"], 6.0))
    pr = Protections(Params(tiltback_speed=3.0, tiltback_duty_angle=5.0,
                            tiltback_duty_speed=20.0, tiltback_return_speed=20.0))
    for _ in range(400):
        r = pr.update(dt, erpm=-9000.0, forward_speed=-6.0, torque_nm=-5.0, balance_offset=0.0)
    check("...and leans forward while reversing",
          r["tiltback_deg"] < -1.0, "got %+.2f deg" % r["tiltback_deg"])

    # 2. Ramp: the angle must arrive over time, not in one tick, and return at
    #    tiltback_return_speed. main.c:273-286 picks a different speed per state.
    pr = Protections(Params(tiltback_speed=3.0, tiltback_duty_angle=5.0,
                            tiltback_duty_speed=1.0, tiltback_return_speed=5.0))
    first = pr.update(dt, erpm=9000.0, forward_speed=6.0, torque_nm=0.0, balance_offset=0.0)
    check("tiltback ramps instead of snapping (no rider ejected in one tick)",
          abs(first["tiltback_deg"]) < 0.01, "first tick %+.4f deg" % first["tiltback_deg"])
    settled = None
    for _ in range(6000):
        settled = pr.update(dt, erpm=9000.0, forward_speed=6.0, torque_nm=0.0, balance_offset=0.0)
    check("tiltback reaches tiltback_duty_angle",
          abs(settled["tiltback_deg"] - 5.0) < 0.05, "got %.2f" % settled["tiltback_deg"])
    back = None
    for _ in range(6000):
        back = pr.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0, balance_offset=0.0)
    check("tiltback returns to level (return_speed path)",
          abs(back["tiltback_deg"]) < 0.05, "got %.2f" % back["tiltback_deg"])

    # 3. Torque tilt: nothing below the start current, and the regen strength is a
    #    separate knob (torque_tilt.c:60) - a port that reads it once passes these.
    pr = Protections(Params(torquetilt_strength=0.5, torquetilt_start_current=15.0,
                            torquetilt_angle_limit=8.0, torquetilt_strength_regen=2.0))
    low = None
    for _ in range(3000):
        low = pr.update(dt, erpm=1000.0, forward_speed=1.0,
                        torque_nm=15.0 * TORQUE_CONSTANT_COMPAT, balance_offset=0.0)
    check("torque tilt is silent below torquetilt_start_current",
          abs(low["torquetilt_deg"]) < 1e-6, "got %.4f" % low["torquetilt_deg"])
    hi = None
    for _ in range(4000):
        hi = pr.update(dt, erpm=1000.0, forward_speed=1.0,
                       torque_nm=45.0 * TORQUE_CONSTANT_COMPAT, balance_offset=0.0)
    check("torque tilt leans with drive current", hi["torquetilt_deg"] > 0.5,
          "got %.2f (expect (45-15)*0.5=15 clamped to 8)" % hi["torquetilt_deg"])
    check("torque tilt clamps at torquetilt_angle_limit",
          abs(hi["torquetilt_deg"] - 8.0) < 0.05, "got %.2f" % hi["torquetilt_deg"])
    rg = None
    for _ in range(4000):
        rg = pr.update(dt, erpm=-1000.0, forward_speed=-1.0,
                       torque_nm=-45.0 * TORQUE_CONSTANT_COMPAT, balance_offset=0.0)
    check("regen side uses its own strength and sign",
          rg["torquetilt_deg"] < -3.0, "got %.2f (regen strength 2.0 -> clamped -8)" % rg["torquetilt_deg"])

    # 4. Turn tilt needs BOTH a rate and a sustained aggregate (turn_tilt.c:104): a wiggle
    #    over pavement must not tilt the rider, and a steady lean at zero erpm must not
    #    either (turn_tilt.c:132-133).
    pr = Protections(Params(turntilt_strength=40.0, turntilt_start_angle=5.0,
                            turntilt_start_erpm=2000.0, turntilt_angle_limit=3.0,
                            turntilt_erpm_boost=200.0, turntilt_erpm_boost_end=5000.0,
                            turntilt_yaw_aggregate=90.0))
    wig = None
    for _ in range(2000):
        wig = pr.update(dt, erpm=5000.0, forward_speed=6.0, torque_nm=0.0, balance_offset=0.0,
                        yaw_rate_dps=10.0 * math.sin(_ * dt * 4.0))
    check("turn tilt ignores wiggling (rate under 30 deg/s)",
          abs(wig["turntilt_deg"]) < 1e-6, "got %.4f" % wig["turntilt_deg"])
    slow = None
    for _ in range(4000):
        slow = pr.update(dt, erpm=500.0, forward_speed=0.4, torque_nm=0.0, balance_offset=0.0,
                         yaw_rate_dps=60.0)
    check("turn tilt off below turntilt_start_erpm",
          abs(slow["turntilt_deg"]) < 1e-6, "got %.4f" % slow["turntilt_deg"])
    turn = None
    for _ in range(6000):
        turn = pr.update(dt, erpm=5000.0, forward_speed=6.0, torque_nm=0.0, balance_offset=0.0,
                         yaw_rate_dps=60.0)
    check("turn tilt engages in a sustained turn", turn["turntilt_deg"] > 0.5,
          "got %.2f (aggregate %.0f)" % (turn["turntilt_deg"], turn["yaw_aggregate"]))
    check("turn tilt clamps at turntilt_angle_limit",
          turn["turntilt_deg"] <= 3.0 + 1e-6, "got %.2f" % turn["turntilt_deg"])
    noneed = Protections(Params(turntilt_strength=40.0, turntilt_start_angle=5.0,
                               turntilt_start_erpm=2000.0, turntilt_angle_limit=3.0))
    noyaw = noneed.update(dt, erpm=5000.0, forward_speed=6.0, torque_nm=0.0, balance_offset=0.0)
    check("turn tilt stays inactive when yaw is not injected",
          noyaw["turntilt_deg"] == 0.0)

    # 5. MODE_HANDTEST (main.c:1481-1497) must silence every modifier at once, including
    #    ATR's own strengths. This is the coupling that a per-behaviour test never sees:
    #    the note in my port spec attributed it to ATR, which was wrong - handtest is the
    #    mode that disables ATR, not the other way round.
    over = Params(torquetilt_strength=0.5, torquetilt_start_current=5.0,
                  torquetilt_angle_limit=8.0, torquetilt_strength_regen=0.5,
                  turntilt_strength=40.0, turntilt_start_angle=5.0, turntilt_start_erpm=2000.0,
                  turntilt_angle_limit=3.0, turntilt_erpm_boost=200.0,
                  turntilt_erpm_boost_end=5000.0, turntilt_yaw_aggregate=90.0,
                  tiltback_constant=3.0, tiltback_constant_erpm=100.0,
                  tiltback_variable=0.5, tiltback_variable_max=5.0,
                  tiltback_speed=3.0, tiltback_duty_angle=5.0)
    live = Protections(over)
    for _ in range(6000):
        live_out = live.update(dt, erpm=9000.0, forward_speed=6.0, torque_nm=40.0,
                               balance_offset=0.0, yaw_rate_dps=60.0)
    check("with everything on, the setpoint is being moved",
          abs(live_out["setpoint_offset"]) > 3.0,
          "got %.2f" % live_out["setpoint_offset"])
    ht = Protections(over, handtest=True)
    for _ in range(6000):
        ht_out = ht.update(dt, erpm=9000.0, forward_speed=6.0, torque_nm=40.0,
                           balance_offset=0.0, yaw_rate_dps=60.0)
    # MODE_HANDTEST zeroes ki, both brake scales, both booster angles, torque tilt (both
    # strengths), ATR's two strengths, turn tilt, and the tiltback *variable/constant*
    # modifiers (main.c:1484-1495, repeated at 1935-1953). What it pointedly does NOT zero
    # is tiltback_speed / tiltback_duty_angle / the voltage tiltbacks: a board on the bench
    # with a hand on it should still push your foot down before it runs away. My port spec
    # claimed ATR was the mode that disabled the tilts; reading the block said otherwise -
    # handtest disables ATR, along with everything else. So the pin asserts BOTH halves:
    # modifiers silenced, hard tiltbacks alive. An assertion that only checked "handtest
    # quieted it" would have passed on a board that had lost its runaway protection.
    check("MODE_HANDTEST silences the tune modifiers",
          abs(ht_out["torquetilt_deg"]) < 1e-9 and abs(ht_out["turntilt_deg"]) < 1e-9
          and abs(ht_out["noseangling_deg"]) < 1e-9,
          "tt=%.4f tur=%.4f nn=%.4f"
          % (ht_out["torquetilt_deg"], ht_out["turntilt_deg"], ht_out["noseangling_deg"]))
    check("...and silences all three of them, not just one",
          abs(ht_out["torquetilt_deg"] + ht_out["turntilt_deg"] + ht_out["noseangling_deg"])
          < 1e-9)
    check("over-speed tiltback SURVIVES handtest (main.c:1484-1495 does not zero it)",
          ht_out["tiltback_deg"] > 1.0, "got %.2f" % ht_out["tiltback_deg"])
    check("so what remains in handtest is exactly the hard tiltback",
          abs(ht_out["setpoint_offset"] - ht_out["tiltback_deg"]) < 1e-9,
          "total %.2f vs tiltback %.2f"
          % (ht_out["setpoint_offset"], ht_out["tiltback_deg"]))

    # 5b. Brake tilt (brake_tilt.c:37-97): the lift that makes hard braking feel like the
    #     nose coming up. Needs braking, speed, AND the rider actually demanding the brake
    #     (sign of the proportional term) - three gates, and each one has been dropped in a
    #     rewrite at some point, which is why all four cases are pinned.
    #    A word on `atr_amps_decel_ratio` here and in the brake pins below: these pins test
    #    brake tilt, but ATR now runs in the same call and reads the *injected* (braking=True,
    #    positive torque) pair as "current flowing, wheel not slowing" — i.e. a downhill — and
    #    its damper then divides brake tilt by >2 (measured: accel_diff 2.03 -> damper 2.01 ->
    #    target 0). That is the firmware behaving correctly on an inconsistent bench state, and
    #    the damper has its own pins below. Loosening the ratio here keeps ATR out of a test
    #    that is not about ATR; it does not touch what any of these pins assert.
    btkw = dict(braketilt_strength=10.0, braketilt_lingering=2.0,  # factor = -(0.5+2) = -2.5
                atr_amps_decel_ratio=40.0)
    # one instance, ramped to steady state
    bp = Protections(Params(**btkw))
    for _ in range(4000):
        brk = bp.update(dt, erpm=-5000.0, forward_speed=-3.0, torque_nm=5.0,
                        balance_offset=2.0, braking=True)
    check("brake tilt lifts the nose against the brake demand",
          brk["braketilt_deg"] < 0.0, "offset %+.1f -> %+.2f deg (factor -2.5)"
          % (2.0, brk["braketilt_deg"]))
    check("brake tilt magnitude is offset/factor, not offset",
          abs(brk["braketilt_deg"] - (-2.0 / 2.5)) < 0.05, "got %.3f" % brk["braketilt_deg"])
    nb = Protections(Params(**btkw))
    for _ in range(4000):
        no_brk = nb.update(dt, erpm=-5000.0, forward_speed=-3.0, torque_nm=5.0,
                           balance_offset=2.0, braking=False)
    check("no brake tilt without braking", abs(no_brk["braketilt_deg"]) < 1e-9,
          "got %.4f" % no_brk["braketilt_deg"])
    slow_brk = Protections(Params(**btkw))
    for _ in range(4000):
        sl = slow_brk.update(dt, erpm=-800.0, forward_speed=-0.5, torque_nm=5.0,
                             balance_offset=2.0, braking=True)
    check("no brake tilt below 2000 erpm (brake_tilt.c:75)",
          abs(sl["braketilt_deg"]) < 1e-9, "got %.4f" % sl["braketilt_deg"])
    same_brk = Protections(Params(**btkw))
    for _ in range(4000):
        sm = same_brk.update(dt, erpm=-5000.0, forward_speed=-3.0, torque_nm=5.0,
                             balance_offset=-2.0, braking=True)
    check("no brake tilt when the rider is leaning WITH the brake (brake_tilt.c:78)",
          abs(sm["braketilt_deg"]) < 1e-9, "got %.4f" % sm["braketilt_deg"])
    off_brk = Protections(Params())      # braketilt_strength default 0 => factor 0
    for _ in range(4000):
        ob = off_brk.update(dt, erpm=-5000.0, forward_speed=-3.0, torque_nm=5.0,
                            balance_offset=2.0, braking=True)
    check("braketilt_strength 0 means the feature is off, not weak",
          abs(ob["braketilt_deg"]) < 1e-9, "got %.4f" % ob["braketilt_deg"])

    # 6. Injected-input honesty: no voltage injected => no voltage tiltback, ever.
    pr = Protections(Params(tiltback_lv=4.0, tiltback_lv_angle=10.0, tiltback_hv=42.0,
                            tiltback_hv_speed=2.0, tiltback_lv_speed=2.0))
    nv = None
    for _ in range(2000):
        nv = pr.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0, balance_offset=0.0)
    check("no injected battery voltage => no voltage tiltback",
          nv["tiltback_which"] == "", "which=%r" % nv["tiltback_which"])
    lv = None
    for _ in range(6000):
        lv = pr.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0, balance_offset=0.0,
                       battery_volt=3.5)
    check("injected sag below tiltback_lv does tilt (and uses the HV ramp speed)",
          lv["tiltback_which"] == "lv" and abs(lv["tiltback_deg"]) > 1.0,
          "which=%r deg=%.2f" % (lv["tiltback_which"], lv["tiltback_deg"]))

    # 6b. NEXT 356: the per-cell multiply the firmware applies BEFORE either voltage threshold
    #     is compared (motor_data.c:79-91). settings.xml ships tiltback_lv=3.0 and
    #     tiltback_hv=4.3, and those are PER-CELL volts — its help text says "recommended
    #     4.3V, which is now default" — scaled by VESC's si_battery_cells unless the stored
    #     number is already 10 V or more. Compare them raw and a healthy pack satisfies
    #     `>= 4.3` forever. The first pin below is the one that measured the bug: on the
    #     shipped defaults a 42 V pack sat in HV tiltback at 8 degrees from the first tick,
    #     which also meant every LV test after it was reading the wrong branch.
    healthy = Protections(Params())
    row = None
    for _ in range(400):
        row = healthy.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0,
                             balance_offset=0.0, battery_volt=42.0)
    check("a 12S pack at 42 V on the shipped per-cell defaults is NOT in voltage tiltback",
          row["tiltback_which"] == "",
          "which=%r deg=%.2f" % (row["tiltback_which"], row["tiltback_deg"]))

    p12 = Protections(Params(si_battery_cells=12))
    p6 = Protections(Params(si_battery_cells=6))
    pack12 = None
    pack6 = None
    for _ in range(4000):
        pack12 = p12.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0,
                            balance_offset=0.0, battery_volt=30.0)
        pack6 = p6.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0,
                          balance_offset=0.0, battery_volt=30.0)
    check("one 30 V pack is LOW voltage on a 12S board and HIGH voltage on a 6S one",
          pack12["tiltback_which"] == "lv" and pack6["tiltback_which"] == "hv",
          "12S=%r 6S=%r — the same volts, opposite protections"
          % (pack12["tiltback_which"], pack6["tiltback_which"]))

    p_abs = Protections(Params(tiltback_hv=51.6))     # a tuner who typed an absolute number
    abs_hv = None
    for _ in range(400):
        abs_hv = p_abs.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0,
                              balance_offset=0.0, battery_volt=52.0)
    check("a threshold of 10 V or more is left alone (motor_data.c:85), not multiplied again",
          abs_hv["tiltback_which"] == "hv",
          "which=%r — multiplying would have moved the threshold to 619 V and never fired"
          % (abs_hv["tiltback_which"],))

    p_nocells = Protections(Params(si_battery_cells=0))   # firmware that reports no count
    nocells = None
    for _ in range(4000):
        nocells = p_nocells.update(dt, erpm=0.0, forward_speed=0.0, torque_nm=0.0,
                                   balance_offset=0.0, battery_volt=2.9)
    check("si_battery_cells 0 uses the stored number as-is rather than multiplying by zero",
          nocells["tiltback_which"] == "lv", "which=%r" % (nocells["tiltback_which"],))

    want_lv, want_hv = Protections(Params())._lv_hv_thresholds()   # the helper, on purpose
    check("the per-cell rule lands exactly where motor_data.c:82-86 says",
          abs(want_lv - 36.0) < 1e-9 and abs(want_hv - 51.6) < 1e-9,
          "got lv=%.3f hv=%.3f, expected 36.000 and 51.600" % (want_lv, want_hv))

    # 7-11. The filter itself (NEXT 354). These are the pins that keep SmoothSetpoint a port
    #     rather than a vibe: a linear limiter passes every steady-state pin above, because it
    #     lands on the same angle — what it gets wrong is the part the rider feels.
    f = SmoothSetpoint(1000.0)
    f.configure(0.2, 0.08, 0.16, 0.2, 100.0, 100.0, 100.0, 100.0)
    # Hand-computed, two frequencies, so neither a lost 2*pi nor a misplaced 2.146 passes:
    # omega = min(1/(tc*f), 0.5); alpha_base = omega - omega^2/2 (ema.c:24-34).
    # tc 0.2 @ 1000 Hz -> omega 0.005            -> 0.0049875,  x2.146 -> 0.01070319
    # tc 0.2 @  500 Hz -> omega 0.01             -> 0.00995,    x2.146 -> 0.02135270
    # tc 0.08 @ 1000 Hz -> omega 0.0125          -> 0.01242188  (NO 2.146: it is not 2nd order)
    check("alpha comes from the time constant, 2.146 second-order factor included",
          abs(f.alpha - 0.010703190) < 1e-6, "alpha=%.6f" % f.alpha)
    g500 = SmoothSetpoint(500.0)
    g500.configure(0.2, 0.08, 0.16, 0.2, 100.0, 100.0, 100.0, 100.0)
    check("...and it halves the raw alpha when the loop slows down",
          abs(g500.alpha - 0.021352700) < 1e-6, "alpha=%.6f" % g500.alpha)
    check("the speed alphas are first-order: no 2.146 on them (smooth_setpoint.c:59-61)",
          abs(f.on_speed_alpha - 0.012421875) < 1e-6, "on_speed_alpha=%.6f" % f.on_speed_alpha)
    check("a zero time constant saturates at alpha 0.375 instead of dividing by zero",
          abs(_ema_alpha_time_constant(0.0, 500.0) - 0.375) < 1e-9)

    f.reset()
    first = f.update(10.0, True, 1.0, dt)          # one tick toward a 10 deg demand
    check("it is NOT a rate limiter: the first tick is far under limit*dt",
          abs(first) < 0.01, "first tick %.5f deg (a limiter would give 0.100)" % first)
    settled = first
    for _ in range(5000):
        settled = f.update(10.0, True, 1.0, dt)
    check("...and it still gets there", abs(settled - 10.0) < 0.5, "settled %.2f deg" % settled)

    # on/off asymmetry: the release limit is the release limit, both ways round.
    g = SmoothSetpoint(1000.0)
    g.configure(0.2, 0.08, 0.16, 0.2, 40.0, 2.0, 40.0, 2.0)   # grow 40 deg/s, release 2 deg/s
    up = 0.0
    for _ in range(500):
        up = g.update(10.0, True, 1.0, dt)
    down = up
    for _ in range(500):
        down = g.update(0.0, True, 1.0, dt)
    check("growing uses on_speed and shrinking uses off_speed (asymmetric by design)",
          up > 8.0 and down > 0.5, "up %.2f, still %.2f after the same time releasing" % (up, down))

    # crossing zero gets max(on, off) — smoother.c:96-97, the case a two-speed limiter misses.
    h = SmoothSetpoint(1000.0)
    h.configure(0.2, 0.08, 0.16, 0.2, 40.0, 2.0, 40.0, 2.0)
    for _ in range(800):
        h.update(5.0, True, 1.0, dt)
    cross = h.value
    for _ in range(300):
        cross = h.update(-5.0, True, 1.0, dt)
    slow = SmoothSetpoint(1000.0)
    slow.configure(0.2, 0.08, 0.16, 0.2, 40.0, 2.0, 40.0, 2.0)
    for _ in range(800):
        slow.update(5.0, True, 1.0, dt)
    relax = slow.value
    for _ in range(300):
        relax = slow.update(0.0, True, 1.0, dt)
    check("crossing zero is faster than releasing (speed_limit = max of the two)",
          cross < relax - 1.0, "crossed to %+.2f vs released to %+.2f" % (cross, relax))

    f.reset()
    for _ in range(5000):
        f.update(10.0, True, 1.0, dt)
    before = f.value
    f.winddown()
    dropped = f.value
    f.update(10.0, True, 1.0, dt)
    check("winddown shaves the value by winddown_alpha and re-seeds on the next tick",
          abs(dropped - before * (1.0 - f.winddown_alpha)) < 1e-9 and not f.is_winddown,
          "%.4f -> %.4f" % (before, dropped))

    # 12. MODE_HANDTEST on turn tilt FREEZES it (turn_tilt.c:89-91 returns before the filter).
    #     Zeroing it instead — which is what the obvious port does — is not neutral: it silently
    #     removes a tilt the board would still be holding.
    turn_params = Params(turntilt_strength=30.0, turntilt_start_angle=1.0,
                         turntilt_start_erpm=1000.0, turntilt_angle_limit=3.0,
                         turntilt_yaw_aggregate=600.0, turntilt_erpm_boost=0.0,
                         turntilt_erpm_boost_end=10000.0)
    pr = Protections(turn_params)
    tr = None
    for _ in range(1500):
        tr = pr.update(dt, erpm=5000.0, forward_speed=5.0, torque_nm=0.0, balance_offset=0.0,
                       yaw_rate_dps=55.0)
    frozen_at = tr["turntilt_deg"]
    pr.handtest = True                                  # what the firmware's cmd_handtest does
    after = None
    for _ in range(1000):
        after = pr.update(dt, erpm=5000.0, forward_speed=5.0, torque_nm=0.0,
                          balance_offset=0.0, yaw_rate_dps=55.0)
    check("turn tilt HOLDS its angle under MODE_HANDTEST rather than releasing",
          frozen_at > 0.5 and abs(after["turntilt_deg"] - frozen_at) < 1e-9,
          "held %+.3f deg for 1 s of handtest" % after["turntilt_deg"])

    # 13. brake_tilt.c:78 has no else: lean WITH the brake and the target goes stale, it does
    #     not drop to zero. The stale value is the "lingering" in the feature's own comment.
    brake_params = Params(braketilt_strength=10.0, braketilt_lingering=1.0)
    pr = Protections(brake_params)
    early = None
    for _ in range(3):
        early = pr.update(dt, erpm=5000.0, forward_speed=5.0, torque_nm=-10.0,
                          balance_offset=-2.0, braking=True)
    climbing = early["braketilt_deg"]
    later = None
    for _ in range(4000):
        later = pr.update(dt, erpm=5000.0, forward_speed=5.0, torque_nm=-10.0,
                          balance_offset=+2.0, braking=True)   # rider leans WITH the brake
    # The test is "it finished the journey to the target it was already going to", not
    # "it moved": offset -2.0 over factor -2.5 is +0.80 deg, so a port that releases on the
    # closed gate lands on 0.0 and a port that holds lands on 0.80. Compare to the number,
    # not to the previous tick — a relative check goes flaky the moment the filter is fast.
    # 4000 ticks, not 400: brake tilt's filter is configured with atr's 0.3 s time constant,
    # so 0.4 s only gets to 0.8*(1-e^-1.33) = 0.60, which is the filter behaving CORRECTLY
    # and a pin too impatient to notice (measured, not guessed).
    check("brake tilt keeps ramping to its stale target when the rider leans with the brake",
          abs(later["braketilt_deg"] - 0.80) < 0.03 and climbing < 0.1,
          "%.3f -> %.3f deg, expected the stale target +0.80" % (climbing, later["braketilt_deg"]))

    # 14. main.c:947-956: torque tilt and brake tilt do not simply add.
    both = Params(braketilt_strength=10.0, braketilt_lingering=1.0,
                  torquetilt_strength=1.0, torquetilt_strength_regen=1.0,
                  torquetilt_start_current=0.0, torquetilt_angle_limit=8.0)
    pr = Protections(both)
    r = None
    for _ in range(4000):
        r = pr.update(dt, erpm=5000.0, forward_speed=5.0, torque_nm=45.0,
                      balance_offset=-2.0, braking=True)        # both tilt the same way
    check("same sign: the larger of torque tilt and brake tilt wins, they do not add",
          r["torquetilt_deg"] > 0 and r["braketilt_deg"] > 0
          and abs(r["setpoint_offset"] - max(r["torquetilt_deg"], r["braketilt_deg"])) < 1e-9,
          "torque %+.2f brake %+.2f total %+.2f"
          % (r["torquetilt_deg"], r["braketilt_deg"], r["setpoint_offset"]))
    pr = Protections(both)
    r = None
    for _ in range(4000):
        r = pr.update(dt, erpm=5000.0, forward_speed=5.0, torque_nm=-45.0,
                      balance_offset=-2.0, braking=True)        # torque tilts the other way
    check("opposite signs: they add (so a cancellation has to be earned, not assumed)",
          r["torquetilt_deg"] < 0 and r["braketilt_deg"] > 0
          and abs(r["setpoint_offset"] - (r["torquetilt_deg"] + r["braketilt_deg"])) < 1e-9,
          "torque %+.2f brake %+.2f total %+.2f"
          % (r["torquetilt_deg"], r["braketilt_deg"], r["setpoint_offset"]))

    # 15. Config fidelity, cheap and boring, but these are the numbers that decide feel and
    #     two of them are hard-coded in the firmware rather than configurable.
    pr = Protections(Params(braketilt_strength=10.0, braketilt_lingering=4.0))
    check("turn tilt's rate limit is the firmware's hard-coded 20 deg/s",
          pr.turn_filter.on_speed_up == 20.0 and pr.turn_filter.on_speed_down == 20.0)
    check("brake tilt's release speed is atr off_speed_limit / braketilt_lingering",
          abs(pr.brake_filter.off_speed_up
              - pr.p["atr.filter.off_speed_limit"] / 4.0) < 1e-9,
          "off_speed %.3f deg/s" % pr.brake_filter.off_speed_up)

    # 7. ATR (atr.c). Every pin here is about the thing a torque-only reading gets wrong: ATR
    #    is the difference between the current you pull and the acceleration it buys, so each
    #    half of that difference has to be pinned separately or the port can pass one test and
    #    still be inventing hill-lean on a flat road.
    #
    #    7a. The SMA window is DERIVED, not chosen (sma.c:26-29). Same 8 Hz cutoff, two
    #    different plants, two different window lengths — hardcoding an n is how a filter
    #    quietly becomes "some smoothing" instead of the firmware's smoothing.
    check("ATR's acceleration SMA window comes from the cutoff (8 Hz @ 500 Hz -> 27)",
          Sma(8.0, 500.0).n == 27, "got n=%d" % Sma(8.0, 500.0).n)
    check("...and grows with the plant rate (8 Hz @ 1 kHz -> 55)",
          Sma(8.0, 1000.0).n == 55, "got n=%d" % Sma(8.0, 1000.0).n)
    run = Sma(4.0, 100.0)
    for _ in range(400):
        run.update(1.0)
    check("SMA converges to a constant input", abs(run.value - 1.0) < 1e-9,
          "got %.6f" % run.value)

    #    7b. The hill case, which is the whole reason the feature exists: current is being
    #    pulled, the wheel is NOT accelerating (constant erpm), so accel_diff is positive and
    #    the nose comes up. Strength is raised from the 1.0 default so the lean clears the
    #    1.5 deg threshold and is measurable rather than merely nonzero.
    atr_kw = dict(atr_strength_up=2.0, atr_strength_down=1.0, atr_threshold_up=1.5,
                  atr_threshold_down=1.5, atr_amps_accel_ratio=9.0, atr_amps_decel_ratio=8.0,
                  atr_angle_limit=8.0, atr_speed_boost=0.0)
    hp = Protections(Params(**atr_kw))
    hill = None
    for _ in range(4000):
        hill = hp.update(dt, erpm=3000.0, forward_speed=2.0, torque_nm=12.0,
                         balance_offset=0.0, braking=False)
    check("ATR leans on a hill load (torque demanded, wheel not accelerating)",
          hill["atr_deg"] > 0.5, "got %.3f deg, accel_diff %.3f"
          % (hill["atr_deg"], hill["atr_accel_diff"]))
    check("...and the lean is below atr_angle_limit, not past it",
          hill["atr_deg"] <= 8.0 + 1e-9, "got %.3f" % hill["atr_deg"])
    #    7c. The cruise case, the pin that the hill case cannot substitute for: at exactly the
    #    8 A holding current (motor_data.c's own offset, atr.c:84) a steady board buys no
    #    acceleration and ATR must stay completely out of it. A port that reads torque alone
    #    leans here, and it looks fine in every other test.
    cp = Protections(Params(**dict(atr_kw, atr_strength_up=5.0)))
    cruise = None
    for _ in range(4000):
        cruise = cp.update(dt, erpm=3000.0, forward_speed=2.0,
                           torque_nm=8 * TORQUE_CONSTANT_COMPAT, balance_offset=0.0,
                           braking=False)
    check("ATR is silent at the holding current (the 8 A offset, atr.c:84)",
          abs(cruise["atr_deg"]) < 1e-6, "got %.4f deg, accel_diff %.4f"
          % (cruise["atr_deg"], cruise["atr_accel_diff"]))
    #    7d. The standstill gate (atr.c:107-117): identical load, under 250 erpm, and accel_diff
    #    is pinned to 0 — ATR is a riding feature, not something that leans you over at a kerb.
    sp = Protections(Params(**atr_kw))
    stood = None
    for _ in range(4000):
        stood = sp.update(dt, erpm=200.0, forward_speed=0.1, torque_nm=12.0,
                          balance_offset=0.0, braking=False)
    check("ATR is gated off below 250 erpm (atr.c:107)",
          abs(stood["atr_deg"]) < 1e-9 and abs(stood["atr_accel_diff"]) < 1e-9,
          "got %.4f deg" % stood["atr_deg"])
    #    7e. Both strengths at 0 => the feature is off, not weak, and off means the filter is
    #    still running toward 0 rather than frozen (unlike turn tilt's early return).
    zp = Protections(Params(**dict(atr_kw, atr_strength_up=0.0, atr_strength_down=0.0)))
    zeroed = None
    for _ in range(4000):
        zeroed = zp.update(dt, erpm=3000.0, forward_speed=2.0, torque_nm=12.0,
                           balance_offset=0.0, braking=False)
    check("atr_strength_up/down both 0 means ATR is off, not weak",
          abs(zeroed["atr_deg"]) < 1e-9, "got %.6f" % zeroed["atr_deg"])
    #    7f. MODE_HANDTEST (main.c:1484-1495) zeroes atr_strength_up/down, so the same board on
    #    the same hill must come back to level. This is the coupling the port spec once
    #    attributed to ATR itself, and it is the reason _p() is read for every ATR parameter.
    ht_atr = Protections(Params(**atr_kw), handtest=True)
    ht_hill = None
    for _ in range(4000):
        ht_hill = ht_atr.update(dt, erpm=3000.0, forward_speed=2.0, torque_nm=12.0,
                                balance_offset=0.0, braking=False)
    check("MODE_HANDTEST silences ATR on that same hill",
          abs(ht_hill["atr_deg"]) < 1e-6 and hill["atr_deg"] > 0.5,
          "handtest %.5f vs live %.3f" % (ht_hill["atr_deg"], hill["atr_deg"]))
    #    7g. The aggregation (main.c:947-956) with ATR actually contributing: same sign as
    #    torque tilt means the LARGER wins, it does not add. With ATR at 0.0 this branch was an
    #    identity; now it has teeth, and a sum-Everything port changes the rider's lean here.
    ag_kw = dict(atr_kw)
    ag_kw.update(torquetilt_strength=1.0, torquetilt_start_current=0.0,
                 torquetilt_angle_limit=8.0, torquetilt_strength_regen=1.0)
    ap = Protections(Params(**ag_kw))
    agg = None
    for _ in range(6000):
        agg = ap.update(dt, erpm=3000.0, forward_speed=2.0, torque_nm=12.0,
                        balance_offset=0.0, braking=False)
    expected = max(abs(agg["atr_deg"]), abs(agg["torquetilt_deg"]))
    shared = agg["tiltback_deg"] + agg["noseangling_deg"] + agg["turntilt_deg"]
    check("same-sign ATR + torque tilt take the larger, they do not add",
          _sign(agg["atr_deg"]) == _sign(agg["torquetilt_deg"])
          and abs((agg["setpoint_offset"] - shared) - expected) < 1e-6,
          "atr %.2f tt %.2f total-shared %.2f"
          % (agg["atr_deg"], agg["torquetilt_deg"], agg["setpoint_offset"] - shared))
    check("the naive sum would have leaned the rider further than the firmware does",
          expected < abs(agg["atr_deg"]) + abs(agg["torquetilt_deg"]) - 1e-6,
          "max %.2f vs sum %.2f" % (expected, abs(agg["atr_deg"]) + abs(agg["torquetilt_deg"])))
    #    7h. The firmware's sign(0) == +1 (lib/utils.h:99) is load-bearing in atr.c:101/:143.
    check("firmware sign(0) is +1 where _sign(0) is 0",
          _fw_sign(0.0) == 1 and _sign(0.0) == 0.0)

    #    7i. The brake-tilt downhill damper (brake_tilt.c:79-90), the reason brake tilt and ATR
    #    are coupled in the firmware at all: brake tilt must NOT lift the nose when gravity is
    #    already doing the slowing. It reads ATR's raw accel_diff, so it only exists once ATR
    #    does — before this port it was hard-pinned at 1 and the steep-downhill disable at :87
    #    could not fire. These two pins are the whole point of that line of the port.
    #    Flat braking (ATR sees no shortfall): the lift is the plain offset/factor, -0.80 deg.
    dt_kw = dict(braketilt_strength=10.0, braketilt_lingering=2.0, atr_amps_decel_ratio=400.0)
    flat = Protections(Params(**dt_kw))
    flat_out = None
    for _ in range(4000):
        flat_out = flat.update(dt, erpm=-5000.0, forward_speed=-3.0, torque_nm=5.0,
                               balance_offset=2.0, braking=True)
    #    Same everything, tight ratio: ATR now reads accel_diff ~ +2.03, damper ~ 2.01 > 2, so
    #    the firmware disables the feature outright ("steep downhills, we don't enable this
    #    feature at all!") and the nose stays level instead of lifting into the turn.
    steep = Protections(Params(**dict(dt_kw, atr_amps_decel_ratio=8.0)))
    steep_out = None
    for _ in range(4000):
        steep_out = steep.update(dt, erpm=-5000.0, forward_speed=-3.0, torque_nm=5.0,
                                 balance_offset=2.0, braking=True)
    check("brake tilt lifts the nose on flat braking (damper 1)",
          abs(flat_out["braketilt_deg"] + 0.8) < 0.03 and abs(steep_out["atr_accel_diff"]) > 1.0,
          "flat %+.3f deg, atr accel_diff %+.2f"
          % (flat_out["braketilt_deg"], steep_out["atr_accel_diff"]))
    check("...and a steep downhill KILLS the lift (brake_tilt.c:87-90, unportable before ATR)",
          abs(steep_out["braketilt_deg"]) < 1e-6 and steep_out["atr_accel_diff"] > 1.0,
          "downhill lift %+.6f deg, accel_diff %+.2f"
          % (steep_out["braketilt_deg"], steep_out["atr_accel_diff"]))

    # --- wheelslip (main.c:483-509) ----------------------------------------------------
    # The input that makes four behaviours change their minds at once, and the one a table
    # cannot show: under a slide none of the modifiers goes to zero and none freezes, all four
    # are shaved by the winddown alpha. Before this port the sim had no wheelslip at all, so
    # four reachable firmware branches (torque_tilt.c:54, atr.c:77, brake_tilt.c:69,
    # turn_tilt.c:93) were dead code and the gate had no way to notice.
    ws_kw = dict(braketilt_strength=10.0, braketilt_lingering=2.0, atr_amps_decel_ratio=40.0,
                 torquetilt_strength=0.5, torquetilt_start_current=15.0,
                 torquetilt_angle_limit=8.0, turntilt_strength=30.0,
                 turntilt_start_erpm=1000.0, turntilt_start_angle=20.0,
                 turntilt_yaw_aggregate=320.0)

    def _slide(p, ticks, duty, e0, accelerate=True):
        """Drive one instance; +100 erpm/tick is 100,000 erpm/s at this dt, ten times the
        enter threshold, with the sign agreeing with erpm (main.c:486)."""
        rep = {}
        for i in range(ticks):
            e = e0 + (100.0 * i if accelerate else 0.0)
            rep = p.update(dt, erpm=e, forward_speed=8.0, torque_nm=45.0, balance_offset=-2.0,
                           braking=True, yaw_rate_dps=55.0, duty=duty)
        return rep

    def _val(p, k):
        return getattr(p, {"torquetilt_deg": "torque_tilt_out", "turntilt_deg": "turn_tilt_out",
                           "braketilt_deg": "brake_tilt_out", "atr_deg": "atr_out"}[k])

    keys = ("torquetilt_deg", "turntilt_deg", "braketilt_deg", "atr_deg")
    # NEXT 365: the remote's lean-in trim. Three things can be wrong here and each has its own line in
    # remote.c to be wrong about — the deadband REMAP (:92-97), the darkride sign flip (:151-153), and
    # "no stick fitted means no trim" (:81-88). The last one is the default state of the whole stage,
    # so it is the check that protects every existing number in this table.
    it_kw = {"inputtilt_remote_type": 1.0, "inputtilt_angle_limit": 10.0,
             "inputtilt_deadband": 0.1, "remote.filter.time_constant": 0.3}
    dt_it = 0.001

    def _drive(p, ticks=4000, **kw):
        for _ in range(ticks):
            out = p.update(dt_it, erpm=6000, forward_speed=6.0, torque_nm=5.0,
                           balance_offset=0.0, **kw)
        return out

    ipn = _drive(Protections(Params(**dict(it_kw, inputtilt_remote_type=0.0))), rider_input=1.0)
    check("no stick fitted (INPUTTILT_NONE) means no trim, however hard the stick is pushed "
          "(remote.c:81-88)", abs(ipn["inputtilt_deg"]) < 1e-12 and ipn["inputtilt_stick"] == 0.0)

    ip = Protections(Params(**it_kw))
    out_b = _drive(ip, rider_input=0.16)
    # 0.16 with a 0.10 deadband is NOT 0.16 and NOT 0: remote.c:96 remaps it to (0.16-0.1)/0.9, and
    # :149 turns that into an angle. A plain threshold would make the trim jump as the stick leaves
    # the deadband, which is the exact jump the remap exists to prevent.
    want = (0.16 - 0.1) / (1.0 - 0.1) * 10.0
    check("the deadband is REMAPPED as (|v|-db)/(1-db), not clipped, and the trim converges on "
          "input x angle_limit (remote.c:92-97, :149)",
          abs(ip.input_tilt.input - (0.16 - 0.1) / 0.9) < 1e-9
          and abs(out_b["inputtilt_deg"] - want) < 1e-6)

    out_d = _drive(Protections(Params(**it_kw)), rider_input=0.16, darkride=True)
    check("darkride flips the trim's sign (remote.c:151-153)",
          out_d["inputtilt_deg"] < 0.0 and abs(out_d["inputtilt_deg"] + want) < 1e-6)

    out_off = _drive(Protections(Params()), rider_input=0.5)
    check("with every inputtilt setting at its shipped default the stage contributes exactly zero, "
          "so every protections number measured before NEXT 365 still means what it said",
          abs(out_off["inputtilt_deg"]) < 1e-12)

    slip_p = _slide(Protections(Params(**ws_kw)), 300, 0.9, 5000.0)
    check("wheelslip detected on a hard acceleration at real duty (main.c:485-489)",
          slip_p["wheelslip"] == 1.0, "wheelslip=%s after 300 ticks at +100000 erpm/s"
          % slip_p["wheelslip"])

    def _pre_and_slip(kw, e0, rate_warm, rate_slip, torque, balance, braking, yaw,
                      warm_ticks=900, slip_ticks=25):
        """Drive one instance with NO duty (so the detector cannot enter), freeze the four
        modifier values, then bring the duty up AND the acceleration in for `slip_ticks`, and
        return before/after pairs. Two rates because the two states need them differently: the
        forward state is already accelerating and only lacks the duty, while brake tilt needs a
        steady erpm for seconds to ramp up at all (brake_tilt.c's ramp is slow — its own pins
        run 4000 ticks to reach -0.80 deg) and gets its acceleration as a kick at the slide."""
        p = Protections(Params(**kw))
        before = {}
        e = e0
        for i in range(warm_ticks + slip_ticks):
            e += rate_warm if i < warm_ticks else rate_slip
            p.update(dt, erpm=e, forward_speed=8.0 if e >= 0 else -8.0, torque_nm=torque,
                     balance_offset=balance, braking=braking, yaw_rate_dps=yaw,
                     duty=None if i < warm_ticks else 0.9)
            if i == warm_ticks - 1:
                before = {k: _val(p, k) for k in keys}
        return {k: (before[k], _val(p, k)) for k in keys}, p

    def _decayed(obs, k):
        b, a = obs[k]
        return abs(b) > 0.2 and 1e-6 < abs(a) < abs(b)

    # +30,000 erpm/s: above the 10,000 enter threshold, but far below the ~77,000 a 45 N*m
    # command should produce, so ATR has a real accel_diff to lean on at the same time the
    # detector is satisfied. (The first version of this pin used +100,000 and ATR sat at zero:
    # accelerating as well as the torque asks is exactly the case ATR does NOT lean on, so it
    # was asserting a winddown on a modifier that was never live.)
    fwd, p_fwd = _pre_and_slip(ws_kw, 5000.0, 30.0, 0.0, 45.0, 0.0, False, 55.0)
    # Brake tilt and ATR want OPPOSITE things from this signal, which took three tries to learn
    # and is worth writing down: brake tilt's target is computed at a *steady* braking erpm and
    # its ramp is slow (its own pin runs 4000 ticks to reach -0.80 deg), while ATR leans on the
    # wheel under-accelerating, which a dead-steady erpm never produces. One state cannot show
    # both, so the winddown is proved with three states and each modifier is checked in the state
    # that actually makes it live — no modifier is quietly skipped because its bar was never met.
    brk, p_brk = _pre_and_slip(ws_kw, -5000.0, 0.0, -300.0, 5.0, 2.0, True, 0.0, warm_ticks=4000)
    atr_s, p_atr = _pre_and_slip(ws_kw, -5000.0, -3.0, -300.0, 5.0, 2.0, True, 0.0,
                                 warm_ticks=4000)

    def _pair(obs, ks):
        return {k: (round(obs[k][0], 2), round(obs[k][1], 2)) for k in ks}

    check("under a slide torque tilt and turn tilt WIND DOWN — not frozen, not zeroed "
          "(torque_tilt.c:54, turn_tilt.c:93)",
          _decayed(fwd, "torquetilt_deg") and _decayed(fwd, "turntilt_deg"),
          "fwd %s" % _pair(fwd, ("torquetilt_deg", "turntilt_deg")))
    check("...and brake tilt WIND DOWNs under a slide (brake_tilt.c:69), target left standing",
          _decayed(brk, "braketilt_deg"), "brake %s" % _pair(brk, ("braketilt_deg",)))
    check("...and so does ATR (atr.c:77) — the one that would otherwise lean INTO the slide",
          _decayed(atr_s, "atr_deg"), "atr %s" % _pair(atr_s, ("atr_deg",)))

    noduty = _slide(Protections(Params(**ws_kw)), 300, None, 5000.0)
    check("...and the detector reads the duty rather than guessing it (main.c:487)",
          noduty["wheelslip"] == 0.0, "duty=None -> wheelslip=%s" % noduty["wheelslip"])
    lazy = _slide(Protections(Params(**ws_kw)), 300, 0.2, 5000.0)
    check("...and a gentle spin-up at 0.2 duty is not a slide (main.c:487 floor is 0.3)",
          lazy["wheelslip"] == 0.0, "duty=0.2 -> wheelslip=%s" % lazy["wheelslip"])
    steady = _slide(Protections(Params(**ws_kw)), 300, 0.9, 25000.0, accelerate=False)
    check("fast and steady is not a slide: the gate is on acceleration, not speed (main.c:485)",
          steady["wheelslip"] == 0.0, "erpm +25000 held -> wheelslip=%s" % steady["wheelslip"])

    held = Protections(Params(**ws_kw))
    _slide(held, 300, 0.9, 5000.0)                      # enter the slide
    for _ in range(int(0.1 / dt)):                      # signals clear, duty drops to 0.5
        held.update(dt, erpm=20000.0, forward_speed=8.0, torque_nm=0.0, balance_offset=0.0,
                    braking=False, duty=0.5)
    early = held.wheelslip
    for _ in range(int(0.25 / dt)):
        held.update(dt, erpm=20000.0, forward_speed=8.0, torque_nm=0.0, balance_offset=0.0,
                    braking=False, duty=0.5)
    check("a slide is HELD 0.2 s after the signals clear, then released (main.c:501-507)",
          early and not held.wheelslip,
          "at 0.1 s slipped=%s, at 0.35 s slipped=%s" % (early, held.wheelslip))

    return fails


def main() -> int:
    print("protection pins (signs, ramps, gating, handtest coupling):")
    fails = _pins()
    if fails:
        print("FAIL %d pin(s): %s" % (len(fails), ", ".join(fails)))
        return 1
    print("protections ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

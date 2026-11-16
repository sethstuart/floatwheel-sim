"""Gymnasium environment around the MuJoCo floatwheel plant.

Usage:
    python -m sim.env                 # smoke test: random actions, prints obs/reward
    python -m tools.random_rollout    # the real rollout harness (N episodes, stats)

Scenes (the rider's three policy targets), all sharing one plant and one observation:
  * balance - hold the board upright through pushes. `act_on="torque"` makes the
    policy the balance controller outright; `act_on="setpoint"` leaves Refloat's PID
    in charge and lets the policy lean the board, which is what a rider does.
  * ride    - track a commanded speed: reward is speed tracking, not just survival.
  * tune    - no policy: run the firmware controller and report the metrics a human
    would judge a tune by (see sim.control._run for the same numbers).

Observation entries are named after Refloat's own telemetry fields
(doc/commands/REALTIME_DATA.md) so a learned policy and a real board read the same
quantities, and sim/tuning can ask "what would VESC Tool have shown?".
"""

from __future__ import annotations

import math
import random

import gymnasium as gym
import mujoco
import numpy as np

from . import control as C
from . import model as M
from . import spec

# Observation layout, named after Refloat's telemetry.
OBS_FIELDS = (
    "balance_pitch_deg",   # what the PID actually steers on
    "pitch_deg",
    "pitch_rate_dps",
    "roll_deg",
    "wheel_speed_rps",
    "erpm_k",
    "speed_mps",
    "torque_Nm",
    "current_A",
    "setpoint_deg",
    "last_action",
)
OBS_SCALE = np.array(
    [10.0, 10.0, 100.0, 10.0, 50.0, 20.0, 5.0, spec.g("TORQUE_MAX"), 40.0, 6.0, 1.0]
)

# NEXT 366: the two extra channels a policy sees ONLY when it is given the rider's lean-in stick
# (`FloatwheelEnv(lean_action=True)`). They live here rather than in OBS_FIELDS on purpose: every
# checkpoint, every reward curve and every number in docs/TUNING.md was measured on the 11-field
# observation, and silently widening it would have quietly invalidated all of them while leaving the
# files looking the same. Ask for the stick, get the stick; ask for nothing, get exactly what the
# lane has always had.
LEAN_FIELDS = (
    "rider_stick",        # the action itself, fed back — what the rider asked for
    "inputtilt_deg",      # what the remote's filter actually delivered (remote.c:156)
)
LEAN_SCALE = np.array([1.0, 6.0])   # stick is already +-1; trim is a lean angle, like setpoint_deg


class FloatwheelEnv(gym.Env):
    """One board, one rider, one floor. See the module docstring for the scenes."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        scene: str = "balance",
        act_on: str = "torque",
        param_overrides: dict | None = None,
        control_hz: float | None = None,
        max_seconds: float = 10.0,
        push_mean_s: float | None = 4.0,
        # Calibrated on the balance test: 250 N for 0.1 s tips the board only ~0.4 deg
        # under the PID, which is a nudge, not a shove, and would make every policy
        # look steady. These values tip it a few degrees, so surviving says something.
        push_N: tuple[float, float] = (300.0, 700.0),
        noise_deg: float = 0.0,
        foot_adc: float = 3.0,
        # What action 1.0 commands, as a fraction of TORQUE_MAX. Keep this at 1.0:
        # torque_scale=0.3 was an attempt to give the policy finer resolution and it
        # instead capped its authority at 8.4 N*m, which physically cannot hold more
        # than 0.6 deg of lean - recovery from a shove needs near 27 N*m - so the policy
        # fell in 100% of episodes no matter how long it trained. Resolution near zero
        # is what `torque_map` is for; authority stays full.
        torque_scale: float = 1.0,
        # "quad": tau = sign(a) * scale * TORQUE_MAX * a^2. Fine control in the
        # +-1.5 N*m band where balancing actually happens, full authority at the ends,
        # and the policy's own gradient stays small near 0 where it needs precision.
        torque_map: str = "quad",
        # NEXT 366: expose the rider's lean-in stick (remote.c's inputtilt) as a second action and
        # two extra observations. Off by default: the action space stays 1-D and the observation
        # stays the 11 fields every checkpoint and tuning number in this lane was measured on.
        # NEXT 367: whether the protection fleet runs at all. It has been hard-OFF in this env since
        # the day it was written — reset() built its controller with BalanceController's default
        # `protections=False` — so every policy this lane has ever trained ran on a board with the
        # tiltbacks, ATR, brake tilt and wheelslip detector compiled out, and was then scored against
        # hand-PD bars measured on a board that has them. Defaulting to False keeps that behaviour and
        # every number already in docs/TUNING.md reproducible; the row is about re-measuring it, not
        # about silently changing it.
        protections: bool = False,
        lean_action: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        assert act_on in ("torque", "setpoint"), act_on
        assert scene in ("balance", "ride"), scene
        self.scene = scene
        self.act_on = act_on
        self.overrides = dict(param_overrides or {})
        self.control_hz = control_hz or spec.g("CONTROL_HZ")
        self.max_seconds = max_seconds
        self.push_mean_s = push_mean_s
        self.push_N = push_N
        self.noise_deg = noise_deg
        self.foot_adc = foot_adc
        self.torque_scale = torque_scale
        self.torque_map = torque_map
        self.rng = random.Random(seed)
        # NEXT 366: what the stick means to this env, and what it means to the board.
        self.lean_action = bool(lean_action)
        if self.lean_action:
            # In `torque` mode the policy IS the balance controller and `controller.update()` never
            # runs, so nothing in sim/protections.py is stepped — and the trim lives there, entering
            # at main.c:922-923. A stick handed to that mode would be a channel that does nothing and
            # reports nothing, which is the NEXT 365 control-table failure all over again. Refuse it at
            # construction rather than letting a training run discover it an hour in.
            assert act_on == "setpoint", (
                "lean_action=True needs act_on='setpoint': in torque mode the protections are never "
                "stepped, so the rider's trim could not reach the board"
            )
            # A stick the firmware would read as INPUTTILT_NONE (remote.c:81-88) reads input 0 forever
            # — a policy trained on nothing. Default it to the UART stick's js_y channel (:75-79) so
            # the stage the policy trains against is the stage the board would actually run.
            self.overrides.setdefault("inputtilt_remote_type", 1.0)
        # The stick needs the protections (its stage lives inside them), so asking for one implies the
        # other. Anything wider is the rider's call per run, not a default this file gets to make.
        self.protections = bool(protections) or self.lean_action

        self.model = M.load()
        self.data = mujoco.MjData(self.model)
        self.rd = M._root_dof(self.model)
        self.substeps = max(1, int(round(1.0 / self.control_hz / self.model.opt.timestep)))

        # Widths follow the two switches above: no stick asked for, no stick in the action and no
        # stick fields in the observation — the 1-D / 11-field shapes the whole lane is measured on.
        n_act = 2 if self.lean_action else 1
        n_obs = len(OBS_FIELDS) + (len(LEAN_FIELDS) if self.lean_action else 0)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n_act,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (n_obs,), dtype=np.float64)
        self.controller: C.BalanceController | None = None
        self._last_action = 0.0
        self._prev_action = 0.0
        self._target_speed = 0.0
        self._next_push = math.inf
        self._episode_steps = 0
        self._last_info: dict = {}

    # --- gym API ---------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = M.upright_qpos(self.model)

        # Never start in perfect equilibrium: an exact inverted pendulum at rest stands
        # up forever with no controller at all, which turns every test vacuous.
        tip = math.radians(self.rng.uniform(0.1, 0.4))
        rate = math.radians(self.rng.uniform(-3.0, 3.0))
        jadr = self.model.jnt_qposadr[0]
        self.data.qpos[jadr + 3 : jadr + 7] = [math.cos(tip / 2), 0.0, math.sin(tip / 2), 0.0]
        self.data.qvel[self.rd + 4] = rate

        params = C.Params(**self.overrides)
        # NEXT 366: the random-policy line in main() was not reproducible — three runs of the same
        # seed gave reward 26.0/697, 90.6/1138 and 43.2/754, because this env's custom reset() never
        # seeds the action space, so `action_space.sample()` was pulling from gym's unseeded global
        # stream. A line that looks like a measurement and cannot be reproduced is worse than no line,
        # because it gets quoted. The stick now derives from `self.rng`, which IS seeded, so a given
        # seed gives a given episode and the self-check can be trusted as a gate.
        self.action_space.seed(self.rng.randrange(1 << 31))
        # NEXT 366: the rider's trim lives INSIDE the protections (main.c:922-923), so an env that hands
        # out a stick has to actually step them. NEXT 367 widened that to any caller: the controller used
        # to be built with BalanceController's default protections=False, so nothing in sim/protections.py
        # was ever stepped during a training run — not a tiltback, not ATR, not brake tilt, not the
        # wheelslip detector — and the policies were then scored against hand-PD bars measured on a board
        # that does have them. The flag is the caller's now; the default is still False, so every number
        # already in docs/TUNING.md stays reproducible and says what it was measured on.
        self.controller = C.BalanceController(
            params, self.control_hz, protections=self.protections
        )
        self._last_action = 0.0
        self._last_stick = 0.0
        self._prev_action = 0.0
        self._target_speed = self.rng.uniform(1.5, 4.0) if self.scene == "ride" else 0.0
        self._next_push = (
            self.rng.expovariate(1.0 / self.push_mean_s) if self.push_mean_s else math.inf
        )
        self._episode_steps = 0
        return self._obs(), self._info()

    def set_push_mean(self, seconds):
        """Disturbance schedule, for training: hold it upright first, then start
        shoving. VecEnv reaches this through env_method between training phases."""
        self.push_mean_s = seconds
        self._next_push = (
            math.inf if not seconds else self.data.time + self.rng.expovariate(1.0 / seconds)
        )

    def step(self, action):
        _act = np.asarray(action, dtype=float).reshape(-1)
        a = float(np.clip(_act[0], -1.0, 1.0))
        self._last_action = a
        # NEXT 366: the stick is the policy's second action, in the firmware's own range. remote.c's
        # `value` is a normalized stick in [-1, 1] (UART js_y at :77), and the deadband remap that runs
        # downstream — (|v|-db)/(1-db), :92-97 — is nonsense beyond 1, so it is clipped here rather
        # than trusted. A missing second channel is a caller bug, not a reason to silently ride at 0.
        if self.lean_action:
            if _act.size < 2:
                raise ValueError(
                    "this env was built with lean_action=True, so its action is [torque, stick] "
                    "and a 1-element action is a caller bug, not a stick at rest"
                )
            self._last_stick = float(np.clip(_act[1], -1.0, 1.0))
        else:
            self._last_stick = 0.0
        terminated = False
        truncated = False
        reward = 0.0

        for _ in range(self.substeps):
            pitch_deg = math.degrees(M.pitch_of(self.model, self.data))
            roll_deg = math.degrees(M.roll_of(self.model, self.data))
            rate_dps = math.degrees(M.pitch_rate(self.model, self.data))
            if not self.controller.engaged:
                self.controller.engage(pitch_deg, roll_deg, self.foot_adc)
            if self.data.time >= self._next_push:
                self._shove()
                self._next_push = self.data.time + self.rng.expovariate(1.0 / self.push_mean_s)

            if self.act_on == "torque":
                # The IMU filter runs on the board no matter who is commanding torque.
                # This call was missing, so in torque mode `balance_pitch` - observation
                # field 0, the number any balance controller leans on - sat at exactly
                # 0.000 for the whole episode: the hand controller and every PPO policy
                # were steering blind. The setpoint branch gets it inside update().
                self.controller.update_balance_pitch(
                    self.model.opt.timestep, pitch_deg, rate_dps
                )
                # Gate the way the board does: no feet on the pads, no torque. Without
                # this a policy can learn to drive a board nobody is standing on.
                if self.controller.engaged:
                    gain = self.torque_scale * spec.g("TORQUE_MAX")
                    tau = (
                        math.copysign(gain * a * a, a)
                        if self.torque_map == "quad"
                        else a * gain
                    )
                else:
                    tau = 0.0
                self.controller.torque = tau
                self.controller.current = tau / self.controller.p["torque_per_current"]
            else:
                tau = self.controller.update(
                    self.model.opt.timestep * self.substeps,
                    pitch_deg,
                    rate_dps,
                    roll_deg,
                    C.erpm_of(M.wheel_speed(self.model, self.data), self.controller.p["motor_poles"]),
                    setpoint_deg=a * 6.0,
                    # NEXT 366: the rider's second action, handed straight to the remote's stage.
                    # Nothing here pre-processes it — the deadband, the invert and the angle limit
                    # belong to remote.c (:92-97, :114, :149), and double-handling it is how a sim
                    # quietly drifts away from the board it is supposed to predict.
                    rider_input=self._last_stick,
                )
            M.set_torque(self.model, self.data, tau)
            mujoco.mj_step(self.model, self.data)
            self._episode_steps += 1

            pitch_abs = abs(math.degrees(M.pitch_of(self.model, self.data)))
            roll_abs = abs(math.degrees(M.roll_of(self.model, self.data)))
            if pitch_abs > spec.g("STOP_PITCH_DEG") or roll_abs > self.controller.p.get(
                "fault_roll", 60.0
            ):
                terminated = True
                break
            if self.data.time >= self.max_seconds:
                truncated = True
                break

        reward = self._reward(terminated)
        return self._obs(), reward, terminated, truncated, self._info()

    # --- internals -------------------------------------------------------------
    def _shove(self):
        m_total = spec.g("DECK_MASS") + spec.g("RIDER_MASS")
        h = spec.g("COM_HEIGHT")
        force = self.rng.uniform(*self.push_N)
        sign = self.rng.choice([-1.0, 1.0])
        # angular impulse about the axle at torso height, I = m h^2 (a shove on the
        # *base* is a velocity boost and does nothing - see sim/control.py)
        self.data.qvel[self.rd + 4] += sign * force * 0.1 * M.RIDER_H / (m_total * h * h)

    def _reward(self, terminated: bool) -> float:
        pitch = math.degrees(M.pitch_of(self.model, self.data))
        rate = math.degrees(M.pitch_rate(self.model, self.data))
        u = self._last_action
        upright = math.exp(-(pitch / 4.0) ** 2)
        calm = math.exp(-(rate / 60.0) ** 2)
        # Rate penalty, not just magnitude: without it a random-exploration policy
        # learns to thrash the hub between +/- full torque, which no rider would accept
        # and which made the first PPO run fall in 100% of episodes at std 1.0.
        effort = 0.02 * u * u + 0.08 * (u - self._prev_action) ** 2
        r = upright + 0.3 * calm + 0.05 - effort
        self._prev_action = u
        if self.scene == "ride":
            err = (M.forward_speed(self.model, self.data) - self._target_speed) / 3.0
            r += 0.5 * math.exp(-err * err)
        # -2, not -5: with gamma this high (0.998 at 100 Hz, a ~5 s horizon) a huge fall
        # penalty dominates every early episode and the policy learns to freeze instead
        # of to recover. The alive bonus above is what pays for staying up.
        if terminated:
            r -= 2.0
        return float(r)

    def _obs(self) -> np.ndarray:
        c = self.controller
        pitch_deg = math.degrees(M.pitch_of(self.model, self.data))
        obs = np.array(
            [
                c.balance_pitch if c else 0.0,
                pitch_deg,
                math.degrees(M.pitch_rate(self.model, self.data)),
                math.degrees(M.roll_of(self.model, self.data)),
                M.wheel_speed(self.model, self.data),
                C.erpm_of(M.wheel_speed(self.model, self.data), c.p["motor_poles"]) / 1000.0
                if c
                else 0.0,
                float(M.forward_speed(self.model, self.data)),
                c.torque if c else 0.0,
                c.current if c else 0.0,
                c.setpoint if c else 0.0,
                self._last_action,
            ]
            + (
                # NEXT 366: what the rider asked for, and what the remote's filter actually delivered
                # (remote.c:156). Both, not one of them: a policy learning that the stick leans the
                # board has to see the lag between the command and the realised trim, and a probe has
                # to be able to tell "no trim asked for" from "trim asked for and the stage did
                # nothing" — the same reason the wheelslip row exists in the plant gate.
                [self._last_stick, (c.prot_report or {}).get("inputtilt_deg", 0.0)]
                if self.lean_action
                else []
            )
        )
        if self.noise_deg:
            obs[0] += self.rng.gauss(0.0, self.noise_deg)
            obs[1] += self.rng.gauss(0.0, self.noise_deg)
            obs[2] += self.rng.gauss(0.0, self.noise_deg * 20.0)
        # The base 11 scales, exactly as they have always been, plus the stick's two when they were
        # asked for — concatenated at the end so no existing field changes meaning.
        return obs / (np.concatenate([OBS_SCALE, LEAN_SCALE]) if self.lean_action else OBS_SCALE)

    def _info(self) -> dict:
        c = self.controller
        info = {
            # Refloat telemetry names, so a tuning report reads like VESC Tool's
            "pitch": math.degrees(M.pitch_of(self.model, self.data)),
            "balance_pitch": c.balance_pitch if c else 0.0,
            "pitch_rate": math.degrees(M.pitch_rate(self.model, self.data)),
            "roll": math.degrees(M.roll_of(self.model, self.data)),
            "speed": float(M.forward_speed(self.model, self.data)),
            "current": c.current if c else 0.0,
            "setpoint": c.setpoint if c else 0.0,
            "engaged": bool(c and c.engaged),
            "target_speed": self._target_speed,
            "time": float(self.data.time),
        }
        self._last_info = info
        return info


def main() -> int:
    env = FloatwheelEnv(scene="balance", act_on="torque", seed=0)
    obs, info = env.reset(seed=0)
    print("obs fields:", OBS_FIELDS)
    print("obs:", np.round(obs, 3))
    total = 0.0
    for _ in range(400):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    print(
        "random policy: reward %.1f after %d steps, pitch %.1f deg, engaged %s, fell %s"
        % (total, env._episode_steps, info["pitch"], info["engaged"], term)
    )
    # NEXT 366: three things that would make this channel a lie, each proved rather than asserted.
    plain = FloatwheelEnv(scene="balance", act_on="torque", seed=1)
    print(
        "no stick asked for: action %s, obs %d fields (must stay (1,) / %d — every checkpoint, "
        "policy and TUNING.md number in the lane was measured on exactly these)"
        % (plain.action_space.shape, len(OBS_FIELDS), len(OBS_FIELDS))
    )
    lean = FloatwheelEnv(scene="balance", act_on="setpoint", lean_action=True, seed=1)
    obs0, _ = lean.reset(seed=1)
    print("stick asked for: action %s, obs %d fields (adds %s)"
          % (lean.action_space.shape, obs0.size, LEAN_FIELDS))

    def _roll(stick):
        e = FloatwheelEnv(scene="balance", act_on="setpoint", lean_action=True, seed=7)
        e.reset(seed=7)
        info = {}
        for _ in range(300):
            *_, term, trunc, info = e.step([0.0, stick])
            if term or trunc:
                break
        return info["pitch"], (e.controller.prot_report or {}).get("inputtilt_deg", 0.0)

    # Same seed, same torque action, stick held hard one way then hard the other. If those two
    # trajectories agree, the channel is compiled out and this whole row is decoration — the NEXT 362
    # and NEXT 365 failure, third time, so it is now checked here instead of trusted.
    fwd, back = _roll(1.0), _roll(-1.0)
    soft_f, soft_b = _roll(0.3), _roll(-0.3)
    print("stick +-0.3 (a lean-in lean): realised trim %+.2f / %+.2f deg, pitch %+.2f / %+.2f deg"
          % (soft_f[1], soft_b[1], soft_f[0], soft_b[0]))
    print("stick +-1.0 (the stick pinned): realised trim %+.2f / %+.2f deg, pitch %+.2f / %+.2f deg"
          % (fwd[1], back[1], fwd[0], back[0]))
    assert fwd[1] > 1.0 and back[1] < -1.0, "the stick never reached the remote's stage"
    assert abs(fwd[0] - back[0]) > 1.0, "the trim moved and the plant ignored it"
    assert abs(soft_f[1] + soft_b[1]) < 0.5, "the trim is not symmetric about the stick's sign"
    assert abs(soft_f[0]) < 45.0 and abs(soft_b[0]) < 45.0, "a gentle lean should not end on the deck"
    # Pinned backward at 10 deg the board goes over, and that IS the result rather than a broken check:
    # demanding a 10 deg lean one way while riding tips you off the back of it. That is the whole reason
    # firmware has inputtilt_angle_limit and inputtilt_deadband between the stick and the setpoint, and
    # it is why the angle limit is a safety parameter and not a comfort one.
    try:
        FloatwheelEnv(scene="balance", act_on="torque", lean_action=True, seed=0)
    except AssertionError as exc:
        print("torque mode refuses the stick, which is the point: %s" % exc)
    else:
        raise AssertionError("torque mode accepted a stick it can never deliver")
    # NEXT 367: the flag has to be a switch, not a keyword the controller ignores. Same seed, same
    # full-forward demand, same over-speed tiltback armed in both — the only difference is whether the
    # fleet is stepped at all, which is exactly the comparison every training run in this lane has been
    # missing. Reported as "nonzero offsets seen", keyed on nothing, so it cannot pass by asserting on a
    # report field that a refactor renames.
    def _armed(protections):
        e = FloatwheelEnv(scene="ride", act_on="setpoint", protections=protections, seed=3,
                         param_overrides={"tiltback_speed": 2.0, "tiltback_speed_angle": 15.0})
        e.reset(seed=3)
        info = {}
        for _ in range(500):
            *_, term, trunc, info = e.step([1.0])
            if term or trunc:
                break
        rep = e.controller.prot_report or {}
        seen = sorted(k for k, v in rep.items()
                      if isinstance(v, (int, float)) and abs(float(v)) > 1.0)
        return seen, info

    off_seen, off_info = _armed(False)
    on_seen, on_info = _armed(True)
    print("fleet OFF (how this lane has always trained): offsets over 1 deg = %s | "
          "pitch %.2f deg" % (off_seen or "none", off_info["pitch"]))
    print("fleet ON, over-speed tiltback armed at 2.0 m/s, 15 deg: offsets over 1 deg = %s | "
          "pitch %.2f deg" % (on_seen or "none", on_info["pitch"]))
    assert not off_seen, "protections=False is still stepping something: the flag is not a switch"
    assert on_seen, (
        "protections=True armed an over-speed tiltback and nothing moved — either the flag is inert or "
        "the demand never reached tiltback_speed, and either way the comparison is not being measured"
    )
    print("env protections flag switches the fleet, which is the point")
    print("env lean-in channel ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

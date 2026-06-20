from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
from mujoco import MjModel


@dataclass
class Config:
    velocity_kick: tuple[float, float] = (5.0, 20.0)  # m/s
    kick_duration: tuple[float, float] = (0.05, 0.2)  # seconds
    kick_interval: tuple[float, float] = (1.0, 3.0)   # seconds
    torso_body_id: int = 1


class PerturbationWrapper:
    def __init__(self, env: Any, mj_model: MjModel, config: Config, ctrl_dt: float):
        self._env = env
        self._config = config
        self._ctrl_dt = ctrl_dt
        self._torso_mass = float(mj_model.body_mass[config.torso_body_id])
        self._nbody = mj_model.nbody

    @property
    def action_size(self) -> int:
        return self._env.action_size

    @property
    def mjx_model(self):
        return self._env.mjx_model

    @property
    def unwrapped(self):
        return self

    @property
    def _mjx_model(self):
        return self._env.unwrapped._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self._env.unwrapped._mjx_model = value

    def reset(self, rng):
        state = self._env.reset(rng)
        rng = state.info["rng"]
        cfg = self._config
        dt = self._ctrl_dt

        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        angle = jax.random.uniform(k1, minval=0.0, maxval=jp.pi * 2)

        state.info["rng"] = rng
        state.info["pert_active"] = jp.bool_(False)
        state.info["pert_step"] = jp.int32(0)
        state.info["pert_duration"] = jp.int32(
            jp.round(jax.random.uniform(k2, minval=cfg.kick_duration[0], maxval=cfg.kick_duration[1]) / dt)
        )
        state.info["pert_duration_seconds"] = jax.random.uniform(
            k2, minval=cfg.kick_duration[0], maxval=cfg.kick_duration[1]
        )
        state.info["wait_step"] = jp.int32(0)
        state.info["pert_steps_until_next"] = jp.int32(
            jp.round(jax.random.uniform(k3, minval=cfg.kick_interval[0], maxval=cfg.kick_interval[1]) / dt)
        )
        state.info["pert_dir"] = jp.array([jp.cos(angle), jp.sin(angle), 0.0])
        state.info["pert_mag"] = jax.random.uniform(
            k4, minval=cfg.velocity_kick[0], maxval=cfg.velocity_kick[1]
        )
        state.info["pert_triggered"] = jp.bool_(False)

        return state

    def step(self, state, action):
        cfg = self._config
        dt = self._ctrl_dt
        rng = state.info["rng"]
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)

        # Sample candidates for a potential new kick
        angle = jax.random.uniform(k1, minval=0.0, maxval=jp.pi * 2)
        new_dir = jp.array([jp.cos(angle), jp.sin(angle), 0.0])
        new_mag = jax.random.uniform(k2, minval=cfg.velocity_kick[0], maxval=cfg.velocity_kick[1])
        new_dur_s = jax.random.uniform(k3, minval=cfg.kick_duration[0], maxval=cfg.kick_duration[1])
        new_dur_steps = jp.int32(jp.round(new_dur_s / dt))
        new_wait_steps = jp.int32(jp.round(
            jax.random.uniform(k4, minval=cfg.kick_interval[0], maxval=cfg.kick_interval[1]) / dt
        ))

        pert_active = state.info["pert_active"]
        pert_step = state.info["pert_step"]
        pert_duration = state.info["pert_duration"]
        pert_dur_s = state.info["pert_duration_seconds"]
        wait_step = state.info["wait_step"]
        steps_until_next = state.info["pert_steps_until_next"]
        triggered = state.info["pert_triggered"]

        # Transitions
        kick_ended = pert_active & (pert_step >= pert_duration)
        timer_fired = ~pert_active & (wait_step >= steps_until_next)
        start_kick = (timer_fired | triggered) & ~pert_active

        # Update state
        pert_active = jp.where(start_kick, jp.bool_(True), jp.where(kick_ended, jp.bool_(False), pert_active))
        pert_step = jp.where(start_kick, jp.int32(0), jp.where(pert_active, pert_step + 1, pert_step))
        pert_duration = jp.where(start_kick, new_dur_steps, pert_duration)
        pert_dur_s = jp.where(start_kick, new_dur_s, pert_dur_s)
        pert_dir = jp.where(start_kick, new_dir, state.info["pert_dir"])
        pert_mag = jp.where(start_kick, new_mag, state.info["pert_mag"])
        wait_step = jp.where(kick_ended, jp.int32(0), jp.where(pert_active, wait_step, wait_step + 1))
        steps_until_next = jp.where(start_kick, new_wait_steps, steps_until_next)

        # Compute force
        t = pert_step.astype(jp.float32) * dt
        u_t = 0.5 * jp.sin(jp.pi * t / (pert_dur_s + 1e-6))
        force_mag = u_t * self._torso_mass * pert_mag / (pert_dur_s + 1e-6)
        force = force_mag * pert_dir * pert_active.astype(jp.float32)

        xfrc = jp.zeros((self._nbody, 6))
        xfrc = xfrc.at[cfg.torso_body_id, :3].set(force)

        state.info["rng"] = rng
        state.info["pert_active"] = pert_active
        state.info["pert_step"] = pert_step
        state.info["pert_duration"] = pert_duration
        state.info["pert_duration_seconds"] = pert_dur_s
        state.info["pert_dir"] = pert_dir
        state.info["pert_mag"] = pert_mag
        state.info["wait_step"] = wait_step
        state.info["pert_steps_until_next"] = steps_until_next
        state.info["pert_triggered"] = jp.bool_(False)

        state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
        state = self._env.step(state, action)
        return state

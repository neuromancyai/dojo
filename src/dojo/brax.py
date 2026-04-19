from collections.abc import Callable
from typing import Any, Optional

import flax.struct
import jax
import jax.numpy as jp

from brax.envs.wrappers import training as brax_training
from jax import Array
from mujoco import MjModel, mjx

from .environment import FeatureExtractor as AbstractFeatureExtract, Observe, Reward
from .utility.mujoco import make_data, step


@flax.struct.dataclass
class State:
    data: mjx.Data
    obs: dict[str, Array]
    reward: dict[str, Array]
    done: Array
    metrics: dict[str, Array]
    info: dict[str, Any]


type FeatureExtractor[F] = AbstractFeatureExtract[mjx.Data, F]
type FeatureExtractorFactory[F] = \
    Callable[[MjModel, mjx.Model], FeatureExtractor[F]]


class Environment[F]:
    def __init__(
        self,
        mj_model: MjModel,
        feature_extractor_factory: FeatureExtractorFactory[F],
        observe: Observe[F],
        reward: Reward[F],
        control_dt: float,
        substeps: int,
        nconmax: int,
        njmax: int
    ) -> None:
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl="warp")
        self._feature_extractor_factory = feature_extractor_factory
        self._feature_extractor = None
        self._observe = observe
        self._reward = reward
        self._control_dt = control_dt
        self._substeps = substeps
        self._nconmax = nconmax
        self._njmax = njmax

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    def reset(self, rng: Array) -> State:
        self._feature_extractor = self._feature_extractor_factory(
            self._mj_model,
            self._mjx_model
        )

        data = make_data(
            self._mj_model,
            qpos=jp.array(self._mj_model.keyframe("home").qpos),
            qvel=jp.zeros(self._mjx_model.nv),
            nconmax=self._nconmax,
            njmax=self._njmax,
            impl="warp",
            device=None
        )

        data = mjx.forward(self._mjx_model, data)
        features, done, rng = self._feature_extractor.init(data, rng)
        obs = self._observe(features, done)
        reward = jp.zeros(())
        metrics = {
            f"reward/{k}": jp.zeros(())
            for k in self._reward(features, done)
        }
        metrics["reward"] = jp.zeros(())

        return State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info={
                "rng": rng,
                "features": features
            }
        )

    def step(self, state: State, action: Array) -> State:
        default_pose = self._mj_model.keyframe("home").qpos[7:]
        lower_control_limits = self._mj_model.actuator_ctrlrange[:, 0]
        upper_control_limits = self._mj_model.actuator_ctrlrange[:, 1]

        targets = default_pose + 0.3 * action
        targets = jp.clip(targets, lower_control_limits, upper_control_limits)

        data = step(
            self._mjx_model,
            state.data,
            targets,
            self._substeps
        )

        rng = state.info["rng"]
        previous_features = state.info["features"]
        current_features, done, rng = self._feature_extractor.step(
            previous_features,
            data,
            action,
            rng
        )

        obs = self._observe(current_features, done)

        reward_terms = self._reward(current_features, done)
        reward = sum(reward_terms.values()) * self._control_dt
        metrics = {f"reward/{k}": v for k, v in reward_terms.items()}
        metrics["reward"] = reward

        state.info["rng"] = rng
        state.info["features"] = current_features
        state = state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics)

        return state


class AutoResetWrapper:
  def __init__(self, env: Any, full_reset: bool):
    self._env = env
    self._full_reset = full_reset
    self._info_key = "AutoResetWrapper"

  @property
  def action_size(self) -> int:
    return self._env.action_size

  def reset(self, rng: Array) -> State:
    rng_key = jax.vmap(jax.random.split)(rng)
    rng, key = rng_key[..., 0], rng_key[..., 1]
    state = self._env.reset(key)
    state.info[f'{self._info_key}_first_data'] = state.data
    state.info[f'{self._info_key}_first_obs'] = state.obs
    state.info[f'{self._info_key}_rng'] = rng
    state.info[f'{self._info_key}_done_count'] = jp.zeros(
        key.shape[:-1],
        dtype=int
    )

    return state

  def step(self, state: State, action: Array) -> State:
    reset_state = None
    rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
    reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]

    if self._full_reset:
      reset_state = self.reset(reset_key)
      reset_data = reset_state.data
      reset_obs = reset_state.obs
    else:
      reset_data = state.info[f'{self._info_key}_first_data']
      reset_obs = state.info[f'{self._info_key}_first_obs']

    state = state.replace(done=jp.zeros_like(state.done))
    state = self._env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, reset_data, state.data)
    obs = jax.tree.map(where_done, reset_obs, state.obs)

    next_info = state.info
    done_count_key = f'{self._info_key}_done_count'
    if self._full_reset and reset_state:
      next_info = jax.tree.map(where_done, reset_state.info, state.info)
      next_info[done_count_key] = state.info[done_count_key]

      preserve_info_key = f'{self._info_key}_preserve_info'
      if preserve_info_key in next_info:
        next_info[preserve_info_key] = state.info[preserve_info_key]

      for key in ('episode_metrics', 'steps'):
        if key in state.info:
          next_info[key] = state.info[key]

    next_info[done_count_key] += state.done.astype(int)
    next_info[f'{self._info_key}_rng'] = reset_rng

    return state.replace(data=data, obs=obs, info=next_info)


def wrap(
    env: Any,
    episode_length: int,
    action_repeat: int,
    full_reset: bool,
    **_
) -> Any:
    env = brax_training.VmapWrapper(env)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env, full_reset)

    return env

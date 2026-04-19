
import functools
import time

from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco import MjModel

from .brax import Environment, wrap
from .quadruped.joystick.brax import (
    Config as EnvironmentConfig,
    feature_extractor,
    observe,
    reward
)
from .training import Config as TrainingConfig


COMMAND = jp.array([1.0, 0.0, 0.0])  # vx, vy, yaw


def main():
    mj_model_path = Path("./scene.xml")
    mj_model = MjModel.from_xml_path(str(mj_model_path))
    mj_data = mujoco.MjData(mj_model)

    environment_config = EnvironmentConfig()
    environment = Environment(
        mj_model,
        functools.partial(feature_extractor, environment_config),
        observe,
        reward(environment_config.reward),
        control_dt=environment_config.ctrl_dt,
        substeps=int(environment_config.ctrl_dt / environment_config.sim_dt),
        nconmax=environment_config.nconmax,
        njmax=environment_config.njmax
    )

    training_config = TrainingConfig()
    network_factory_config = training_config.network_factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **asdict(network_factory_config)
    )

    checkpoints_dir = Path("./checkpoints").resolve()
    checkpoint_path = str(max(checkpoints_dir.iterdir(), key=lambda p: p.name))
    print(f"Loading checkpoint: {checkpoint_path}")

    make_inference_fn, params, _ = ppo.train(
        environment=environment,
        num_timesteps=0,
        num_evals=0,
        episode_length=training_config.episode_length,
        network_factory=network_factory,
        normalize_observations=training_config.normalize_observations,
        seed=1,
        restore_checkpoint_path=checkpoint_path,
        wrap_env_fn=functools.partial(wrap, full_reset=training_config.full_reset),
    )

    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    step_fn = jax.jit(environment.step)

    @jax.jit
    def inject_command(state, command):
        features = state.info["features"].replace(
            current_command=command,
            previous_command=command
        )
        new_obs = observe(features, state.done)
        state.info["features"] = features
        return state.replace(obs=new_obs)

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    state = environment.reset(reset_key)
    state = inject_command(state, COMMAND)

    print("Warming up JIT...")
    rng, warmup_key = jax.random.split(rng)
    action, _ = inference_fn(state.obs, warmup_key)
    state = step_fn(state, action)
    state = inject_command(state, COMMAND)
    jax.block_until_ready(state)
    print("Ready.")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            rng, step_key = jax.random.split(rng)
            action, _ = inference_fn(state.obs, step_key)
            state = step_fn(state, action)
            state = inject_command(state, COMMAND)

            if state.done:
                rng, reset_key = jax.random.split(rng)
                state = environment.reset(reset_key)
                state = inject_command(state, COMMAND)

            mj_data.qpos[:] = state.data.qpos
            mj_data.qvel[:] = state.data.qvel
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(environment_config.ctrl_dt)


if __name__ == "__main__":
    main()

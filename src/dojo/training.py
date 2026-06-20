
import argparse
import functools
import os

from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import jax
import matplotlib.pyplot as plt

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco import MjModel, mjx

from .brax import Environment, wrap
from .perturbation import Config as PerturbationConfig, PerturbationWrapper
from .utility.dataclasses import default_field


os.environ["MUJOCO_GL"] = "egl"


@dataclass
class Config:

    @dataclass
    class NetworkFactory:
        policy_hidden_layer_sizes: list[int] = default_field([
            128,
            128,
            128,
            128
        ])

        value_hidden_layer_sizes: list[int] = default_field([
            256,
            256,
            256,
            256,
            256
        ])

        policy_obs_key: str = "policy"
        value_obs_key: str = "value"

    num_timesteps: int = 500_000_000
    num_evals: int = 50
    reward_scaling: float = 1.0
    episode_length: int = 1000
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-2
    num_envs: int = 8192
    batch_size: int = 256
    max_grad_norm: float = 1.0
    num_resets_per_eval: int = 10
    num_eval_envs: int = 128
    full_reset: bool = True
    network_factory: NetworkFactory = default_field(NetworkFactory())


TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        rng, key = jax.random.split(rng)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        rng, key = jax.random.split(rng)
        kd = model.dof_damping[6:] * jax.random.uniform(
            key, (model.nu,), minval=0.8, maxval=1.2
        )
        dof_damping = model.dof_damping.at[6:].set(kd)

        rng, key = jax.random.split(rng)
        torso_mass = model.body_mass[TORSO_BODY_ID] * jax.random.uniform(
            key, minval=0.8, maxval=1.2
        )
        body_mass = model.body_mass.at[TORSO_BODY_ID].set(torso_mass)

        rng, key = jax.random.split(rng)
        floor_sliding = model.geom_friction[0, 0] * jax.random.uniform(
            key, minval=0.8, maxval=1.2
        )
        geom_friction = model.geom_friction.at[0, 0].set(floor_sliding)

        return actuator_gainprm, actuator_biasprm, dof_damping, body_mass, geom_friction

    actuator_gainprm, actuator_biasprm, dof_damping, body_mass, geom_friction = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "dof_damping": 0,
        "body_mass": 0,
        "geom_friction": 0,
    })

    model = model.tree_replace({
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
        "dof_damping": dof_damping,
        "body_mass": body_mass,
        "geom_friction": geom_friction,
    })

    return model, in_axes

_CURRICULA = ("stability", "sit", "hind_stand", "joystick")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("curriculum", choices=_CURRICULA)
    parser.add_argument("checkpoint", nargs="?", default=None)
    args = parser.parse_args()

    if args.curriculum == "stability":
        from .quadruped.stability import (
            Config as EnvironmentConfig,
            feature_extractor, observe, reward
        )
    elif args.curriculum == "hind_stand":
        from .quadruped.hind_stand import (
            Config as EnvironmentConfig,
            feature_extractor, observe, reward
        )
    elif args.curriculum == "joystick":
        from .quadruped.joystick_prm import (
            Config as EnvironmentConfig,
            feature_extractor, observe, reward
        )
    else:
        from .quadruped.sit_prm import (
            Config as EnvironmentConfig,
            feature_extractor, observe, reward
        )

    history = defaultdict(list)
    steps_history = []

    def progress(num_steps, metrics):
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(f"steps={num_steps:>12,}  reward={float(reward):>10.3f}  {metrics}")

        keys = [
            k for k in metrics
            if k.startswith("eval/episode_reward") and not k.endswith("_std")
        ]
        steps_history.append(num_steps)
        for k in keys:
            history[k].append(float(metrics[k]))

        plt.clf()
        for k, values in history.items():
            plt.plot(steps_history[:len(values)], values, label=k.removeprefix("eval/episode_reward/"))
        plt.xlabel("steps")
        plt.ylabel("reward")
        plt.legend(fontsize="x-small")
        plt.tight_layout()
        plt.savefig("rewards.png", dpi=100)

    mj_model_path = Path("./scene.xml")
    mj_model = MjModel.from_xml_string(
        mj_model_path.read_text()
    )
    
    environment_config = EnvironmentConfig()
    environment = Environment(
        mj_model,
        functools.partial(feature_extractor, environment_config),
        observe,
        reward(environment_config),
        control_dt=environment_config.ctrl_dt,
        substeps=int(environment_config.ctrl_dt / environment_config.sim_dt),
        nconmax=environment_config.nconmax,
        njmax=environment_config.njmax
    )
    if args.curriculum not in ("hind_stand", "sit"):
        perturbation_config = PerturbationConfig(
            velocity_kick=(5.0, 20.0) if args.curriculum == "stability" else (5.0, 10.0)
        )
        environment = PerturbationWrapper(
            environment,
            mj_model,
            perturbation_config,
            ctrl_dt=environment_config.ctrl_dt
        )

    training_config = Config()

    network_factory_config = training_config.network_factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **asdict(network_factory_config)
    )

    num_eval_envs = training_config.num_eval_envs
    full_reset = training_config.full_reset
    training_config = asdict(training_config)

    del training_config["num_eval_envs"]
    del training_config["network_factory"]
    del training_config["full_reset"]

    train = functools.partial(
        ppo.train,
        **training_config,
        network_factory=network_factory,
        seed=1,
        save_checkpoint_path=str(Path("./checkpoints").resolve()),
        restore_checkpoint_path=str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        wrap_env_fn=functools.partial(wrap, full_reset=full_reset),
        randomization_fn=domain_randomize,
        num_eval_envs=num_eval_envs
    )

    make_inference_function, params, _ = train(
        environment=environment,
        progress_fn=progress
    )

    print("Done training.")


if __name__ == "__main__":
    main()

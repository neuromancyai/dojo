
import functools
import os

from dataclasses import dataclass, asdict
from pathlib import Path

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco import MjModel

from .brax import Environment, wrap 
from .utility.dataclasses import default_field
from .quadruped.sit import (
    Config as EnvironmentConfig,
    feature_extractor,
    observe,
    reward
)


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
    randomization_fn: None = None
    network_factory: NetworkFactory = default_field(NetworkFactory())


def main():
    def progress(num_steps, metrics):
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(f"steps={num_steps:>12,}  reward={float(reward):>10.3f}  {metrics}")

    mj_model_path = Path("./scene.xml")
    mj_model = MjModel.from_xml_string(mj_model_path.read_text())
    
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
    del training_config["randomization_fn"]

    train = functools.partial(
        ppo.train,
        **training_config,
        network_factory=network_factory,
        seed=1,
        save_checkpoint_path=str(Path("./checkpoints").resolve()),
        wrap_env_fn=functools.partial(wrap, full_reset=full_reset),
        num_eval_envs=num_eval_envs
    )

    make_inference_function, params, _ = train(
        environment=environment,
        progress_fn=progress
    )

    print("Done training.")


if __name__ == "__main__":
    main()

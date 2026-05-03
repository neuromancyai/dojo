from pathlib import Path

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import NetworkFactory, Params, Policy


def load_checkpoint(
    network_factory: NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    preprocess_observations_fn: 
    path: Path
) -> tuple[Callable[[Params, bool], Policy], Params]:
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )

    make_policy = ppo_networks.make_inference_fn(
        ppo_network,
        compute_value=bootstrap_on_timeout or clipping_epsilon_value is not None,
        use_distributional_critic=use_distributional_critic,
    )

    params = checkpoint.load(restore_checkpoint_path)

    return make_policy, params

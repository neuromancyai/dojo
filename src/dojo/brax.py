from typing import Any

import flax.struct
import jax


from mujoco import MjModel, mjx


@flax.struct.dataclass
class State:
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    info: dict[str, Any]


class Environment:
    def __init__(self, mj_model: MjModel) -> None:
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl="warp")

    def reset(self, rng: jax.Array) -> State:
        pass

    def step(self, state: State, action: jax.Array) -> State:
        pass

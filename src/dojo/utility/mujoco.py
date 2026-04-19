from typing import Optional

import jax

from jax import Array, Device
from mujoco import mjx, MjModel


def read_sensor(model: MjModel, data: mjx.Data, name: str) -> Array:
  id = model.sensor(name).id
  adr = model.sensor_adr[id]
  dim = model.sensor_dim[id]

  return data.sensordata[adr:adr + dim]


def step(
  model: mjx.Model,
  data: mjx.Data,
  action: Array,
  substeps: int
) -> mjx.Data:
  def single(data, _):
    data = data.replace(ctrl=action)
    data = mjx.step(model, data)

    return data, None

  return jax.lax.scan(single, data, (), substeps)[0]


def make_data(
    model: MjModel,
    qpos: Optional[Array] = None,
    qvel: Optional[Array] = None,
    impl: Optional[str] = None,
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[Device] = None
) -> mjx.Data:
  data = mjx.make_data(
      model,
      impl=impl,
      nconmax=nconmax,
      njmax=njmax,
      device=device
  )

  if qpos is not None:
    data = data.replace(qpos=qpos)

  if qvel is not None:
    data = data.replace(qvel=qvel)

  return data

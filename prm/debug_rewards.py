import time

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from dojo.quadruped.sit_prm import Config, feature_extractor, reward as make_reward

m = mujoco.MjModel.from_xml_path("./scene.xml")
#m.opt.gravity[:] = 0
d = mujoco.MjData(m)
mx = mjx.put_model(m)

config = Config()
extractor = feature_extractor(config, m, mx)
reward_fn = jax.jit(make_reward(config))

rng = jax.random.PRNGKey(0)
dx = mjx.put_data(m, d)
features, done, rng = extractor.init(dx, rng)

step_fn = jax.jit(extractor.step)
action = jp.zeros(m.nu)

print("Compiling JIT functions...")
_ = reward_fn(features, done)
_ = step_fn(features, dx, action, rng)
print("Ready.")

last_print = 0.0

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

        dx = mjx.put_data(m, d)
        features, done, rng = step_fn(features, dx, action, rng)

        now = time.time()
        if now - last_print >= 0.5:
            rewards = reward_fn(features, done)
            total = sum(float(v) for v in rewards.values())
            print(f"\n--- rewards (total: {total:.4f}) ---")
            for k, v in rewards.items():
                print(f"  {k:20s}: {float(v):+.4f}")
            last_print = now

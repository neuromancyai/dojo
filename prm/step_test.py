import time

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

com = data.subtree_com[0].copy()
body_positions = np.array([data.xpos[i] for i in range(1, model.nbody)])
geo_center = body_positions.mean(axis=0)
diff = com - geo_center
print(f"COM:        {com}")
print(f"Geo center: {geo_center}")
print(f"Diff:       {diff}")
print(f"Offset:     {np.linalg.norm(diff):.6f} m")
print()
for i in range(1, model.nbody):
    name = model.body(i).name
    mass = model.body_subtreemass[i]
    pos = data.xpos[i]
    print(f"{name:>30s}  mass={mass:.4f}  pos={pos}")

FOOT_NAMES = ("fl", "fr", "rl", "rr")
FOOT_IDS = np.array([model.site(name).id for name in FOOT_NAMES])

def _add_sphere(scn, idx, pos, rgba, radius=0.03):
    mujoco.mjv_initGeom(
        scn.geoms[idx],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0]),
        pos.astype(np.float64),
        np.eye(3).flatten(),
        rgba.astype(np.float32)
    )
    scn.ngeom = max(scn.ngeom, idx + 1)


step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        com = data.subtree_com[0].copy()
        geo_center = np.array([data.xpos[i] for i in range(1, model.nbody)]).mean(axis=0)

        viewer.user_scn.ngeom = 0
        _add_sphere(viewer.user_scn, 0, com, np.array([1.0, 0.0, 0.0, 0.8]))        # red = COM
        _add_sphere(viewer.user_scn, 1, geo_center, np.array([0.0, 0.4, 1.0, 0.8])) # blue = geo center

        viewer.sync()

        if step % 50 == 0:
            z = data.site_xpos[FOOT_IDS, 2]
            print(f"fl={z[0]:.4f}  fr={z[1]:.4f}  rl={z[2]:.4f}  rr={z[3]:.4f}")

        step += 1
        time.sleep(model.opt.timestep)

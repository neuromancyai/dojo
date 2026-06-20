"""
Plain MuJoCo stability test — no ML, no JAX.
Holds home pose with PD control, sweeps mass/inertia/CoM perturbation seeds,
reports tilt direction and survival time.
"""

import numpy as np
import mujoco

SCENE_XML     = "./prm/scene.xml"
N_STEPS       = 1000          # steps (at sim_dt each)
SIM_DT        = 0.004
CTRL_DT       = 0.02
SUBSTEPS      = int(CTRL_DT / SIM_DT)
KP            = 100.0
KD            = 2.0
SEEDS         = [0, 1, 2, 3, 4, 5, 6, 7]
MASS_NOISE    = 0.10
INERTIA_NOISE = 0.15
COM_NOISE     = 0.005          # metres


def perturb(model: mujoco.MjModel, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(1, model.nbody):
        model.body_mass[i]    *= rng.uniform(1 - MASS_NOISE,    1 + MASS_NOISE)
        model.body_inertia[i] *= rng.uniform(1 - INERTIA_NOISE, 1 + INERTIA_NOISE)
        model.body_ipos[i]    += rng.uniform(-COM_NOISE, COM_NOISE, 3)


def run(model: mujoco.MjModel, n_steps: int):
    data = mujoco.MjData(model)
    home_qpos = model.keyframe("home").qpos.copy()
    home_ctrl = model.keyframe("home").ctrl.copy()
    data.qpos[:] = home_qpos
    data.ctrl[:] = home_ctrl
    mujoco.mj_forward(model, data)

    upvector_sensor = model.sensor("upvector").id
    sensor_adr      = model.sensor_adr[upvector_sensor]

    tilts_x, tilts_y = [], []
    survived = 0

    for step in range(n_steps):
        if step % SUBSTEPS == 0:
            # PD: hold home joint positions
            q    = data.qpos[7:]
            qdot = data.qvel[6:]
            data.ctrl[:] = home_ctrl + KP * (home_qpos[7:] - q) - KD * qdot

        mujoco.mj_step(model, data)

        upvec = data.sensordata[sensor_adr : sensor_adr + 3]
        tilts_x.append(float(upvec[0]))
        tilts_y.append(float(upvec[1]))

        if upvec[2] < 0.5:   # fallen
            break
        survived += 1

    return dict(
        survived=survived,
        tilt_x_mean=float(np.mean(tilts_x)),
        tilt_y_mean=float(np.mean(tilts_y)),
        tilt_mag_mean=float(np.mean(np.hypot(tilts_x, tilts_y))),
        tilt_mag_max=float(np.max(np.hypot(tilts_x, tilts_y))),
    )


def perturb_mass_only(model, seed):
    rng = np.random.default_rng(seed)
    for i in range(1, model.nbody):
        model.body_mass[i] *= rng.uniform(1 - MASS_NOISE, 1 + MASS_NOISE)

def perturb_inertia_only(model, seed):
    rng = np.random.default_rng(seed)
    for i in range(1, model.nbody):
        model.body_inertia[i] *= rng.uniform(1 - INERTIA_NOISE, 1 + INERTIA_NOISE)

def perturb_com_only(model, seed):
    rng = np.random.default_rng(seed)
    for i in range(1, model.nbody):
        model.body_ipos[i] += rng.uniform(-COM_NOISE, COM_NOISE, 3)


def section(label, perturb_fn):
    results = []
    print(f"\n--- {label} ---")
    print(f"{'seed':>6} {'survived':>9} {'tilt_x':>8} {'tilt_y':>8} {'|tilt|mean':>11} {'|tilt|max':>10}")
    for seed in SEEDS:
        m = mujoco.MjModel.from_xml_path(SCENE_XML)
        perturb_fn(m, seed)
        r = run(m, N_STEPS)
        results.append(r)
        print(f"{seed:>6} {r['survived']:>9} {r['tilt_x_mean']:>8.4f} {r['tilt_y_mean']:>8.4f} {r['tilt_mag_mean']:>11.4f} {r['tilt_mag_max']:>10.4f}")
    tilt_means = [r["tilt_mag_mean"] for r in results]
    tilt_xs    = [r["tilt_x_mean"] for r in results]
    tilt_ys    = [r["tilt_y_mean"] for r in results]
    print(f"  |tilt| mean={np.mean(tilt_means):.4f}  max={max(tilt_means):.4f}")
    print(f"  tilt_x bias={np.mean(tilt_xs):+.4f}  tilt_y bias={np.mean(tilt_ys):+.4f}")


def main():
    # print nominal body masses and inertias
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    print(f"{'body':>30} {'mass':>8} {'Ixx':>10} {'Iyy':>10} {'Izz':>10}  ipos")
    print("-" * 85)
    for i in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        mass = m.body_mass[i]
        ixx, iyy, izz = m.body_inertia[i]
        ipos = m.body_ipos[i]
        print(f"{name:>30} {mass:>8.4f} {ixx:>10.5f} {iyy:>10.5f} {izz:>10.5f}  {np.round(ipos, 4)}")

    print()
    r = run(m, N_STEPS)
    print(f"baseline  survived={r['survived']}  tilt_x={r['tilt_x_mean']:+.4f}  tilt_y={r['tilt_y_mean']:+.4f}  |tilt|mean={r['tilt_mag_mean']:.4f}")

    section("mass only   (±10%)",           perturb_mass_only)
    section("inertia only (±15%)",          perturb_inertia_only)
    section("CoM offset only (±5mm)",       perturb_com_only)
    section("combined",                     perturb)


if __name__ == "__main__":
    main()

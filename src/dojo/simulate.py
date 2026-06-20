
import argparse
import functools
import time

from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco import MjModel

from .brax import Environment, wrap
from .perturbation import Config as PerturbationConfig, PerturbationWrapper
from .quadruped.sit_prm import (
    Config as EnvironmentConfig,
    feature_extractor,
    observe,
    reference_horizon_steps,
    reward
)

from .training import Config as TrainingConfig


_keys_held: set = set()


def _on_press(key):
    _keys_held.add(key)


def _on_release(key):
    _keys_held.discard(key)


def _get_command() -> jp.ndarray:
    def held(char):
        return keyboard.KeyCode.from_char(char) in _keys_held

    z = (1.0 if held('s') or keyboard.Key.down in _keys_held else 0.0)

    # vx  =  (-0.8 if held('w') else 0.0) + (0.8 if held('s') else 0.0)
    # vy  =  (-0.6 if held('a') else 0.0) + (0.6 if held('d') else 0.0)
    # yaw = (1.0 if keyboard.Key.left in _keys_held else 0.0) + \
    #       (-1.0 if keyboard.Key.right in _keys_held else 0.0)
    # return jp.array([vx, vy, yaw])
    return jp.array([z])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Record video to file")
    parser.add_argument("--camera", default="track", choices=["track", "top", "side", "back", "dramatic"], help="Camera to use for recording")
    parser.add_argument("--output", default="recording.mp4", help="Output video filename")
    args = parser.parse_args()

    mj_model_path = Path("./scene.xml")
    mj_model = MjModel.from_xml_path(str(mj_model_path))
    #mj_model.opt.gravity[:] = 0
    mj_data = mujoco.MjData(mj_model)

    #_FOOT_NAMES = ("FL", "FR", "HL", "HR")
    #_FOOT_SITE_IDS = np.array([mj_model.site(n).id for n in _FOOT_NAMES])
    # trot diagonal pairs: FL+HR = red, FR+HL = blue
    _FOOT_COLORS = np.array([
        [1.0, 0.35, 0.1, 1.0],
        [0.1, 0.45, 1.0, 1.0],
        [0.1, 0.45, 1.0, 1.0],
        [1.0, 0.35, 0.1, 1.0],
    ])

    environment_config = EnvironmentConfig()
    environment: Any = Environment(
        mj_model,
        functools.partial(feature_extractor, environment_config),
        observe,
        reward(environment_config),
        control_dt=environment_config.ctrl_dt,
        substeps=int(environment_config.ctrl_dt / environment_config.sim_dt),
        nconmax=environment_config.nconmax,
        njmax=environment_config.njmax
    )
    # environment = PerturbationWrapper(
    #     environment,
    #     mj_model,
    #     PerturbationConfig(velocity_kick=(8.0, 10.0)),
    #     ctrl_dt=environment_config.ctrl_dt
    # )

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
    def inject_perturbation(state):
        info = dict(state.info)
        info["pert_triggered"] = jp.bool_(True)
        return state.replace(info=info)

    @jax.jit
    def inject_command(state, command):
        old_features = state.info["features"]
        command_changed = jp.any(jp.abs(command - old_features.current_command) > 1e-6)
        z_range = environment_config.geometry.z_range
        target_body_z = command[0] * (z_range[0] - z_range[1]) + z_range[1]
        trajectory_start_z = jp.where(
            command_changed,
            old_features.body_z_ref,
            old_features.previous_body_z
        )
        features = state.info["features"].replace(
            current_command=command,
            previous_command=command,
            steps_since_last_command=jp.where(
                command_changed,
                jp.int32(0),
                old_features.steps_since_last_command
            ),
            previous_body_z=jp.where(
                command_changed,
                old_features.body_z_ref,
                old_features.previous_body_z
            ),
            body_linvel_z_ref=jp.where(
                command_changed,
                jp.zeros(()),
                old_features.body_linvel_z_ref
            ),
            trajectory_phase=jp.where(
                command_changed,
                jp.zeros(()),
                old_features.trajectory_phase
            ),
            trajectory_blend=jp.where(
                command_changed,
                jp.zeros(()),
                old_features.trajectory_blend
            ),
            trajectory_horizon_steps=jp.where(
                command_changed,
                reference_horizon_steps(
                    environment_config,
                    jp.abs(target_body_z - trajectory_start_z)
                ),
                old_features.trajectory_horizon_steps
            )
        )
        new_obs = observe(features, state.done)
        info = dict(state.info)
        info["features"] = features
        return state.replace(obs=new_obs, info=info)

    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    state = environment.reset(reset_key)
    state = inject_command(state, _get_command())

    print("Warming up JIT...")
    rng, warmup_key = jax.random.split(rng)

    action, _ = inference_fn(state.obs, warmup_key)
    state = step_fn(state, action)
    state = inject_command(state, _get_command())
    jax.block_until_ready(state)
    print("Ready.")
    print("Hold S or ARROW DOWN to sit. Release to return to idle.")

    if args.record:
        renderer = mujoco.Renderer(mj_model, height=1080, width=1920)
        frames = []

    dramatic_cam = mujoco.MjvCamera()
    dramatic_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    dramatic_cam.trackbodyid = mj_model.body("root").id
    dramatic_cam.distance = 2.5
    dramatic_cam.elevation = -20   # degrees, slightly looking down
    dramatic_cam.azimuth = 180     # start behind robot, updated each frame
    root_body_id = mj_model.body("root").id

    _TRAIL_MAX = 60
    #_foot_trails = [[] for _ in _FOOT_NAMES]
    _BODY_Z_REF_COLOR = np.array([0.0, 0.85, 1.0, 0.9])

    def _get_rz_np(phi, swing_height):
        def cubic_bezier(y_start, y_end, x):
            bezier = x ** 3 + 3 * (x ** 2 * (1 - x))
            return y_start + (y_end - y_start) * bezier
        x = (phi + np.pi) / (2 * np.pi)
        stance = cubic_bezier(0.0, swing_height, 2 * x)
        swing = cubic_bezier(swing_height, 0.0, 2 * x - 1)
        return np.where(x <= 0.5, stance, swing)

    def _add_sphere(scn, pos, rgba, radius=0.025):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]),
            pos.astype(np.float64),
            np.eye(3).flatten(),
            rgba.astype(np.float32)
        )
        scn.ngeom += 1

    def _add_segment(scn, p1, p2, rgba, width=0.004):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.eye(3).flatten(),
            rgba.astype(np.float32)
        )
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.emission = 0.5
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
            np.asarray(p1, dtype=np.float64),
            np.asarray(p2, dtype=np.float64)
        )
        scn.ngeom += 1

    _LEGS = ["FL", "RL", "FR", "RR"]
    _JOINT_TYPES = ["haa", "hfe", "kfe"]
    # layout: [haa, hfe, kfe] * 4 legs
    _LEG_COLORS = ["#e05c2a", "#2a7be0", "#2ab85c", "#b82ab8"]

    stats_qvel: dict[str, dict[str, list]] = {jt: {leg: [] for leg in _LEGS} for jt in _JOINT_TYPES}
    stats_force: dict[str, dict[str, list]] = {jt: {leg: [] for leg in _LEGS} for jt in _JOINT_TYPES}
    stats_linvel: list[np.ndarray] = []
    stats_body_z: list[tuple[float, float]] = []

    step_count = 0
    with keyboard.Listener(on_press=_on_press, on_release=_on_release):
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running():
                rng, step_key = jax.random.split(rng)
                action, _ = inference_fn(state.obs, step_key)
                if step_count % 10 == 0:
                    a = np.array(action)
                    upvec = np.array(state.info["features"].gravity)
                    deltas = np.array(state.info["features"].joint_angle_deltas)
                    body_z = float(state.info["features"].body_z)
                    body_z_ref = float(state.info["features"].body_z_ref)
                    body_linvel_z_ref = float(state.info["features"].body_linvel_z_ref)
                    command = np.array(state.info["features"].current_command)
                    linvel_z = float(state.info["features"].global_linvel[2])
                    knee_heights = np.array(state.info["features"].knee_height)
                    knee_height_raw = float(
                        (knee_heights < environment_config.geometry.knee_min_height).sum()
                    )
                    knee_height_reward = float(
                        state.metrics.get("reward/knee_height", jp.nan)
                    )
                    nu = environment.action_size
                    last_action = np.array(state.info["features"].action_history[:nu])
                    print(
                        f"action={np.round(a, 3)} | "
                        f"command={np.round(command, 3)} | "
                        f"action_history[-1]={np.round(last_action, 3)} | "
                        f"upvector={upvec} | "
                        f"joint_deltas={np.round(deltas, 3)} | "
                        f"body_z={body_z:.4f} | "
                        f"body_z_ref={body_z_ref:.4f} | "
                        f"linvel_z={linvel_z:.4f} | "
                        f"linvel_z_ref={body_linvel_z_ref:.4f} | "
                        f"knee_h={np.round(knee_heights, 4)} | "
                        f"knee_raw={knee_height_raw:.4f} | "
                        f"knee_reward={knee_height_reward:.4f}"
                    )

                qvel = np.abs(np.array(state.info["features"].joint_qvel))
                aforce = np.abs(np.array(state.info["features"].actuator_force))
                for li, leg in enumerate(_LEGS):
                    for ji, jt in enumerate(_JOINT_TYPES):
                        idx = li * 3 + ji
                        stats_qvel[jt][leg].append(float(qvel[idx]))
                        stats_force[jt][leg].append(float(aforce[idx]))
                stats_linvel.append(np.array(state.info["features"].local_linvel))
                stats_body_z.append((
                    float(state.info["features"].body_z),
                    float(state.info["features"].body_z_ref)
                ))

                step_count += 1
                if keyboard.Key.ctrl_l in _keys_held or keyboard.Key.ctrl_r in _keys_held:
                    state = inject_perturbation(state)
                state = step_fn(state, action)
                state = inject_command(state, _get_command())

                #if state.done:
                #    rng, reset_key = jax.random.split(rng)
                #    state = environment.reset(reset_key)
                #    state = inject_command(state, _get_command())

                mj_data.qpos[:] = state.data.qpos
                mj_data.qvel[:] = state.data.qvel
                mujoco.mj_forward(mj_model, mj_data)

                # foot_positions = mj_data.site_xpos[_FOOT_SITE_IDS].copy()
                # for i, pos in enumerate(foot_positions):
                #     _foot_trails[i].append(pos)
                #     if len(_foot_trails[i]) > _TRAIL_MAX:
                #         _foot_trails[i].pop(0)

                #phase = np.array(state.info["features"].phase)
                #foot_height = float(state.info["features"].foot_height)
                #ideal_z = _get_rz_np(phase, foot_height)

                viewer.user_scn.ngeom = 0
                body_z_ref = float(state.info["features"].body_z_ref)
                root_xy = mj_data.xpos[root_body_id, :2].copy()
                ref_center = np.array([root_xy[0], root_xy[1], body_z_ref])
                _add_segment(
                    viewer.user_scn,
                    ref_center + np.array([-0.18, 0.0, 0.0]),
                    ref_center + np.array([0.18, 0.0, 0.0]),
                    _BODY_Z_REF_COLOR,
                    width=0.006
                )
                _add_sphere(
                    viewer.user_scn,
                    ref_center,
                    _BODY_Z_REF_COLOR,
                    radius=0.018
                )
                # for i, (trail, color) in enumerate(zip(_foot_trails, _FOOT_COLORS)):
                #     for k in range(1, len(trail)):
                #         alpha = k / len(trail)
                #         c = color.copy()
                #         c[3] = alpha * 0.8
                #         _add_segment(viewer.user_scn, trail[k - 1], trail[k], c)
                #     if trail:
                #         _add_sphere(viewer.user_scn, trail[-1], color)
                #         # ideal_pos = np.array([trail[-1][0], trail[-1][1], ideal_z[i]])
                #         # ideal_color = np.array([*color[:3], 0.5])
                #         # _add_sphere(viewer.user_scn, ideal_pos, ideal_color, radius=0.02)

                viewer.sync()

                if args.record:
                    if args.camera == "dramatic":
                        w, x, y, z = mj_data.qpos[3:7]
                        yaw_deg = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
                        dramatic_cam.azimuth = yaw_deg + 180
                        renderer.update_scene(mj_data, camera=dramatic_cam)
                    else:
                        renderer.update_scene(mj_data, camera=args.camera)
                    frames.append(renderer.render())

                time.sleep(environment_config.ctrl_dt)

    fps = int(1 / environment_config.ctrl_dt)
    if args.record:
        imageio.mimsave(args.output, frames, fps=fps)
        print(f"Saved {args.output} ({len(frames)} frames @ {fps} fps)")

    steps = np.arange(len(stats_qvel["haa"]["FL"]))

    for fname, stats, ylabel, title in [
        ("rewards_vel.png", stats_qvel, "|joint qvel| (rad/s)", "Joint Velocity per Leg"),
        ("rewards_torque.png", stats_force, "|actuator force| (Nm)", "Torque per Leg"),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(title)
        for ai, jt in enumerate(_JOINT_TYPES):
            ax = axes[ai]
            for leg, color in zip(_LEGS, _LEG_COLORS):
                ax.plot(steps, stats[jt][leg], label=leg, color=color, linewidth=0.8, alpha=0.85)
            ax.set_ylabel(f"{jt}\n{ylabel}", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, ncol=4)
            ax.grid(True, linewidth=0.4, alpha=0.5)
        axes[-1].set_xlabel("ctrl step")
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print(f"Saved {fname}")

    linvel = np.stack(stats_linvel)
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (label, color) in enumerate(zip(["x", "y", "z"], ["#e05c2a", "#2a7be0", "#2ab85c"])):
        ax.plot(linvel[:, i], label=label, color=color, linewidth=0.8, alpha=0.85)
    ax.set_ylabel("local_linvel (m/s)")
    ax.set_xlabel("ctrl step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig("linvel.png", dpi=150)
    print("Saved linvel.png")

    body_z = np.array(stats_body_z)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(body_z[:, 0], label="body_z", color="#e05c2a", linewidth=0.9)
    ax.plot(body_z[:, 1], label="body_z_ref", color="#00bcd4", linewidth=0.9)
    ax.set_ylabel("height (m)")
    ax.set_xlabel("ctrl step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig("body_z_ref.png", dpi=150)
    print("Saved body_z_ref.png")


if __name__ == "__main__":
    main()

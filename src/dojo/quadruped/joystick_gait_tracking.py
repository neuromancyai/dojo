from dataclasses import dataclass
from typing import NamedTuple

import flax.struct
import jax
import numpy as np

from jax import Array, numpy as jp
from jaxtyping import Bool, Float, Int
from mujoco import MjModel, mjx

from ..environment import Done, FeatureExtractor, Reward, Rng
from ..utility.dataclasses import default_field
from ..utility.mujoco import read_sensor


# Foot order: FL, FR, HL, HR
_PHASES = np.array([
    [0, np.pi, np.pi, 0],                       # trot
    [0, 0.5 * np.pi, np.pi, 1.5 * np.pi],       # walk
    [0, np.pi, 0, np.pi],                       # pace
    [0, 0, np.pi, np.pi],                       # bound
    [0, 0, 0, 0],                               # pronk
])


def _get_rz(phi: Array, swing_height: Array) -> Array:
    def cubic_bezier(y_start, y_end, x):
        bezier = x ** 3 + 3 * (x ** 2 * (1 - x))
        return y_start + (y_end - y_start) * bezier

    x = (phi + jp.pi) / (2 * jp.pi)
    stance = cubic_bezier(0.0, swing_height, 2 * x)
    swing = cubic_bezier(swing_height, 0.0, 2 * x - 1)
    return jp.where(x <= 0.5, stance, swing)


@dataclass
class Config:

    @dataclass
    class ObservationNoise:

        @dataclass
        class Scale:
            joint_pos: float = 0.05
            gyro: float = 0.1
            gravity: float = 0.03

        scale: Scale = default_field(Scale())

    @dataclass
    class Reward:
        @dataclass
        class Scale:
            tracking_linvel: float = 0.5
            tracking_angvel: float = 0.5
            feet_phase: float = 2.0
            linvel_z: float = -0.5
            angvel_xy: float = -0.5
            hip_splay: float = -0.5
            termination: float = -400.0

        scale: Scale = default_field(Scale())
        tracking_sigma: float = 0.25

    @dataclass
    class Command:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-0.5, 0.5)
        ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

    @dataclass
    class Gait:
        frequency: tuple[float, float] = (0.5, 4.0)
        foot_height: tuple[float, float] = (0.08, 0.4)
        num_gaits: int = 5

    @dataclass
    class Sensor:
        accelerometer: str = "accelerometer"
        local_linvel: str = "local_linvel"
        global_linvel: str = "global_linvel"
        global_angvel: str = "global_angvel"
        gyro: str = "gyro"
        gravity: str = "upvector"
        feet_sites: tuple[str, str, str, str] = ("FL", "FR", "HL", "HR")
        feet_contacts: tuple[str, str, str, str] = (
            "FL_floor_found",
            "FR_floor_found",
            "HL_floor_found",
            "HR_floor_found"
        )

    @dataclass
    class Geometry:
        body: str = "body"

    ctrl_dt: float = 0.02
    sim_dt: float = 0.004
    episode_length: float = 1000
    early_termination: bool = True
    action_repeat: int = 1
    action_scale: float = 0.6
    history_len: int = 3
    obs_noise: ObservationNoise = default_field(ObservationNoise())
    reward: Reward = default_field(Reward())
    command: Command = default_field(Command())
    gait: Gait = default_field(Gait())
    geometry: Geometry = default_field(Geometry())
    nconmax: int = 4 * 8192
    njmax: int = 64
    sensor: Sensor = default_field(Sensor())


@flax.struct.dataclass
class Features:
    steps_since_last_command: Int[Array, ""]
    previous_command: Float[Array, "3"]
    current_command: Float[Array, "3"]

    motor_targets: Float[Array, "12"]
    qpos_error_history: Float[Array, "36"]

    feet_contacts: Bool[Array, "4"]

    phase: Float[Array, "4"]
    phase_dt: Float[Array, ""]
    gait_freq: Float[Array, ""]
    gait_index: Int[Array, ""]
    foot_height: Float[Array, ""]

    accelerometer: Float[Array, "3"]
    local_linvel: Float[Array, "3"]
    global_linvel: Float[Array, "3"]
    global_angvel: Float[Array, "3"]
    gyro: Float[Array, "3"]
    noisy_gyro: Float[Array, "3"]
    gravity: Float[Array, "3"]
    noisy_gravity: Float[Array, "3"]
    joint_angles: Float[Array, "12"]
    noisy_joint_angles: Float[Array, "12"]

    feet_z: Float[Array, "4"]

    body_force: Float[Array, "3"]
    actuator_force: Float[Array, "12"]
    joint_qvel: Float[Array, "12"]
    joint_angle_deltas: Float[Array, "12"]
    noisy_joint_angle_deltas: Float[Array, "12"]


def _sample_command(config: Config.Command, rng: Rng) -> tuple[Array, Rng]:
    rng, key_1, key_2, key_3, key_4 = jax.random.split(rng, 5)

    lin_vel_x = jax.random.uniform(key_1, minval=config.lin_vel_x[0], maxval=config.lin_vel_x[1])
    lin_vel_y = jax.random.uniform(key_2, minval=config.lin_vel_y[0], maxval=config.lin_vel_y[1])
    ang_vel_yaw = jax.random.uniform(key_3, minval=config.ang_vel_yaw[0], maxval=config.ang_vel_yaw[1])
    command = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])

    return (
        jp.where(jax.random.bernoulli(key_4, 0.1), jp.zeros(3), command),
        rng
    )


class _SensorReadout(NamedTuple):
    accelerometer: Float[Array, "3"]
    local_linvel: Float[Array, "3"]
    global_linvel: Float[Array, "3"]
    global_angvel: Float[Array, "3"]
    gyro: Float[Array, "3"]
    noisy_gyro: Float[Array, "3"]
    gravity: Float[Array, "3"]
    noisy_gravity: Float[Array, "3"]
    joint_angles: Float[Array, "12"]
    noisy_joint_angles: Float[Array, "12"]
    feet_contacts: Bool[Array, "4"]


def _read_sensors(
    config: Config,
    mj_model: MjModel,
    data: mjx.Data,
    rng: Rng
) -> tuple[_SensorReadout, Rng]:
    rng, key_1 = jax.random.split(rng)
    accelerometer = read_sensor(mj_model, data, config.sensor.accelerometer)
    local_linvel = read_sensor(mj_model, data, config.sensor.local_linvel)
    global_linvel = read_sensor(mj_model, data, config.sensor.global_linvel)
    global_angvel = read_sensor(mj_model, data, config.sensor.global_angvel)
    gyro = read_sensor(mj_model, data, config.sensor.gyro)
    noisy_gyro = gyro + \
        (2 * jax.random.uniform(key_1, shape=gyro.shape) - 1) * \
        config.obs_noise.scale.gyro

    rng, key_2 = jax.random.split(rng)
    gravity = read_sensor(mj_model, data, config.sensor.gravity)
    noisy_gravity = gravity + \
        (2 * jax.random.uniform(key_2, shape=gravity.shape) - 1) * \
        config.obs_noise.scale.gravity

    rng, key_3 = jax.random.split(rng)
    joint_angles = data.qpos[7:]
    noisy_joint_angles = joint_angles + \
        (2 * jax.random.uniform(key_3, shape=joint_angles.shape) - 1) * \
        config.obs_noise.scale.joint_pos

    feet_contacts = jp.array([
        read_sensor(mj_model, data, name).squeeze() > 0
        for name in config.sensor.feet_contacts
    ])

    return (
        _SensorReadout(
            accelerometer,
            local_linvel,
            global_linvel,
            global_angvel,
            gyro,
            noisy_gyro,
            gravity,
            noisy_gravity,
            joint_angles,
            noisy_joint_angles,
            feet_contacts
        ),
        rng
    )


def feature_extractor(
    config: Config,
    mj_model: MjModel,
    mjx_model: mjx.Model
) -> FeatureExtractor[mjx.Data, Features]:
    default_pose = mj_model.keyframe("home").qpos[7:]
    lower_control_limits = mj_model.actuator_ctrlrange[:, 0]
    upper_control_limits = mj_model.actuator_ctrlrange[:, 1]
    body_id = mj_model.body(config.geometry.body).id
    feet_site_ids = np.array(
        [mj_model.site(name).id for name in config.sensor.feet_sites]
    )
    phases = jp.array(_PHASES)

    def init(data: mjx.Data, rng: Rng) -> tuple[Features, Done, Rng]:
        readout, rng = _read_sensors(config, mj_model, data, rng)
        previous_command, rng = _sample_command(config.command, rng)

        rng, key_gait, key_freq, key_fh = jax.random.split(rng, 4)
        gait_index = jax.random.randint(key_gait, shape=(), minval=0, maxval=config.gait.num_gaits)
        gait_freq = jax.random.uniform(key_freq, minval=config.gait.frequency[0], maxval=config.gait.frequency[1])
        foot_height = jax.random.uniform(key_fh, minval=config.gait.foot_height[0], maxval=config.gait.foot_height[1])
        phase = phases[gait_index]
        phase_dt = 2 * jp.pi * config.ctrl_dt * gait_freq

        body_force = data.xfrc_applied[body_id, :3]
        actuator_force = data.actuator_force
        joint_qvel = data.qvel[6:]
        joint_angle_deltas = readout.joint_angles - default_pose
        noisy_joint_angle_deltas = readout.noisy_joint_angles - default_pose
        feet_z = data.site_xpos[feet_site_ids][..., -1]

        done = jp.zeros((), dtype=jp.bool_)

        return (
            Features(
                steps_since_last_command=jp.zeros((), dtype=jp.int32),
                previous_command=previous_command,
                current_command=previous_command,

                motor_targets=jp.zeros(mjx_model.nu),
                qpos_error_history=jp.zeros(config.history_len * mjx_model.nu),

                feet_contacts=readout.feet_contacts,

                phase=phase,
                phase_dt=phase_dt,
                gait_freq=gait_freq,
                gait_index=gait_index,
                foot_height=foot_height,

                accelerometer=readout.accelerometer,
                local_linvel=readout.local_linvel,
                global_linvel=readout.global_linvel,
                global_angvel=readout.global_angvel,
                gyro=readout.gyro,
                noisy_gyro=readout.noisy_gyro,
                gravity=readout.gravity,
                noisy_gravity=readout.noisy_gravity,
                joint_angles=readout.joint_angles,
                noisy_joint_angles=readout.noisy_joint_angles,

                feet_z=feet_z,
                body_force=body_force,
                actuator_force=actuator_force,
                joint_qvel=joint_qvel,
                joint_angle_deltas=joint_angle_deltas,
                noisy_joint_angle_deltas=noisy_joint_angle_deltas
            ),
            done,
            rng
        )

    def step(
        previous: Features,
        data: mjx.Data,
        action: Array,
        rng: Rng
    ) -> tuple[Features, Done, Rng]:
        motor_targets = default_pose + config.action_scale * action
        motor_targets = jp.clip(motor_targets, lower_control_limits, upper_control_limits)

        readout, rng = _read_sensors(config, mj_model, data, rng)

        qpos_error_history = (
            jp.roll(previous.qpos_error_history, mjx_model.nu)
                .at[:mjx_model.nu]
                .set(readout.noisy_joint_angles - motor_targets)
        )

        steps_since_last_command = previous.steps_since_last_command + 1
        new_command, rng = _sample_command(config.command, rng)
        previous_command = previous.current_command
        current_command = jp.where(
            steps_since_last_command > 200,
            new_command,
            previous.current_command
        )
        steps_since_last_command = jp.where(
            steps_since_last_command > 200,
            jp.int32(0),
            steps_since_last_command
        )

        phase = jp.fmod(previous.phase + previous.phase_dt + jp.pi, 2 * jp.pi) - jp.pi

        body_force = data.xfrc_applied[body_id, :3]
        actuator_force = data.actuator_force
        joint_qvel = data.qvel[6:]
        joint_angle_deltas = readout.joint_angles - default_pose
        noisy_joint_angle_deltas = readout.noisy_joint_angles - default_pose
        feet_z = data.site_xpos[feet_site_ids][..., -1]

        done = jp.where(
            config.early_termination,
            readout.gravity[-1] < 0.85,
            jp.zeros((), dtype=jp.bool_)
        )

        return (
            Features(
                steps_since_last_command=steps_since_last_command,
                previous_command=previous_command,
                current_command=current_command,

                motor_targets=motor_targets,
                qpos_error_history=qpos_error_history,

                feet_contacts=readout.feet_contacts,

                phase=phase,
                phase_dt=previous.phase_dt,
                gait_freq=previous.gait_freq,
                gait_index=previous.gait_index,
                foot_height=previous.foot_height,

                accelerometer=readout.accelerometer,
                local_linvel=readout.local_linvel,
                global_linvel=readout.global_linvel,
                global_angvel=readout.global_angvel,
                gyro=readout.gyro,
                noisy_gyro=readout.noisy_gyro,
                gravity=readout.gravity,
                noisy_gravity=readout.noisy_gravity,
                joint_angles=readout.joint_angles,
                noisy_joint_angles=readout.noisy_joint_angles,

                feet_z=feet_z,
                body_force=body_force,
                actuator_force=actuator_force,
                joint_qvel=joint_qvel,
                joint_angle_deltas=joint_angle_deltas,
                noisy_joint_angle_deltas=noisy_joint_angle_deltas
            ),
            done,
            rng
        )

    return FeatureExtractor(init=init, step=step)


def observe(features: Features, _: Done) -> dict[str, Array]:
    policy = jp.hstack([
        features.noisy_gyro,
        features.noisy_gravity,
        features.noisy_joint_angles,
        features.qpos_error_history,
        features.feet_contacts.astype(jp.float32),
        jp.cos(features.phase),
        jp.sin(features.phase),
        jp.atleast_1d(features.gait_freq),
        jp.atleast_1d(jp.array(features.gait_index, dtype=jp.float32)),
        jp.atleast_1d(features.foot_height),
        features.current_command
    ])

    return {
        "policy": policy,
        "value": jp.hstack([
            policy,
            features.gyro,
            features.gravity,
            features.global_linvel,
            features.global_angvel,
            features.joint_angle_deltas,
            features.joint_qvel,
            features.actuator_force,
            features.body_force
        ])
    }


def reward(config: Config) -> Reward[Features]:
    config = config.reward
    hip_indices = jp.array([0, 3, 6, 9])

    def call(features: Features, done: Done) -> dict[str, Array]:
        def tracking_linvel():
            error = jp.sum(jp.square(features.previous_command[:2] - features.local_linvel[:2]))
            return jp.exp(-error / config.tracking_sigma)

        def tracking_angvel():
            error = jp.square(features.previous_command[2] - features.gyro[2])
            return jp.exp(-error / config.tracking_sigma)

        def feet_phase():
            rz = _get_rz(features.phase, features.foot_height)
            error = jp.sum(jp.square(features.feet_z - rz))
            return jp.exp(-error / 0.1)

        def linvel_z():
            return jp.square(features.global_linvel[2])

        def angvel_xy():
            return jp.sum(jp.square(features.global_angvel[:2]))

        def hip_splay():
            return jp.sum(jp.square(features.joint_angle_deltas[hip_indices]))

        def termination():
            return done.astype(jp.float32)

        terms = {
            "tracking_linvel": tracking_linvel(),
            "tracking_angvel": tracking_angvel(),
            "feet_phase": feet_phase(),
            "linvel_z": linvel_z(),
            "angvel_xy": angvel_xy(),
            "hip_splay": hip_splay(),
            "termination": termination()
        }

        return {k: v * getattr(config.scale, k) for k, v in terms.items()}

    return call

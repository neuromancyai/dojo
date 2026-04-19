from dataclasses import dataclass
from typing import NamedTuple

import flax.struct
import jax
import numpy as np

from jax import Array, numpy as jp
from jaxtyping import Bool, Float, Int
from mujoco import MjModel, mjx

from ...environment import Done, FeatureExtractor, Reward, Rng
from ...utility.dataclasses import default_field
from ...utility.mujoco import read_sensor


@dataclass
class Config:

    @dataclass
    class ObservationNoise:

        @dataclass
        class Scale:
            joint_pos: float = 0.05
            gyro: float = 0.1
            gravity: float = 0.03
            feet_pos: tuple[float, float, float] = (0.01, 0.005, 0.02)

        scale: Scale = default_field(Scale())

    @dataclass
    class Reward:

        @dataclass
        class Scale:
            tracking_linvel: float = 1.5
            tracking_angvel: float = 0.8
            linvel_z: float = -2.0
            angvel_xy: float = -0.05
            orientation: float = -5.0
            posture: float = 1.0
            termination: float = -1.0
            torques: float = -0.0002
            action_rate: float = -0.01
            energy: float = -0.001
            feet_slip: float = -0.1
            feet_clearance: float = -2.0
            feet_height: float = -0.1
            feet_air_time: float = 0.1

        scale: Scale = default_field(Scale())
        tracking_sigma: float = 0.25
        max_foot_height: float = 0.12

    @dataclass
    class Perturbation:
        enable: bool = False
        velocity_kick: tuple[float, float] = (0.0, 3.0)
        kick_durations: tuple[float, float ] = (0.05, 0.2)
        kick_wait_times: tuple[float, float] = (1.0, 3.0)

    @dataclass
    class Command:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-0.8, 0.8)
        ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

    @dataclass
    class Sensor:
        accelerometer: str = "accelerometer"
        local_linvel: str = "local_linvel"
        global_linvel: str = "global_linvel"
        global_angvel: str = "global_angvel"
        gyro: str = "gyro"
        gravity: str = "upvector"
        feet_sites: tuple[str, str, str, str] = (
            "FL",
            "FR",
            "HL",
            "HR"
        )

        feet_pos: tuple[str, str, str, str] = (
            "FL_pos",
            "FR_pos",
            "HL_pos",
            "HR_pos"
        )

        feet_contacts: tuple[str, str, str, str] = (
            "FL_floor_found",
            "FR_floor_found",
            "HL_floor_found",
            "HR_floor_found"
        )

        foot_linvel: tuple[str, str, str, str] = (
            "FL_global_linvel",
            "FR_global_linvel",
            "HL_global_linvel",
            "HR_global_linvel"
        )
    
    @dataclass
    class Geometry:
        body: str = "body"

    ctrl_dt: float = 0.02
    sim_dt: float = 0.004
    episode_length: float = 1000
    kp: float = 300.0
    kd: float = 1.0
    early_termination: bool = True
    action_repeat: int = 1
    action_scale: float = 0.3
    history_len: int = 3
    obs_noise: ObservationNoise = default_field(ObservationNoise())
    reward: Reward = default_field(Reward())
    pert: Perturbation = default_field(Perturbation())
    command: Command = default_field(Command())
    geometry: Geometry = default_field(Geometry())
    nconmax: int = 4 * 8192
    njmax: int = 32
    sensor: Sensor = default_field(Sensor())


@flax.struct.dataclass
class Features:
    steps_since_last_command: Int[Array, ""]
    previous_command: Float[Array, "3"]
    current_command: Float[Array, "3"]

    action_history: Float[Array, "36"]
    motor_targets: Float[Array, "12"]
    qpos_error_history: Float[Array, "36"]

    first_contact: Bool[Array, "4"]
    feet_air_time: Float[Array, "4"]
    swing_peak: Float[Array, "4"]

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
    feet_pos: Float[Array, "12"]
    noisy_feet_pos: Float[Array, "12"]
    feet_contacts: Bool[Array, "4"]
    foot_linvel: Float[Array, "12"]

    feet_z: Float[Array, "4"]

    body_force: Float[Array, "3"]
    actuator_force: Float[Array, "12"]
    joint_qvel: Float[Array, "12"]
    joint_angle_deltas: Float[Array, "12"]
    noisy_joint_angle_deltas: Float[Array, "12"]


def _sample_command(config: Config.Command, rng: Rng) -> tuple[Array, Rng]:
    rng, key_1, key_2, key_3, key_4 = jax.random.split(rng, 5)

    lin_vel_x = jax.random.uniform(
        key_1,
        minval=config.lin_vel_x[0],
        maxval=config.lin_vel_x[1]
    )

    lin_vel_y = jax.random.uniform(
        key_2,
        minval=config.lin_vel_y[0],
        maxval=config.lin_vel_y[1]
    )

    ang_vel_yaw = jax.random.uniform(
        key_3,
        minval=config.ang_vel_yaw[0],
        maxval=config.ang_vel_yaw[1]
    )

    command = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])

    return (
        jp.where(
            jax.random.bernoulli(key_4, 0.1),
            jp.zeros(3),
            command
        ),
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
    feet_pos: Float[Array, "12"]
    noisy_feet_pos: Float[Array, "12"]
    feet_contacts: Bool[Array, "4"]
    foot_linvel: Float[Array, "12"]


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

    feet_pos = jp.vstack([
        read_sensor(mj_model, data, name)
        for name in config.sensor.feet_pos
    ])

    rng, key_4 = jax.random.split(rng)

    noisy_feet_pos = feet_pos \
        .at[..., 0] \
        .add(
            (2 * jax.random.uniform(key_4, shape=feet_pos[..., 0].shape) - 1)
            * config.obs_noise.scale.feet_pos[0]
        )

    noisy_feet_pos = noisy_feet_pos \
        .at[..., 1] \
        .add(
            (2 * jax.random.uniform(key_4, shape=feet_pos[..., 1].shape) - 1)
            * config.obs_noise.scale.feet_pos[1]
        )

    noisy_feet_pos = noisy_feet_pos \
        .at[..., 2]  \
        .add(
            (2 * jax.random.uniform(key_4, shape=feet_pos[..., 2].shape) - 1)
            * config.obs_noise.scale.feet_pos[2]
        )

    feet_pos = feet_pos.ravel()
    noisy_feet_pos = noisy_feet_pos.ravel()

    feet_contacts = jp.array([
        read_sensor(mj_model, data, name).squeeze() > 0
        for name in config.sensor.feet_contacts
    ])

    foot_linvel = jp.array([
        read_sensor(mj_model, data, name)
        for name in config.sensor.foot_linvel
    ]).ravel()


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
            feet_pos,
            noisy_feet_pos,
            feet_contacts,
            foot_linvel
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

    def init(data: mjx.Data, rng: Rng) -> tuple[Features, Done, Rng]:
        readout, rng = _read_sensors(config, mj_model, data, rng)
        previous_command, rng = _sample_command(config.command, rng)

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

                action_history=jp.zeros(config.history_len * mjx_model.nu),
                motor_targets=jp.zeros(mjx_model.nu),
                qpos_error_history=jp.zeros(config.history_len * mjx_model.nu),

                first_contact=jp.zeros(4, dtype=jp.bool_),
                feet_air_time=jp.zeros(4),
                swing_peak=jp.zeros(4),

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
                feet_pos=readout.feet_pos,
                noisy_feet_pos=readout.noisy_feet_pos,
                feet_contacts=readout.feet_contacts,
                foot_linvel=readout.foot_linvel,

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
        motor_targets = default_pose + 0.3 * action
        motor_targets = jp.clip(
            motor_targets,
            lower_control_limits,
            upper_control_limits
        )

        readout, rng = _read_sensors(config, mj_model, data, rng)
        contact_filter = readout.feet_contacts | previous.feet_contacts
        first_contact = (previous.feet_air_time > 0.0) * contact_filter
        feet_air_time = \
            (previous.feet_air_time * ~previous.feet_contacts) + \
            config.ctrl_dt

        feet_positions = data.site_xpos[feet_site_ids]
        feet_z_positions = feet_positions[..., -1]
        swing_peak = jp.maximum(
            previous.swing_peak * ~previous.feet_contacts,
            feet_z_positions
        )

        action_history = (
            jp.roll(previous.action_history, mjx_model.nu)
                .at[:mjx_model.nu]
                .set(action)
        )

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

        body_force = data.xfrc_applied[body_id, :3]
        actuator_force = data.actuator_force
        joint_qvel = data.qvel[6:]
        joint_angle_deltas = readout.joint_angles - default_pose
        noisy_joint_angle_deltas = readout.noisy_joint_angles - default_pose

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

                action_history=action_history,
                motor_targets=motor_targets,
                qpos_error_history=qpos_error_history,

                first_contact=first_contact,
                feet_air_time=feet_air_time,
                swing_peak=swing_peak,

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
                feet_pos=readout.feet_pos,
                noisy_feet_pos=readout.noisy_feet_pos,
                feet_contacts=readout.feet_contacts,
                foot_linvel=readout.foot_linvel,

                feet_z=feet_z_positions,
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
        features.noisy_joint_angle_deltas,
        features.qpos_error_history,
        features.noisy_feet_pos,
        features.action_history,
        features.current_command
    ])

    return {
        "policy": policy,
        "value": jp.hstack([
            policy,
            features.gyro,
            features.accelerometer,
            features.gravity,
            features.local_linvel,
            features.global_angvel,
            features.joint_angle_deltas,
            features.feet_pos,
            features.joint_qvel,
            features.actuator_force,
            features.feet_contacts.astype(jp.float32),
            features.foot_linvel,
            features.feet_air_time,
            features.body_force
        ])
    }


def reward(config: Config.Reward) -> Reward[Features]:
    def call(features: Features, done: Done) -> dict[str, Array]:
        def tracking_linvel():
            error = jp.sum(
                jp.square(
                    features.previous_command[:2] - features.local_linvel[:2]
                )
            )

            return jp.exp(-error / config.tracking_sigma)

        def tracking_angvel():
            error = jp.square(features.previous_command[2] - features.gyro[2])

            return jp.exp(-error / config.tracking_sigma)

        def linvel_z():
            return jp.square(features.global_linvel[2])

        def angvel_xy():
            return jp.sum(jp.square(features.global_angvel[:2]))

        def orientation():
            return jp.sum(jp.square(features.gravity[:2]))

        def posture():
            weights = jp.array([1.0, 1.0, 1.0] * 4)
            cost = jp.sum(jp.square(features.joint_angle_deltas) * weights)
            norm = jp.linalg.norm(features.previous_command)
            weight = jp.where(norm < 0.01, -10.0, 0.0)

            return jp.exp(weight * cost)

        def termination():
            return done

        def torques():
            return (
                jp.sqrt(jp.sum(jp.square(features.actuator_force))) +
                jp.sum(jp.abs(features.actuator_force))
            )

        def action_rate():
            size = 12
            history = features.action_history

            c1 = jp.sum(jp.square(history[:size] - history[size:size * 2]))
            c2 = jp.sum(
                jp.square(
                    history[:size] -
                    2 * history[size:size * 2] +
                    history[size * 2:size * 3]
                )
            )

            return c1 + c2

        def energy():
            return jp.sum(
                jp.abs(features.joint_qvel) * jp.abs(features.actuator_force)
            )

        def feet_slip():
            foot_linvel_xy = features.foot_linvel.reshape(4, 3)[..., :2]

            return jp.sum(
                jp.sum(jp.square(foot_linvel_xy), axis=-1) * features.feet_contacts
            )

        def feet_clearance():
            foot_linvel_xy = features.foot_linvel.reshape(4, 3)[..., :2]
            norm = jp.sqrt(jp.linalg.norm(foot_linvel_xy, axis=-1))
            delta = jp.abs(features.feet_z - config.max_foot_height)

            return jp.sum(delta * norm)

        def feet_height():
            norm = jp.linalg.norm(features.previous_command)
            error = features.swing_peak / config.max_foot_height - 1.0

            cost = jp.sum(jp.square(error) * features.first_contact)
            cost *= norm >= 0.01

            return cost

        def feet_air_time():
            command = jp.linalg.norm(features.previous_command)
            reward = jp.sum(
                (features.feet_air_time - 0.1) * features.first_contact
            )

            reward *= command >= 0.01
            
            return reward

        terms = {
            "tracking_linvel": tracking_linvel(),
            "tracking_angvel": tracking_angvel(),
            "linvel_z": linvel_z(),
            "angvel_xy": angvel_xy(),
            "orientation": orientation(),
            "posture": posture(),
            "termination": termination(),
            "torques": torques(),
            "action_rate": action_rate(),
            "energy": energy(),
            "feet_slip": feet_slip(),
            "feet_clearance": feet_clearance(),
            "feet_height": feet_height(),
            "feet_air_time": feet_air_time()
        }

        rewards = {
            k: v * getattr(config.scale, k)
            for k, v in terms.items()
        }

        return rewards

    return call

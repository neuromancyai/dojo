
import jax
import mujoco
from brax import math
from brax.envs.base import State as BraxState
from brax.mjx.base import State as PipelineState
from jax import numpy as jnp
from jax.typing import ArrayLike
from mujoco import MjModel

from nrv_lab.robots.base_robot_model import BaseRobotModel
from nrv_lab.train.train_env import RewardCoefs

EPS = 1e-5


class Observation:
    def __init__(self, scale, noise):
        self.scale = scale
        self.noise = noise

    def process(self, value, key):
        noise = jax.random.normal(key, value.shape) * self.noise
        return value * self.scale + noise


class PrimusModel(BaseRobotModel):
    #scene_path = "nrv_lab/robots/primus/scene.xml"
    scene_path = "nrv_lab/robots/barkour/scene.xml"

    def __init__(self, mj_model: MjModel):
        super().__init__(mj_model)

        #self.lower_bounds = jnp.array([-0.8, -0.5, -1.5] * 4)
        self.lower_bounds = jnp.array([-1.0472, -1.54706, 0] * 4)
        #self.upper_bounds = jnp.array([0.8, 1.5, 1.0] * 4)
        self.upper_bounds = jnp.array([1.0472, 3.02902, 2.44346] * 4)
        self.default_pose = mj_model.keyframe("home").qpos
        self.jnp_default_pose = jnp.array(self.default_pose)
        self.root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "root")
        self.floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM.value, "floor")

        self.gfeet_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, f"col_lower_leg_{i}") for i in range(1, 5)]
        self.feet_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, f"lower_leg_{i}") for i in range(1, 5)]

        self.base_linear_obs = Observation(jnp.array(1.0), 0.001)
        self.base_angular_obs = Observation(jnp.array(0.25), 0.001)
        self.joint_position_obs = Observation(jnp.array(1.0), 0.0001)
        self.joint_velocity_obs = Observation(jnp.array(0.05), 0.0001)
        self.projected_gravity_obs = Observation(jnp.array(1.0), 0.001)
        self.command_obs = Observation(jnp.array([2.0, 2.0, 0.25]), 0.0)
        self.last_action_obs = Observation(jnp.array(1.0), 0.0)

        self.obs_history_size = 3
        self.action_size = 12
        self.hf_for_samples = 19
        self.hf_lat_samples = 5

    def init(self, pipeline_state: PipelineState) -> dict[str, ArrayLike]:
        dummy_key = jax.random.PRNGKey(0)
        dummy_action = jnp.zeros(self.action_size)
        self.observation_size = self.get_current_observations(pipeline_state, dummy_action, dummy_key).shape[-1]
        self.privileged_size = self.get_privileged_info(pipeline_state, self.mj_model.hfield_data).shape[-1]
        self.orientation = self.get_orientation(pipeline_state)

        robot_state = {
            "pos": self.get_position(pipeline_state).copy(),
            "rot": self.orientation.copy(),
            "qpos": pipeline_state.qpos[7:].copy(),
            "qvel": pipeline_state.qvel[6:].copy(),
            "qfrc_actuator": pipeline_state.qfrc_actuator[6:].copy(),
            "feet_pos": self._get_feet_positions(pipeline_state),
            "feet_contacts": jnp.zeros(4).astype(jnp.bool),
            "action": pipeline_state.qfrc_actuator[6:].copy()
        }

        return robot_state

    def is_done(self, pipeline_state: PipelineState) -> jax.Array:
        return jnp.array(self.get_sensor_data(pipeline_state, "upvector")[-1] < 0.0).astype(jnp.float32)

    def get_position(self, pipeline_state: PipelineState) -> jax.Array:
        return jnp.array(pipeline_state.xpos[self.root_id])

    def get_orientation(self, pipeline_state: PipelineState) -> jax.Array:
        return math.quat_to_euler(pipeline_state.xquat[self.root_id])

    def get_sensor_data(self, pipeline_state: PipelineState, sensor_name: str) -> jax.Array:
        sensor_id = self.mj_model.sensor(sensor_name).id
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        return pipeline_state.sensordata[sensor_adr: sensor_adr + sensor_dim]

    def get_current_observations(self, pipeline_state: PipelineState, prev_act: jax.Array, key: jax.Array,
                                 command: jax.Array = jnp.array([0.0, 0.0, 0.0])) -> jax.Array:
        lin_key, ang_key, jvel_key, jpos_key, command_key, act_key = jax.random.split(key, 6)

        lin = self.base_linear_obs.process(pipeline_state.qvel[:3], lin_key)
        ang = self.base_angular_obs.process(pipeline_state.qvel[3:6], ang_key)

        jvel = self.joint_velocity_obs.process(pipeline_state.qvel[6:], jvel_key)
        jpos = self.joint_position_obs.process(pipeline_state.qpos[7:], jpos_key)

        proj_grav = self.get_sensor_data(pipeline_state, "upvector")
        lin_vel = self.get_sensor_data(pipeline_state, "linvel")
        accel = self.get_sensor_data(pipeline_state, "accelerometer")

        pose = pipeline_state.qpos[7:] - self.default_pose[7:]

        command = self.command_obs.process(command, command_key)
        prev_act = self.last_action_obs.process(prev_act, act_key)

        return jnp.concatenate([lin, ang, jvel, jpos, proj_grav, lin_vel, accel, pose, command, prev_act])

    def get_observations(self, pipeline_state: PipelineState, previous_action: jax.Array, obs_history: jax.Array, key: jax.Array,
                         command: jax.Array = jnp.array([0.0, 0.0, 0.0])) -> jax.Array:
        obs = self.get_current_observations(pipeline_state, previous_action, key, command)
        return jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def get_privileged_info(self, pipeline_state: PipelineState, hf_data: jax.Array) -> jax.Array:
        return jnp.concatenate([
            self.mj_model.actuator_gainprm[0][:3],
            self.mj_model.actuator_biasprm[0][:3],
            jnp.array([self.mj_model.body_mass[self.root_id], sum(self.mj_model.body_mass[self.root_id:])]),
            self.mj_model.geom_friction[self.root_id],
            self.mj_model.geom_friction[self.feet_ids[0]],
            self._get_hf_map(pipeline_state, hf_data).flatten()
        ])

    def get_motor_action(self, action: jax.Array, key: jax.Array, noise_scale: float = 0.0) -> jax.Array:
        action_scale = 0.3
        motor_targets = self.default_pose[7:] + action * action_scale
        motor_targets = jnp.clip(motor_targets, self.lower_bounds, self.upper_bounds)
        motor_targets += noise_scale * jax.random.normal(key, motor_targets.shape)
        return motor_targets

    def get_state(self, pipeline_state: PipelineState, state: BraxState) -> dict[str, jax.Array]:
        feet_contacts = self._get_feet_contacts(pipeline_state)

        contact_filt = feet_contacts | state.info["last_contact"]
        first_contact = (state.info["feet_airtime"] > 0.0) * contact_filt

        feet_airtime = state.info["feet_airtime"] + 1.0
        feet_heights = self._get_feet_heights(pipeline_state)
        feet_swing_peak = jnp.maximum(state.info["feet_swing_peak"], feet_heights)

        return {
            "pos": self.get_position(pipeline_state).copy(),
            "rot": self.get_orientation(pipeline_state).copy(),
            "qpos": pipeline_state.qpos[7:].copy(),  # omit free-joint (3 position, 4 orientation)
            "qvel": pipeline_state.qvel[6:].copy(),  # omit free-joint (3 linear, 3 angular)
            "qfrc_actuator": pipeline_state.qfrc_actuator[6:].copy(),
            "feet_pos": self._get_feet_positions(pipeline_state),
            "feet_contacts": feet_contacts,
            "feet_airtime": feet_airtime,
            "feet_swing_peak": feet_swing_peak,
            "first_contact": first_contact,
            "done": self.is_done(pipeline_state)
        }

    def get_rewards(self,
                    pipeline_state: PipelineState,
                    robot_state: dict[str, jax.Array],
                    previous_state: dict[str, jax.Array],
                    command: jax.Array,
                    action: jax.Array,
                    reward_coefs: RewardCoefs) -> dict[str, jax.Array]:
        is_command_zero = jnp.linalg.norm(command) < 0.1
        linvel = self.get_sensor_data(pipeline_state, "linvel")
        angvel = self.get_sensor_data(pipeline_state, "gyro")

        return {
            "linear": reward_coefs.linear * self._linear_reward(command, linvel),
            "angular": reward_coefs.angular * self._angular_reward(command, angvel),
            "z_linear": reward_coefs.z_linear * self._z_linear_reward(robot_state["pos"], previous_state["pos"]),
            "xy_angular": reward_coefs.xy_angular * self._xy_angular_reward(robot_state["rot"]),

            "joint_torque": reward_coefs.joint_torque * self._joint_torque_reward(robot_state["qfrc_actuator"], previous_state["qfrc_actuator"]),
            "joint_speed": reward_coefs.joint_speed * self._joint_speed_reward(robot_state["qvel"]),
            "action_magnitude": reward_coefs.action_magnitude * self._action_magnitude_reward(previous_state["action"], action),

            "feet_slip": reward_coefs.feet_slip * self._feet_slip_reward(robot_state["feet_pos"], previous_state["feet_pos"], robot_state["feet_contacts"]),
            "feet_airtime": reward_coefs.feet_airtime * self._feet_airtime_reward(robot_state["feet_airtime"], robot_state["first_contact"], is_command_zero),
            "feet_swing_peak": reward_coefs.feet_swing_peak * self._swing_peak_reward(robot_state["feet_swing_peak"], robot_state["first_contact"], is_command_zero),
            "feet_clearance": reward_coefs.feet_clearance * self._feet_clearance_reward(robot_state["feet_pos"], previous_state["feet_pos"]),

            "stand_still": reward_coefs.stand_still * self._stand_still_reward(is_command_zero, robot_state["qpos"]),
            "pose": reward_coefs.pose * self._pose_reward(robot_state["qpos"]),
            "alive": reward_coefs.alive * ((-robot_state["done"]) + 1)
        }

    def _get_hf_map(self, pipeline_state: PipelineState, hf_data: jax.Array) -> jax.Array:
        x, y = self.get_position(pipeline_state)[:2]
        z_rot = self.get_orientation(pipeline_state)[2]

        n_rows = int(self.mj_model.hfield_nrow[0])
        n_cols = int(self.mj_model.hfield_ncol[0])
        hf_size = self.mj_model.hfield_size[0]

        step = (hf_size[0] + hf_size[1]) / n_cols
        range_for, nrange_for, range_lat = 1.0, 0.5, 0.4

        # forward and lateral positions to sample in robot's local frame
        local_forward = jnp.linspace(range_for, -nrange_for, self.hf_for_samples)
        local_lateral = jnp.linspace(-range_lat, range_lat, self.hf_lat_samples)

        # grid of sampling positions
        local_forward_grid, local_lateral_grid = jnp.meshgrid(local_forward, local_lateral, indexing='ij')

        # transform to world coordinates
        world_x = x + local_forward_grid * jnp.cos(z_rot) - local_lateral_grid * jnp.sin(z_rot)
        world_y = y + local_forward_grid * jnp.sin(z_rot) + local_lateral_grid * jnp.cos(z_rot)

        col_idx = jnp.floor((world_x + hf_size[0]) / step).astype(int)
        row_idx = jnp.floor((world_y + hf_size[1]) / step).astype(int)
        col_idx = jnp.clip(col_idx, 0, n_cols - 1)
        row_idx = jnp.clip(row_idx, 0, n_rows - 1)

        height_map = jnp.array(hf_data)[col_idx + row_idx * n_cols]

        return height_map

    def _get_feet_heights(self, pipeline_state: PipelineState) -> jax.Array:
        return jnp.array([pipeline_state.xpos[foot_id][-1] for foot_id in self.feet_ids])

    def _get_feet_contacts(self, pipeline_state: PipelineState) -> jax.Array:
        def foot_collide(fgeom):
            mask = (jnp.array([fgeom, self.floor_id]) == pipeline_state.contact.geom).all(axis=1)
            mask |= (jnp.array([self.floor_id, fgeom]) == pipeline_state.contact.geom).all(axis=1)
            idx = jnp.where(mask, pipeline_state.contact.dist, 1e-4).argmin()
            return pipeline_state.contact.dist[idx] * mask[idx] < 0

        return jnp.array([foot_collide(geom_id) for geom_id in self.gfeet_ids])

    def _get_feet_positions(self, pipeline_state: PipelineState) -> jax.Array:
        return jnp.array([pipeline_state.geom_xpos[foot_id] for foot_id in self.gfeet_ids])

    def _linear_reward(self, command: jax.Array, linvel: jax.Array) -> jax.Array:
        lin_vel_error = jnp.sum(jnp.square(command[:2] - linvel[:2]))
        return jnp.exp(-lin_vel_error / 0.25)

    def _angular_reward(self, command: jax.Array, angvel: jax.Array) -> jax.Array:
        ang_vel_error = jnp.square(command[2] - angvel[2])
        return jnp.exp(-ang_vel_error / 0.25)

    def _z_linear_reward(self, cpos: jax.Array, ppos: jax.Array) -> jax.Array:
        return -jnp.square(cpos[2] - ppos[2])

    def _xy_angular_reward(self, crot: jax.Array) -> jax.Array:
        return -jnp.square(jnp.abs(crot[0]) + jnp.abs(crot[1]))

    def _joint_torque_reward(self, cqfrc_actuator: jax.Array, pqfrc_actuator: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.square(cqfrc_actuator - pqfrc_actuator))

    def _joint_speed_reward(self, cqvel: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.square(cqvel))

    def _action_magnitude_reward(self, prev_action: jax.Array, action: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.square(prev_action - action))

    def _stand_still_reward(self, is_command_zero: jax.Array, joint_angles: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.abs(joint_angles - self.default_pose[7:])) * is_command_zero

    def _pose_reward(self, qpos: jax.Array) -> jax.Array:
        weight = jnp.array([1.0, 1.0, 0.1] * 4)
        return -jnp.sum(jnp.square(qpos - self.default_pose[7:]) * weight)

    def _feet_slip_reward(self, feet_xpos: jax.Array, pfeet_xpos: jax.Array, contacts: jax.Array) -> jax.Array:
        return -jnp.exp(jnp.sum(jnp.sum(jnp.abs(feet_xpos[..., :2] - pfeet_xpos[..., :2]), axis=-1) * contacts))

    def _feet_airtime_reward(self, airtime: jax.Array, first_contact: jax.Array, is_command_zero: jax.Array) -> jax.Array:
        return jnp.sum(jnp.exp(airtime * first_contact) * ~is_command_zero)

    def _swing_peak_reward(self, feet_swing_peak: jax.Array, first_contact: jax.Array, is_command_zero: jax.Array) -> jax.Array:
        return -jnp.exp(jnp.sum(jnp.exp(feet_swing_peak / 0.1) * first_contact) * ~is_command_zero)

    def _feet_clearance_reward(self, feet_xpos: jax.Array, pfeet_xpos: jax.Array) -> jax.Array:
        xy_vel = jnp.abs(feet_xpos[..., :2] - pfeet_xpos[..., :2])
        xy_vel_norm = jnp.sqrt(jnp.linalg.norm(xy_vel, axis=-1))
        return -jnp.sum(jnp.abs(feet_xpos[..., -1] - 1.0) * xy_vel_norm)

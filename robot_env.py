import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os


class FourWheelRobotEnv(gym.Env):
    """
    MuJoCo environment for a four-wheeled robot that needs to move forward 1.5 meters.

    Observation Space (21 dims):
      - Base position: x, y, z (3)
      - Base orientation (quaternion): w, x, y, z (4)
      - Base linear velocity: vx, vy, vz (3)
      - Base angular velocity: wx, wy, wz (3)
      - Wheel joint positions (4)
      - Wheel joint velocities (4)

    Action Space (4 dims): motor torques for four wheels in [-1, 1], scaled internally.
    """

    metadata = {"render_modes": ["rgb_array", "none"], "render_fps": 60}

    def __init__(self, model_path="robot.xml", max_episode_steps=1000, target_distance=1.5, frame_skip=5):
        super().__init__()

        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Parameters
        self.max_episode_steps = int(max_episode_steps)
        self.target_distance = float(target_distance)
        self.frame_skip = int(frame_skip)
        self.current_step = 0

        # Control scaling
        self.motor_scale = 8.0  # Nm

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        # Cache indices
        self._cache_indices()

        # Reset
        self.reset()

    def _cache_indices(self):
        # Root free joint index (to verify presence)
        self.root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
        if self.root_joint_id == -1:
            raise RuntimeError("Expected a free joint named 'root_free' on the robot base.")

        # Wheel joint ids
        self.wheel_joint_names = [
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ]
        self.wheel_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.wheel_joint_names
        ]
        if any(jid == -1 for jid in self.wheel_joint_ids):
            missing = [n for n, jid in zip(self.wheel_joint_names, self.wheel_joint_ids) if jid == -1]
            raise RuntimeError(f"Missing wheel joints in model: {missing}")

        # Motor (actuator) ids
        self.motor_names = [
            "front_left_wheel_joint_motor",
            "front_right_wheel_joint_motor",
            "rear_left_wheel_joint_motor",
            "rear_right_wheel_joint_motor",
        ]
        self.motor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.motor_names
        ]
        if any(aid == -1 for aid in self.motor_ids):
            missing = [n for n, aid in zip(self.motor_names, self.motor_ids) if aid == -1]
            raise RuntimeError(f"Missing motors in model: {missing}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Initial pose just above ground
        self.data.qpos[0:3] = [0.0, 0.0, 0.1]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Zero wheel joints (positions start after 7 free-joint dofs)
        for i in range(4):
            self.data.qpos[7 + i] = 0.0

        # Clear velocities and controls
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.initial_x = float(self.data.qpos[0])
        self.current_step = 0

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Set motor torques
        torques = action * self.motor_scale
        for i, aid in enumerate(self.motor_ids):
            self.data.ctrl[aid] = float(torques[i])

        # Step simulation with frame skip
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()

        self.current_step += 1
        terminated = self._terminated()
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "distance_traveled": float(self.data.qpos[0] - self.initial_x),
            "current_x": float(self.data.qpos[0]),
            "target_distance": self.target_distance,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        obs = np.empty(21, dtype=np.float32)
        # pos (3)
        obs[0:3] = qpos[0:3]
        # quat (4)
        obs[3:7] = qpos[3:7]
        # lin vel (3)
        obs[7:10] = qvel[0:3]
        # ang vel (3)
        obs[10:13] = qvel[3:6]
        # wheel q (4)
        obs[13:17] = qpos[7:11]
        # wheel qvel (4)
        obs[17:21] = qvel[6:10]

        return obs

    def _compute_reward(self):
        x = float(self.data.qpos[0])
        dx = x - self.initial_x

        # Progress reward scaled by target distance
        progress = dx / self.target_distance

        # Small bonus for forward velocity
        vel_forward = max(0.0, float(self.data.qvel[0]))
        vel_bonus = 0.1 * vel_forward

        # Penalties
        lateral_pen = 0.1 * abs(float(self.data.qpos[1]))
        # Simple upright proxy: penalize quat x,y components magnitude
        q = self.data.qpos[3:7]
        tilt_pen = 0.1 * (abs(float(q[1])) + abs(float(q[2])))

        # Control regularization
        u = self.data.ctrl
        ctrl_pen = 0.001 * float(np.sum(u * u))

        reward = progress + vel_bonus - lateral_pen - tilt_pen - ctrl_pen

        # Large success bonus
        if dx >= self.target_distance:
            reward += 100.0

        # Backward penalty
        if dx < 0:
            reward -= 10.0 * abs(dx)

        return float(reward)

    def _terminated(self):
        x = float(self.data.qpos[0])
        dx = x - self.initial_x

        if dx >= self.target_distance:
            return True
        if float(self.data.qpos[2]) < 0.05:
            return True
        if dx < -0.5:
            return True
        if abs(float(self.data.qpos[1])) > 2.0:
            return True
        return False

    def render(self, mode="rgb_array", width=640, height=480):
        if mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.update_scene(self.data)
            return renderer.render()
        return None

    def close(self):
        pass


if __name__ == "__main__":
    env = FourWheelRobotEnv("robot.xml")
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    for i in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"step={i} r={r:.3f} x={info['current_x']:.3f} dx={info['distance_traveled']:.3f}")
        if term or trunc:
            print("Episode end, resetting...")
            obs, info = env.reset()
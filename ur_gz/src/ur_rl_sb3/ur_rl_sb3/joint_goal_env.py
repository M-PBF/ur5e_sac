import time
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

HOME_JOINTS = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
LOWER_LIMITS = np.array([-2.9, -2.2, -2.9, -3.2, -3.2, -6.3], dtype=np.float32)
UPPER_LIMITS = np.array([2.9, 0.0, 2.9, 3.2, 3.2, 6.3], dtype=np.float32)

BASE_FRAME = "base_link"
EE_FRAMES = ("tool0", "tool0_controller", "wrist_3_link")


@dataclass
class JointSnapshot:
    position: np.ndarray
    velocity: np.ndarray


class _RosJointInterface:
    def __init__(self, command_topic: str = "/forward_position_controller/commands") -> None:
        if not rclpy.ok():
            rclpy.init()

        self.node: Node = rclpy.create_node("ur_rl_sb3_env")
        self._joint_state: JointSnapshot | None = None

        self._joint_sub = self.node.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self._cmd_pub = self.node.create_publisher(Float64MultiArray, command_topic, 10)
        self._goal_marker_pub = self.node.create_publisher(Marker, "/ur_rl_sb3/goal_marker", 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self.node)

    def _joint_state_cb(self, msg: JointState) -> None:
        idx: Dict[str, int] = {name: i for i, name in enumerate(msg.name)}
        if not all(name in idx for name in JOINT_NAMES):
            return

        positions = np.array([msg.position[idx[name]] for name in JOINT_NAMES], dtype=np.float32)
        velocities = np.zeros(6, dtype=np.float32)
        if len(msg.velocity) == len(msg.name):
            velocities = np.array([msg.velocity[idx[name]] for name in JOINT_NAMES], dtype=np.float32)

        self._joint_state = JointSnapshot(position=positions, velocity=velocities)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def wait_for_joint_state(self, timeout_sec: float = 10.0) -> JointSnapshot:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self.spin_once(timeout_sec=0.1)
            if self._joint_state is not None:
                return self._joint_state
        raise RuntimeError("Timed out waiting for /joint_states.")

    def get_joint_state(self) -> JointSnapshot:
        if self._joint_state is None:
            raise RuntimeError("No joint state received yet.")
        return self._joint_state

    def command_positions(self, joints: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = joints.astype(np.float64).tolist()
        self._cmd_pub.publish(msg)

    def get_ee_position(self) -> np.ndarray:
        for ee_frame in EE_FRAMES:
            try:
                transform = self._tf_buffer.lookup_transform(BASE_FRAME, ee_frame, rclpy.time.Time())
                translation = transform.transform.translation
                return np.array([translation.x, translation.y, translation.z], dtype=np.float32)
            except TransformException:
                continue

        raise RuntimeError(
            f"Failed to find any EE TF from {BASE_FRAME} to frames {EE_FRAMES}."
        )

    def wait_for_ee_position(self, timeout_sec: float = 10.0) -> np.ndarray:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self.spin_once(timeout_sec=0.1)
            try:
                return self.get_ee_position()
            except RuntimeError:
                pass

        raise RuntimeError("Timed out waiting for end-effector TF pose.")

    def publish_goal_marker(self, goal_point: np.ndarray) -> None:
        marker = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "ur_rl_sb3"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_point[0])
        marker.pose.position.y = float(goal_point[1])
        marker.pose.position.z = float(goal_point[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.06
        marker.scale.y = 0.06
        marker.scale.z = 0.06
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0
        self._goal_marker_pub.publish(marker)

    def close(self) -> None:
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class URJointGoalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        step_dt: float = 0.1,
        max_steps: int = 200,
        action_scale: float = 0.08,
        goal_tolerance: float = 0.10,
    ) -> None:
        super().__init__()

        self.step_dt = step_dt
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.goal_tolerance = goal_tolerance

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

        self._ros = _RosJointInterface()
        self._ros.wait_for_joint_state()
        self._home_ee = self._ros.wait_for_ee_position()

        self.goal_point = self._home_ee.copy()
        self._steps = 0

    def _sample_goal_point(self) -> np.ndarray:
        x = self.np_random.uniform(0.20, 0.70)
        y = self.np_random.uniform(-0.45, 0.45)
        z = self.np_random.uniform(0.10, 0.75)
        return np.array([x, y, z], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        state = self._ros.get_joint_state()
        ee_pos = self._ros.get_ee_position()
        error = self.goal_point - ee_pos
        return np.concatenate([state.position, state.velocity, ee_pos, error]).astype(np.float32)

    def _drive_to_home(self, settle_steps: int = 30) -> None:
        for _ in range(settle_steps):
            self._ros.command_positions(HOME_JOINTS)
            end_time = time.time() + self.step_dt
            while time.time() < end_time:
                self._ros.spin_once(timeout_sec=0.01)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0

        self._drive_to_home()
        self.goal_point = self._sample_goal_point()
        self._ros.publish_goal_marker(self.goal_point)

        obs = self._get_obs()
        info = {"goal_point": self.goal_point.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        state = self._ros.get_joint_state()
        commanded = state.position + self.action_scale * action
        commanded = np.clip(commanded, LOWER_LIMITS, UPPER_LIMITS)

        self._ros.command_positions(commanded)

        end_time = time.time() + self.step_dt
        while time.time() < end_time:
            self._ros.spin_once(timeout_sec=0.01)

        next_state = self._ros.get_joint_state()
        current_ee = self._ros.get_ee_position()
        error = self.goal_point - current_ee
        error_norm = float(np.linalg.norm(error))

        reward = -error_norm - 0.01 * float(np.linalg.norm(action))

        terminated = error_norm < self.goal_tolerance
        truncated = self._steps >= self.max_steps

        if terminated:
            reward += 5.0

        end_reason = "running"
        if terminated:
            end_reason = "goal_reached"
        elif truncated:
            end_reason = "max_steps"

        info = {
            "error_norm": error_norm,
            "goal_point": self.goal_point.copy(),
            "current_joints": next_state.position.copy(),
            "current_ee": current_ee.copy(),
            "episode_end_reason": end_reason,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        self._ros.close()


def make_env(**kwargs) -> URJointGoalEnv:
    return URJointGoalEnv(**kwargs)

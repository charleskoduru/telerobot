"""SAPIEN-backed SO-101 target for Telerobot.

This initial backend deliberately focuses on articulation bring-up and teleoperation.
It exposes the same six ``*.pos`` action/observation keys as the real SO follower,
so both the existing SO-101 leader path and the existing VR IK path can command it.
Physical RealSense cameras are not opened in simulation mode; simulated cameras are
the next layer to add once joint mapping and motion are validated.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from telerobot import PACKAGE_DIR


ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
JOINT_NAMES = (*ARM_JOINTS, GRIPPER_JOINT)


class SimulatedSO101:
    """Single SO-101 articulation controlled through Telerobot joint actions."""

    def __init__(self, cfg: Any, arm_name: str):
        self.cfg = cfg
        self.arm_name = arm_name
        self.arm_cfg = cfg.arms[arm_name]
        self.sim_cfg = cfg.teleoperation.simulation
        if self.sim_cfg is None:
            raise ValueError("Simulation configuration is required.")

        # SingleController only needs bus.motors.keys(). This tiny shim lets the
        # existing VR processor/workspace guard reuse the exact SO-101 joint names.
        self.bus = SimpleNamespace(motors={name: None for name in JOINT_NAMES})

        # Physical cameras are intentionally not opened in simulation mode.
        self.cameras: dict[str, Any] = {}

        self.scene = None
        self.articulation = None
        self.viewer = None
        self._joints: dict[str, Any] = {}
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def observation_features(self) -> dict[str, type]:
        return self.action_features

    def _urdf_path(self) -> Path:
        configured = self.sim_cfg.urdf_path
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = PACKAGE_DIR / path
            return path.resolve()

        return (
            PACKAGE_DIR
            / "simulation"
            / "SO101"
            / "so101_new_calib.urdf"
        ).resolve()

    @staticmethod
    def _limits(joint) -> tuple[float, float]:
        limits = np.asarray(joint.get_limits(), dtype=np.float32)
        return float(limits[0, 0]), float(limits[0, 1])

    def _logical_to_urdf(self, name: str, value: float) -> float:
        lower, upper = self._limits(self._joints[name])

        if name == GRIPPER_JOINT:
            # LeRobot exposes the gripper as 0..100, while the URDF uses radians.
            alpha = float(np.clip(float(value) / 100.0, 0.0, 1.0))
            return lower + alpha * (upper - lower)

        # Simulation mode currently requires use_degrees: true.
        target = math.radians(float(value))
        return float(np.clip(target, lower, upper))

    def _urdf_to_logical(self, name: str, value_rad: float) -> float:
        if name == GRIPPER_JOINT:
            lower, upper = self._limits(self._joints[name])
            if upper <= lower:
                return 0.0
            alpha = (float(value_rad) - lower) / (upper - lower)
            return float(np.clip(alpha * 100.0, 0.0, 100.0))
        return math.degrees(float(value_rad))

    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # Compatibility with the physical LeRobot robot API.
        if self._connected:
            return

        import sapien

        sapien.physx.set_scene_config(gravity=[0.0, 0.0, -9.81])
        scene = sapien.Scene()
        scene.set_timestep(1.0 / float(self.sim_cfg.physics_hz))
        scene.add_ground(altitude=-0.005)
        scene.set_ambient_light([0.45, 0.45, 0.45])
        scene.add_directional_light([1.0, -1.0, -1.0], [0.8, 0.8, 0.8])
        scene.add_point_light([0.4, -0.4, 0.7], [1.0, 1.0, 1.0])

        urdf_path = self._urdf_path()
        if not urdf_path.exists():
            raise FileNotFoundError(f"SO-101 URDF not found: {urdf_path}")

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        articulation = loader.load(str(urdf_path))
        if articulation is None:
            raise RuntimeError(f"SAPIEN failed to load SO-101 URDF: {urdf_path}")

        joints = {joint.name: joint for joint in articulation.get_active_joints()}
        missing = set(JOINT_NAMES) - set(joints)
        if missing:
            raise RuntimeError(
                "SO-101 URDF is missing expected active joints: "
                + ", ".join(sorted(missing))
            )

        for name in JOINT_NAMES:
            joint = joints[name]
            if name == GRIPPER_JOINT:
                stiffness = self.sim_cfg.gripper_stiffness
                damping = self.sim_cfg.gripper_damping
            else:
                stiffness = self.sim_cfg.arm_stiffness
                damping = self.sim_cfg.arm_damping
            joint.set_drive_property(
                stiffness=float(stiffness),
                damping=float(damping),
                force_limit=float(self.sim_cfg.force_limit),
            )

        self.scene = scene
        self.articulation = articulation
        self._joints = joints

        # Keep drive targets on the URDF's initial qpos until the first leader/VR
        # command arrives. This avoids an uncontrolled articulation at startup.
        qpos = np.asarray(articulation.get_qpos(), dtype=np.float32)
        for joint, value in zip(articulation.get_active_joints(), qpos):
            joint.set_drive_target(float(value))

        if self.sim_cfg.viewer:
            viewer = scene.create_viewer()
            viewer.set_camera_xyz(x=-0.55, y=0.0, z=0.35)
            viewer.set_camera_rpy(r=0.0, p=-0.55, y=0.0)
            self.viewer = viewer

        self._connected = True

        print(f"✅ SAPIEN SO-101 loaded: {urdf_path}")
        print("   Active joints:")
        for joint in articulation.get_active_joints():
            lower, upper = self._limits(joint)
            print(
                f"   - {joint.name:16s} "
                f"{math.degrees(lower):+7.1f}° .. {math.degrees(upper):+7.1f}°"
            )

    def set_joint_state(self, action: dict[str, float]) -> None:
        """Teleport once to a logical LeRobot pose and latch matching targets."""
        if not self._connected:
            raise RuntimeError("Simulation is not connected.")

        active = self.articulation.get_active_joints()
        qpos = np.asarray(self.articulation.get_qpos(), dtype=np.float32).copy()

        for index, joint in enumerate(active):
            key = f"{joint.name}.pos"
            if key in action:
                qpos[index] = self._logical_to_urdf(joint.name, float(action[key]))

        self.articulation.set_qpos(qpos)
        for index, joint in enumerate(active):
            joint.set_drive_target(float(qpos[index]))

        self.scene.update_render()
        if self.viewer is not None and not self.viewer.closed:
            self.viewer.render()

    def get_observation(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Simulation is not connected.")

        qpos = np.asarray(self.articulation.get_qpos(), dtype=np.float32)
        observation: dict[str, float] = {}
        for joint, value in zip(self.articulation.get_active_joints(), qpos):
            if joint.name in self._joints:
                observation[f"{joint.name}.pos"] = self._urdf_to_logical(
                    joint.name, float(value)
                )
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Simulation is not connected.")

        for name, joint in self._joints.items():
            key = f"{name}.pos"
            if key in action:
                joint.set_drive_target(
                    self._logical_to_urdf(name, float(action[key]))
                )
        return action

    def step(self) -> None:
        """Advance physics/rendering for one Telerobot control cycle."""
        if not self._connected:
            return

        if self.viewer is not None and self.viewer.closed:
            raise KeyboardInterrupt

        control_hz = max(int(self.cfg.fps), 1)
        physics_hz = max(int(self.sim_cfg.physics_hz), control_hz)
        substeps = max(1, round(physics_hz / control_hz))

        for _ in range(substeps):
            if self.sim_cfg.passive_force_compensation:
                qf = self.articulation.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                self.articulation.set_qf(qf)
            self.scene.step()

        self.scene.update_render()
        if self.viewer is not None:
            self.viewer.render()

    def disconnect(self) -> None:
        if not self._connected:
            return

        if self.viewer is not None:
            close = getattr(self.viewer, "close", None)
            if callable(close):
                close()

        self.viewer = None
        self.articulation = None
        self.scene = None
        self._joints = {}
        self._connected = False

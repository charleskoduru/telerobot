"""SAPIEN xArm6 + Robotiq 2F-85, with seven logical action channels.

The Robotiq is a parallel-opening approximation with physical loop constraints.
Its contact-driven adaptive grasp behavior has not been validated against hardware.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np

from telerobot import PACKAGE_DIR
from telerobot.simulation.so101_robot import SimulatedSO101

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
OUTER_JOINTS = ("left_outer_knuckle_joint", "right_outer_knuckle_joint")
PASSIVE_JOINTS = tuple(f"{side}_{part}_joint" for side in ("left", "right") for part in ("inner_knuckle", "inner_finger"))
LOGICAL_JOINTS = (*ARM_JOINTS, "gripper")


class SimulatedXArm6(SimulatedSO101):
    name = "xarm6_robotiq85_simulation"
    ARM_JOINTS = ARM_JOINTS
    JOINT_NAMES = (*ARM_JOINTS, *OUTER_JOINTS, *PASSIVE_JOINTS)

    def __init__(self, cfg, arm_name):
        super().__init__(cfg, arm_name)
        self.bus = SimpleNamespace(motors={name: None for name in LOGICAL_JOINTS})
        self._last_target = None
        self._loop_drives = []

    @property
    def action_features(self):
        return {f"{name}.pos": float for name in LOGICAL_JOINTS}

    def _urdf_path(self):
        if self.sim_cfg.urdf_path:
            return super()._urdf_path()
        return PACKAGE_DIR / "simulation" / "XARM6" / "xarm6_robotiq.urdf"

    def _configure_articulation(self, sapien):
        # SAPIEN URDF imports do not construct the closed four-bar loops.
        # These anchor coordinates follow ManiSkill's XArm6Robotiq agent.
        links = {link.name: link for link in self.articulation.get_links()}
        for side in ("left", "right"):
            drive = self.scene.create_drive(
                links[f"{side}_inner_knuckle"], sapien.Pose([0, 0.0376, 0.043]),
                links[f"{side}_inner_finger"], sapien.Pose([0, -0.0179011, 0.006516]),
            )
            drive.set_limit_x(0, 0)
            drive.set_limit_y(0, 0)
            drive.set_limit_z(0, 0)
            self._loop_drives.append(drive)
        for name in PASSIVE_JOINTS:
            self._joints[name].set_drive_property(stiffness=0, damping=0.01, force_limit=0)
        for name in OUTER_JOINTS:
            self._joints[name].set_drive_property(
                stiffness=float(self.sim_cfg.gripper_stiffness),
                damping=float(self.sim_cfg.gripper_damping),
                force_limit=float(self.sim_cfg.gripper_force_limit),
            )
        # Disable impossible self-collisions inside the linkage and adjacent wrist.
        names = {"link5", "link6", "robotiq_arg2f_base_link"}
        names.update(f"{side}_{part}" for side in ("left", "right") for part in (
            "inner_knuckle", "outer_knuckle", "inner_finger", "outer_finger", "inner_finger_pad"
        ))
        for name in names:
            for shape in links[name].get_collision_shapes():
                groups = list(shape.get_collision_groups())
                groups[2] |= 1 << 31
                shape.set_collision_groups(groups)
        for side in ("left", "right"):
            material = sapien.physx.PhysxMaterial(2.0, 2.0, 0.0)
            for shape in links[f"{side}_inner_finger_pad"].get_collision_shapes():
                shape.set_physical_material(material)
        # Model-specific camera framing is applied after the shared viewer opens.

    def connect(self, calibrate=True):
        super().connect(calibrate)
        if self.viewer is not None:
            self.viewer.set_camera_xyz(x=1.3, y=-1.3, z=0.9)
            self.viewer.set_camera_rpy(r=0, p=-0.35, y=2.35)

    def _logical_to_urdf(self, name, value):
        if name in ARM_JOINTS:
            lower, upper = self._limits(self._joints[name])
            return float(np.clip(math.radians(value), lower, upper))
        return float(value)  # expanded Robotiq values are already radians

    def _urdf_to_logical(self, name, value_rad):
        return math.degrees(value_rad) if name in ARM_JOINTS else float(value_rad)

    def _validate_action(self, action):
        if set(action) != set(self.action_features):
            raise ValueError(f"xarm6 needs exactly {sorted(self.action_features)}")
        if not np.isfinite(list(action.values())).all():
            raise ValueError("xarm6 targets must be finite.")

    @staticmethod
    def _expanded(action):
        expanded = {f"{name}.pos": float(action[f"{name}.pos"]) for name in ARM_JOINTS}
        angle = (1 - float(np.clip(action["gripper.pos"], 0, 100)) / 100) * 0.81
        for side in ("left", "right"):
            expanded[f"{side}_outer_knuckle_joint.pos"] = angle
            expanded[f"{side}_inner_knuckle_joint.pos"] = angle
            expanded[f"{side}_inner_finger_joint.pos"] = -angle
        return expanded

    def _clamp_action(self, action):
        targets = dict(action)
        for name in ARM_JOINTS:
            lo, hi = self._limits(self._joints[name])
            targets[f"{name}.pos"] = float(np.clip(targets[f"{name}.pos"], math.degrees(lo), math.degrees(hi)))
        targets["gripper.pos"] = float(np.clip(targets["gripper.pos"], 0, 100))
        return targets

    def set_joint_state(self, action):
        self._validate_action(action)
        targets = self._clamp_action(action)
        expanded = self._expanded(targets)
        super().set_joint_state(expanded)
        self._last_target = dict(targets)

    def get_observation(self):
        obs = super().get_observation()
        out = {f"{name}.pos": obs[f"{name}.pos"] for name in ARM_JOINTS}
        angle = sum(obs[f"{name}.pos"] for name in OUTER_JOINTS) / 2
        out["gripper.pos"] = float(np.clip(100 * (1 - angle / 0.81), 0, 100))
        out.update({name: obs[name] for name in self.cameras})
        return out
    def send_action(self, action):
        if not self.is_connected:
            raise RuntimeError("Simulation is not connected.")
        self._validate_action(action)
        targets = self._clamp_action(action)
        previous = self._last_target or self.get_observation()
        step = self.sim_cfg.max_joint_speed_deg_s / self.cfg.fps
        for name in ARM_JOINTS:
            key = f"{name}.pos"
            targets[key] = float(np.clip(targets[key], previous[key] - step, previous[key] + step))
        expanded = self._expanded(targets)
        # Drive outer knuckles only; inner joints follow the physical four-bars.
        for name in (*ARM_JOINTS, *OUTER_JOINTS):
            self._joints[name].set_drive_target(self._logical_to_urdf(name, expanded[f"{name}.pos"]))
        self._last_target = targets
        return dict(targets)

    def disconnect(self):
        try:
            super().disconnect()
        finally:
            self._loop_drives.clear()
            self._last_target = None

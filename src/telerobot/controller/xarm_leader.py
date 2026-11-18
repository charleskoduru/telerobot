"""Joint-space controller for xArm6 simulation; keeps FK workspace checks."""
from types import SimpleNamespace

from telerobot.controller.controller import SingleController
from telerobot.controller.safety import WorkspaceSafetyGuard
from telerobot.simulation.urdf_fk import URDFForwardKinematics
from telerobot.simulation.xarm6_robot import ARM_JOINTS


class XArmLeaderController(SingleController):
    def _build_processors(self):
        arm_cfg = self.cfg.arms[self.arm_name]
        self.processor = SimpleNamespace(steps=[])
        self.workspace_guard = WorkspaceSafetyGuard(
            kinematics=URDFForwardKinematics(self.robot._urdf_path(), joint_names=ARM_JOINTS),
            motor_names=list(ARM_JOINTS),
            end_effector_bounds=arm_cfg.end_effector_bounds,
            enabled=arm_cfg.hard_workspace_guard,
            margin_m=arm_cfg.workspace_guard_margin_m,
        )

    def process_vr_observation(self, vr_obs):
        raise ValueError("xArm6 simulation currently uses GELLO leader joint control.")

    def recalibrate(self):
        print("GELLO reference calibration is performed at startup; restart to recalibrate.")
        self.awaiting_recalibration = False

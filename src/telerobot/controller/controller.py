"""VR teleop controllers for single- and dual-arm follower robots."""

import copy
from abc import ABC, abstractmethod

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower

from telerobot.config import ArmConfig, RobotConfig
from telerobot.controller.vr_processor import build_vr_to_arm_processor
from telerobot.controller.kinematics import build_kinematics


def _reset_processor_state(processor) -> None:
    """Reset internal processor state: EE latch, last EE position, IK guess, etc."""
    for step in getattr(processor, "steps", []):
        reset_fn = getattr(step, "reset", None)
        if callable(reset_fn):
            reset_fn()


class Controller(ABC):
    """Base class for VR teleop controllers that manage processors and arm dispatch."""

    def __init__(self, robot: Robot, cfg: RobotConfig):
        self.robot = robot
        self.cfg = cfg
        self.has_initial_position = True
        self.awaiting_recalibration = False
        self.clutch_was_enabled = False

    def _build_processor(self, motor_names: list[str], arm_cfg: ArmConfig):
        """Create a kinematics solver and VR-to-arm processor pipeline."""
        kinematics_solver = build_kinematics(
            arm_type=arm_cfg.type,
            motor_names=motor_names,
            regularization=arm_cfg.regularization,
        )

        return build_vr_to_arm_processor(
            motor_names=motor_names,
            kinematics_solver=kinematics_solver,
            end_effector_step_sizes=arm_cfg.end_effector_step_sizes,
            end_effector_bounds=arm_cfg.end_effector_bounds,
            max_ee_step_m=arm_cfg.max_ee_step_m,
            gripper_speed_factor=arm_cfg.gripper_speed_factor,
        )

    @abstractmethod
    def _build_processors(self) -> None:
        """Create VR-to-arm processor pipelines."""

    @abstractmethod
    def capture_initial_observations(self) -> None:
        """Capture the initial arm observations used for reset."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot to its initial position and rebuild processors."""

    @abstractmethod
    def recalibrate(self) -> None:
        """Re-anchor controller pose to the current robot pose without moving the robot."""

    @abstractmethod
    def get_arm_observations(self) -> dict[str, RobotObservation]:
        """Return per-arm observations keyed by arm name."""

    @abstractmethod
    def process_vr_observation(self, vr_obs: dict) -> tuple[RobotObservation, RobotAction] | None:
        """Dispatch a VR observation to the appropriate arms."""


class SingleController(Controller):
    """Controller for a single SOFollower arm."""

    def __init__(self, robot: Robot, cfg: RobotConfig):
        super().__init__(robot, cfg)
        self.arm_name = next(iter(cfg.arms))  # "left" or "right"
        self._build_processors()

    def _build_processors(self) -> None:
        arm_cfg = self.cfg.arms[self.arm_name]
        self.processor = self._build_processor(
            list(self.robot.bus.motors.keys()),
            arm_cfg=arm_cfg,
        )

    def capture_initial_observations(self) -> None:
        self.initial_obs = self.robot.get_observation()

    def reset(self) -> None:
        print("Resetting robot to initial position...")
        self.robot.send_action(self.initial_obs)
        _reset_processor_state(self.processor)
        self.has_initial_position = True
        self.awaiting_recalibration = False

    def recalibrate(self) -> None:
        """
        Enter re-anchor mode.

        The robot does NOT move here.
        The next Grip frame captures a fresh robot/controller reference.
        """
        if self.awaiting_recalibration:
            return

        print("Recalibrating controller... now press Grip to capture new reference.")
        self.awaiting_recalibration = True
        self.has_initial_position = True
        self.clutch_was_enabled = False
        _reset_processor_state(self.processor)

    def get_arm_observations(self) -> dict[str, RobotObservation]:
        return {self.arm_name: self.robot.get_observation()}

    def process_vr_observation(self, vr_obs: dict) -> tuple[RobotObservation, RobotAction] | None:
        controller_obs = copy.deepcopy(vr_obs[self.arm_name])
        enabled = bool(controller_obs.get("enabled", False))

        # CLUTCH RELEASED:
        # Grip is not held, so freeze the robot by sending no new action.
        # Also reset the processor state so the next Grip press captures a fresh reference.
        if not enabled:
            if self.clutch_was_enabled:
                print("Clutch released: robot frozen.")
                _reset_processor_state(self.processor)

            self.clutch_was_enabled = False
            return None

        # CLUTCH ENGAGED:
        # First frame after pressing Grip. Force zero controller delta so the robot
        # latches its CURRENT FK pose as the new reference and does not jump.
        if not self.clutch_was_enabled:
            print("Clutch engaged: new reference captured.")
            _reset_processor_state(self.processor)

            controller_obs["pos"] = [0.0, 0.0, 0.0]
            controller_obs["rot"] = [0.0, 0.0, 0.0, 1.0]

            self.clutch_was_enabled = True
            self.awaiting_recalibration = False
            self.has_initial_position = True

        # A-button recalibration uses the same clutch flow:
        # after A is pressed, the next Grip frame becomes the new zero point.
        if self.awaiting_recalibration:
            print("New teleop reference captured.")
            _reset_processor_state(self.processor)

            controller_obs["pos"] = [0.0, 0.0, 0.0]
            controller_obs["rot"] = [0.0, 0.0, 0.0, 1.0]

            self.awaiting_recalibration = False
            self.has_initial_position = True

        self.has_initial_position = False

        obs = self.robot.get_observation()
        joint_action = self.processor((controller_obs, obs))
        self.robot.send_action(joint_action)
        return obs, joint_action


class BiController(Controller):
    """Controller for a BiSOFollower dual-arm robot."""

    def __init__(self, robot: Robot, cfg: RobotConfig):
        super().__init__(robot, cfg)
        self._build_processors()

    def _build_processors(self) -> None:
        self.processors = {
            "left": self._build_processor(
                list(self.robot.left_arm.bus.motors.keys()),
                arm_cfg=self.cfg.arms["left"],
            ),
            "right": self._build_processor(
                list(self.robot.right_arm.bus.motors.keys()),
                arm_cfg=self.cfg.arms["right"],
            ),
        }

    def capture_initial_observations(self) -> None:
        self.initial_left_obs = self.robot.left_arm.get_observation()
        self.initial_right_obs = self.robot.right_arm.get_observation()

    def reset(self) -> None:
        print("Resetting robot to initial position...")
        self.robot.right_arm.send_action(self.initial_right_obs)
        self.robot.left_arm.send_action(self.initial_left_obs)

        for processor in self.processors.values():
            _reset_processor_state(processor)

        self.has_initial_position = True
        self.awaiting_recalibration = False

    def recalibrate(self) -> None:
        if not self.awaiting_recalibration:
            print("Recalibrating controllers... release/re-grip or keep still.")

        self.awaiting_recalibration = True
        self.has_initial_position = True

        for processor in self.processors.values():
            _reset_processor_state(processor)

    def get_arm_observations(self) -> dict[str, RobotObservation]:
        return {
            "left": self.robot.left_arm.get_observation(),
            "right": self.robot.right_arm.get_observation(),
        }

    def process_vr_observation(self, vr_obs: dict) -> tuple[RobotObservation, RobotAction] | None:
        combined_obs: RobotObservation = {}
        combined_action: RobotAction = {}

        any_enabled = any(bool(vr_obs[side].get("enabled", False)) for side in ("right", "left"))

        if self.awaiting_recalibration:
            if not any_enabled:
                return None

            print("New dual-arm teleop reference captured.")

            for processor in self.processors.values():
                _reset_processor_state(processor)

            for side in ("right", "left"):
                vr_obs[side]["pos"] = [0.0, 0.0, 0.0]
                vr_obs[side]["rot"] = [0.0, 0.0, 0.0, 1.0]

            self.awaiting_recalibration = False
            self.has_initial_position = True

        for side in ("right", "left"):
            arm = getattr(self.robot, f"{side}_arm")
            controller_obs = copy.deepcopy(vr_obs[side])

            if controller_obs["enabled"]:
                self.has_initial_position = False

            obs = arm.get_observation()
            joint_action = self.processors[side]((controller_obs, obs))
            arm.send_action(joint_action)

            combined_obs.update({f"{side}_{k}": v for k, v in obs.items()})
            combined_action.update({f"{side}_{k}": v for k, v in joint_action.items()})

        full_obs = self.robot.get_observation()
        for key in full_obs:
            if key not in combined_obs:
                combined_obs[key] = full_obs[key]

        return combined_obs, combined_action


def build_controller(robot: Robot, cfg: RobotConfig) -> Controller:
    """Create the appropriate controller based on the robot type."""
    if isinstance(robot, SOFollower):
        return SingleController(robot, cfg)
    if isinstance(robot, BiSOFollower):
        return BiController(robot, cfg)
    raise TypeError(f"Unsupported robot type: {type(robot).__name__}")

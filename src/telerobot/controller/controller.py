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

#resets robot joint state
def _reset_processor_state(processor) -> None:
    """Reset internal processor state: EE latch, last EE position, IK guess, etc."""
    for step in getattr(processor, "steps", []):
        reset_fn = getattr(step, "reset", None)
        if callable(reset_fn):
            reset_fn()


class Controller(ABC):
    def __init__(self, robot: Robot, cfg: RobotConfig):
        self.robot = robot
        self.cfg = cfg
        self.has_initial_position = True
        self.awaiting_recalibration = False

    #This funcation defines how to intit a robot
    def _build_processor(self, motor_names: list[str], arm_cfg: ArmConfig):
        #kinematics_solver is used to calculate the joint angles needed to achieve the requested end effector pose. This does not move the robot. 
        
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

    #This funcation's purpose is to init a single or multiple arms using 
    #the _build_processor funcation.
    @abstractmethod
    def _build_processors(self) -> None:
        pass

    #Save the robot's starting points, so ex once you end the data capture via 
    # hugging face and click 'save dataset' the robot returns to the initial 
    #joint position it started. 
    @abstractmethod
    def capture_initial_observations(self) -> None:
        pass

    #Moves robot back to inital joint position and rests the controller state.
    @abstractmethod
    def reset(self) -> None:
        pass

    #sets a new transform frame between the end effector and the vr controller. 
    #This is used as pressing the A button. 
    @abstractmethod
    def recalibrate(self) -> None:
        pass

    #returns joins and sensor information of an arm. 
    @abstractmethod
    def get_arm_observations(self) -> dict[str, RobotObservation]:
        pass

    # Uses the VR processor to convert the VR controller pose into a target
    # robot joint action, sends that action to the robot, and returns both
    # the robot's current observation and the commanded joint action.
    abstractmethod
    def process_vr_observation(self, vr_obs: dict) -> tuple[RobotObservation, RobotAction] | None:
        pass


class SingleController(Controller):
    def __init__(self, robot: Robot, cfg: RobotConfig):
        super().__init__(robot, cfg)
        self.arm_name = next(iter(cfg.arms))
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
        Re-zero the VR controller to the robot's current end-effector pose.

        The robot does NOT move.
        The next enabled Grip frame captures a fresh robot/controller reference.
        """
        if not self.awaiting_recalibration:
            print("Recalibrating controller... release/re-grip or keep still.")

        self.awaiting_recalibration = True
        self.has_initial_position = True
        _reset_processor_state(self.processor)

    def get_arm_observations(self) -> dict[str, RobotObservation]:
        return {self.arm_name: self.robot.get_observation()}

    def process_vr_observation(self, vr_obs: dict) -> tuple[RobotObservation, RobotAction] | None:
        # Controller data may temporarily be None while the Quest/WebXR
        # connection is starting, refreshing, or reconnecting.
        if not isinstance(vr_obs, dict):
            return None

        raw_controller_obs = vr_obs.get(self.arm_name)

        if not isinstance(raw_controller_obs, dict):
            return None

        controller_obs = copy.deepcopy(raw_controller_obs)
        enabled = bool(controller_obs.get("enabled", False))

        if self.awaiting_recalibration:
            if not enabled:
                return None

            print("New teleop reference captured.")
            _reset_processor_state(self.processor)

            controller_obs["pos"] = [0.0, 0.0, 0.0]
            controller_obs["rot"] = [0.0, 0.0, 0.0, 1.0]

            self.awaiting_recalibration = False
            self.has_initial_position = True

        if enabled:
            self.has_initial_position = False

        obs = self.robot.get_observation()
        joint_action = self.processor((controller_obs, obs))
        self.robot.send_action(joint_action)
        return obs, joint_action


class BiController(Controller):
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
    if isinstance(robot, SOFollower):
        return SingleController(robot, cfg)

    if isinstance(robot, BiSOFollower):
        return BiController(robot, cfg)

    raise TypeError(f"Unsupported robot type: {type(robot).__name__}")

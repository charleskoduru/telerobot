from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("map_phone_action_to_robot_action")
@dataclass
class MapVRActionToRobotAction(RobotActionProcessorStep):
    """
    Raw Quest controller pose -> robot EE target.

    This is the direct/full-rotation version.
    Quest XYZ controls EE XYZ.
    Quest rotation controls EE orientation.
    """

    _enabled_prev: bool = field(default=False, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        enabled = bool(action.pop("enabled"))
        joystickX = action.pop("joystickX")
        pos = action.pop("pos")
        rot = action.pop("rot")

        if pos is None or rot is None:
            raise ValueError("pos and rot must be present in action")

        rot = Rotation.from_quat(rot)
        rotvec = rot.as_rotvec()

        gripper_vel = joystickX

        action["enabled"] = enabled

        # Position mapping
        action["target_x"] = -pos[2] if enabled else 0.0
        action["target_y"] = -pos[0] if enabled else 0.0
        action["target_z"] =  pos[1] if enabled else 0.0

        # Full Quest rotation mapping
        action["target_wx"] = -rotvec[1] if enabled else 0.0
        action["target_wy"] = -rotvec[0] if enabled else 0.0
        action["target_wz"] = -rotvec[2] if enabled else 0.0

        action["gripper_vel"] = gripper_vel

        self._enabled_prev = enabled
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "pos", "rot"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


def build_vr_to_arm_processor(
    motor_names: list[str],
    kinematics_solver: RobotKinematics,
    end_effector_step_sizes: dict[str, float],
    end_effector_bounds: dict[str, list[float]],
    max_ee_step_m: float,
    gripper_speed_factor: float,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Build pipeline to convert VR action to EE pose action to joint action."""

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapVRActionToRobotAction(),

            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=end_effector_step_sizes,
                motor_names=motor_names,
                use_latched_reference=True,
            ),

            EEBoundsAndSafety(
                end_effector_bounds=end_effector_bounds,
                max_ee_step_m=max_ee_step_m,
                raise_on_jump=False,
            ),

            GripperVelocityToJoint(
                speed_factor=gripper_speed_factor,
            ),

            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
                orientation_weight=0.003,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

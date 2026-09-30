"""Load and validate robot configuration from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot

import yaml

#Defines containers used to setup the robot via congfig.yaml

@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    index: int | str = 0
    type: str = "opencv"
    serial_number: str | None = None
    use_depth: bool = False
    align_depth: bool = True
    depth_width: int = 640
    depth_height: int = 480
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str | None = None
    vr_gamma: float = 1.0  # Less than 1 brightens the headset feed only.
    vr_gain: float = 1.0  # Multiplies pixel values before gamma correction.
    vr_brightness: int = 0  # Adds a fixed offset before gamma correction.


@dataclass
class ArmConfig:
    """Configuration for a single robot arm."""
    type: str
    port: str
    use_degrees: bool = True
    regularization: float = 1e-3
    end_effector_step_sizes: dict[str, float] = field(default_factory=lambda: {"x": 0.5, "y": 0.5, "z": 0.5})
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    )
    max_ee_step_m: float = 0.20
    gripper_speed_factor: float = 20.0
    hard_workspace_guard: bool = True
    workspace_guard_margin_m: float = 0.005
    cameras: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Configuration for recording episodes to a LeRobot dataset."""
    repo_id: str
    single_task: str
    root: str | None = None
    push_to_hub: bool = False


@dataclass
class LeaderConfig:
    """Configuration for an SO-100/SO-101 leader teleoperator."""
    type: str
    port: str
    id: str
    use_degrees: bool = True


@dataclass
class TeleoperationConfig:
    """Select the active motion-command source."""
    mode: str = "vr"
    leader: LeaderConfig | None = None


@dataclass
class RobotConfig:
    """Top-level robot configuration."""
    id: str
    fps: int
    cameras: dict[str, CameraConfig]
    arms: dict[str, ArmConfig]
    teleoperation: TeleoperationConfig = field(default_factory=TeleoperationConfig)
    dataset: DatasetConfig | None = None
    use_rerun: bool = True


def load_config(path: str | Path) -> RobotConfig:
    """Load a robot configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated RobotConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Copy config.example.yaml to config.yaml and adjust to match your setup."
        )

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Parse cameras

    cameras: dict[str, CameraConfig] = {}

    for name, cam in raw.get("cameras", {}).items():
        camera_type = str(cam.get("type", "opencv")).lower()
        if camera_type not in {"opencv", "realsense"}:
            raise ValueError(f"Camera '{name}': type must be opencv or realsense.")
        use_depth = cam.get("use_depth", camera_type == "realsense")
        align_depth = cam.get("align_depth", True)
        if not isinstance(use_depth, bool) or not isinstance(align_depth, bool):
            raise ValueError(f"Camera '{name}': use_depth and align_depth must be YAML booleans.")
        serial = cam.get("serial_number")
        if camera_type == "realsense":
            if not isinstance(serial, str) or not serial or not serial.isdigit():
                raise ValueError(f"Camera '{name}': specify serial_number in quotes, e.g. '123456789012'.")
            if any(existing.serial_number == serial for existing in cameras.values()):
                raise ValueError(f"Camera '{name}': RealSense serial_number is already in use.")
            if fourcc := cam.get("fourcc"):
                raise ValueError(f"Camera '{name}': fourcc is for OpenCV cameras only.")
        elif use_depth:
            raise ValueError(f"Camera '{name}': depth capture requires type: realsense.")
        elif "index" not in cam:
            raise ValueError(f"Camera '{name}': OpenCV requires index.")
        for setting in ("width", "height", "fps", "depth_width", "depth_height"):
            if setting in cam and (type(cam[setting]) is not int or cam[setting] <= 0):
                raise ValueError(f"Camera '{name}': {setting} must be a positive integer.")
        vr_gamma = float(cam.get("vr_gamma", 1.0))
        vr_gain = float(cam.get("vr_gain", 1.0))
        vr_brightness = int(cam.get("vr_brightness", 0))

        fourcc = cam.get("fourcc")
        if fourcc is not None:
            fourcc = str(fourcc).upper()

            if len(fourcc) != 4:
                raise ValueError(
                    f"Camera '{name}': fourcc must contain exactly four characters."
                )

        if not 0.1 <= vr_gamma <= 3.0:
            raise ValueError(
                f"Camera '{name}': vr_gamma must be between 0.1 and 3.0."
            )

        if not 0.1 <= vr_gain <= 4.0:
            raise ValueError(
                f"Camera '{name}': vr_gain must be between 0.1 and 4.0."
            )

        if not 0 <= vr_brightness <= 100:
            raise ValueError(
                f"Camera '{name}': vr_brightness must be between 0 and 100."
            )

        cameras[name] = CameraConfig(
            index=cam.get("index", 0),
            type=camera_type,
            serial_number=serial,
            use_depth=use_depth,
            align_depth=align_depth,
            depth_width=cam.get("depth_width", cam.get("width", 640)),
            depth_height=cam.get("depth_height", cam.get("height", 480)),
            width=cam.get("width", 640),
            height=cam.get("height", 480),
            fps=cam.get("fps", 30),
            fourcc=fourcc,
            vr_gamma=vr_gamma,
            vr_gain=vr_gain,
            vr_brightness=vr_brightness,
        )

    # Parse arms
    arms: dict[str, ArmConfig] = {}
    for name, arm in raw.get("arms", {}).items():
        arm_cameras = arm.get("cameras", [])
        # Validate that referenced cameras exist
        for cam_name in arm_cameras:
            if cam_name not in cameras:
                raise ValueError(
                    f"Arm '{name}' references camera '{cam_name}' which is not defined in cameras section."
                )
        arms[name] = ArmConfig(
            type=arm["type"],
            port=arm["port"],
            use_degrees=arm.get("use_degrees", True),
            regularization=arm.get("regularization", 1e-3),
            end_effector_step_sizes=arm.get("end_effector_step_sizes", {"x": 0.5, "y": 0.5, "z": 0.5}),
            end_effector_bounds=arm.get("end_effector_bounds", {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}),
            max_ee_step_m=arm.get("max_ee_step_m", 0.20),
            gripper_speed_factor=arm.get("gripper_speed_factor", 20.0),
            hard_workspace_guard=bool(arm.get("hard_workspace_guard", True)),
            workspace_guard_margin_m=float(arm.get("workspace_guard_margin_m", 0.005)),
            cameras=arm_cameras,
        )

    for camera_name, camera in cameras.items():
        owners = [arm_name for arm_name, arm in arms.items() if camera_name in arm.cameras]
        if camera.type == "realsense" and len(owners) != 1:
            raise ValueError(f"RealSense '{camera_name}' must belong to exactly one arm.")

    # Parse dataset config (optional)
    dataset_cfg: DatasetConfig | None = None
    dataset_section = raw.get("dataset")
    if dataset_section is not None:
        repo_id = dataset_section.get("repo_id")
        single_task = dataset_section.get("single_task")
        if not repo_id or not single_task:
            raise ValueError("dataset section requires both 'repo_id' and 'single_task'.")
        dataset_cfg = DatasetConfig(
            repo_id=repo_id,
            single_task=single_task,
            root=dataset_section.get("root"),
            push_to_hub=dataset_section.get("push_to_hub", False),
        )

    # Parse teleoperation mode. Missing section preserves the original VR behavior.
    teleoperation_section = raw.get("teleoperation", {}) or {}
    teleoperation_mode = str(teleoperation_section.get("mode", "vr")).lower()
    if teleoperation_mode not in {"vr", "leader"}:
        raise ValueError("teleoperation.mode must be either 'vr' or 'leader'.")

    leader_cfg: LeaderConfig | None = None
    leader_section = teleoperation_section.get("leader")
    if leader_section is not None:
        leader_type = str(leader_section.get("type", "so101_leader")).lower()
        if leader_type not in {"so100_leader", "so101_leader"}:
            raise ValueError(
                "teleoperation.leader.type must be 'so100_leader' or 'so101_leader'."
            )
        leader_port = leader_section.get("port")
        if not leader_port:
            raise ValueError("teleoperation.leader.port is required.")
        leader_cfg = LeaderConfig(
            type=leader_type,
            port=str(leader_port),
            id=str(leader_section.get("id", leader_type)),
            use_degrees=bool(leader_section.get("use_degrees", True)),
        )

    if teleoperation_mode == "leader":
        if leader_cfg is None:
            raise ValueError(
                "teleoperation.mode is 'leader', but teleoperation.leader is missing."
            )
        if len(arms) != 1:
            raise ValueError(
                "Leader mode currently supports one follower arm and one leader arm."
            )
        follower_cfg = next(iter(arms.values()))
        if follower_cfg.use_degrees != leader_cfg.use_degrees:
            raise ValueError(
                "Leader and follower use_degrees values must match to prevent unsafe commands."
            )

    robot_section = raw.get("robot", {})
    return RobotConfig(
        id=robot_section.get("id", "duo_robot"),
        fps=robot_section.get("fps", 30),
        cameras=cameras,
        arms=arms,
        teleoperation=TeleoperationConfig(
            mode=teleoperation_mode,
            leader=leader_cfg,
        ),
        dataset=dataset_cfg,
        use_rerun=robot_section.get("use_rerun", True),
    )




def build_leader_teleoperator(cfg: RobotConfig):
    """Build the configured leader arm, or return None in VR mode."""
    if cfg.teleoperation.mode != "leader":
        return None

    leader_cfg = cfg.teleoperation.leader
    if leader_cfg is None:
        raise ValueError("Leader configuration is required in leader mode.")

    from lerobot.teleoperators.so_leader import (
        SO100Leader,
        SO100LeaderConfig,
        SO101Leader,
        SO101LeaderConfig,
    )

    config_class, teleoperator_class = {
        "so100_leader": (SO100LeaderConfig, SO100Leader),
        "so101_leader": (SO101LeaderConfig, SO101Leader),
    }[leader_cfg.type]

    leader_device_config = config_class(
        port=leader_cfg.port,
        id=leader_cfg.id,
        use_degrees=leader_cfg.use_degrees,
    )
    return teleoperator_class(leader_device_config)


def load_robot(path: str | Path) -> tuple[Robot, RobotConfig]:
    """Load config and build a ready-to-use robot.

    Returns a BiSOFollower when the config defines two arms ("left" and "right"),
    or a single SOFollower when only one arm is defined.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A tuple of (robot_instance, config).
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig, Cv2Backends
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig, SOFollowerRobotConfig
    from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
    from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower

    #cfg are the values from the config.yaml file.
    cfg = load_config(path)

    # Build camera configs
    camera_configs = {
        name: OpenCVCameraConfig(
            index_or_path=cam.index,
            width=cam.width,
            height=cam.height,
            fps=cam.fps,
            fourcc=cam.fourcc,
            backend=Cv2Backends.V4L2,
        )
        for name, cam in cfg.cameras.items()
    }

    # Build arm configs
    arm_configs = {}
    for name, arm in cfg.arms.items():
        arm_cameras = {cam_name: camera_configs[cam_name] for cam_name in arm.cameras}
        arm_configs[name] = SOFollowerConfig(
            port=arm.port,
            use_degrees=arm.use_degrees,
            cameras=arm_cameras if arm_cameras else {},
        )

    if len(arm_configs) == 1:
        # Single-arm configuration
        arm_name, arm_config = next(iter(arm_configs.items()))
        single_config = SOFollowerRobotConfig(
            port=arm_config.port,
            use_degrees=arm_config.use_degrees,
            cameras=arm_config.cameras,
            id=f"{cfg.id}_{arm_name}",
        )
        robot = SOFollower(single_config)
        _install_realsense_cameras(robot, cfg)
        return robot, cfg
    else:
        # Dual-arm configuration
        duo_robot_config = BiSOFollowerConfig(
            left_arm_config=arm_configs["left"],
            right_arm_config=arm_configs["right"],
            id=cfg.id,
        )
        robot = BiSOFollower(duo_robot_config)
        _install_realsense_cameras(robot, cfg)
        return robot, cfg


def _install_realsense_cameras(robot, cfg):
    """Replace unconnected OpenCV placeholders before any robot connection.

    The placeholders keep LeRobot's normal RGB observation feature schema.
    Actual RGB and depth acquisition share our single RealSense pipeline.
    """
    from telerobot.cameras.realsense import PairedRealSenseCamera
    targets = ({next(iter(cfg.arms)): robot} if len(cfg.arms) == 1 else
               {"left": robot.left_arm, "right": robot.right_arm})
    for arm_name, target in targets.items():
        for name in cfg.arms[arm_name].cameras:
            if cfg.cameras[name].type == "realsense":
                target.cameras[name] = PairedRealSenseCamera(cfg.cameras[name])
    if len(cfg.arms) == 2:
        robot.cameras = {**robot.left_arm.cameras, **robot.right_arm.cameras}

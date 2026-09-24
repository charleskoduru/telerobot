import argparse
import json
import time
from pathlib import Path

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # noqa: F401 (imported conditionally)

from telerobot.config import build_leader_teleoperator, load_robot
from telerobot.controller import build_controller
try:
    from telerobot.dataset import (
        setup_dataset,
        end_active_episode,
        record_step,
        finalize_dataset,
        delete_episodes_from_dataset,
    )
    DATASET_AVAILABLE = True

except Exception as e:
    print(f"Dataset support disabled: {e}")

    DATASET_AVAILABLE = False

    def setup_dataset(*args, **kwargs):
        return None

    def end_active_episode(*args, **kwargs):
        pass

    def record_step(*args, **kwargs):
        pass

    def finalize_dataset(*args, **kwargs):
        pass

    def delete_episodes_from_dataset(*args, **kwargs):
        raise RuntimeError(
            "Dataset support is unavailable because the installed LeRobot "
            "version is newer than the dataset API expected by Telerobot."
        )
from telerobot.logger import get_logger, log_message, maybe_log_loop_timing
from telerobot.server import setup_webxr_server, setup_websocket_server

# Resolve default config path relative to the project root (two levels up from this file)
DEFAULT_CONFIG_PATH = str(Path(__file__).parent.parent.parent / "config.yaml")


def main():
    logger = get_logger()

    #This section enable version of telerobot to be operated: telerobot or telerobot run start the program. Use telerobot --help to see the available commands and options.
    parser = argparse.ArgumentParser(description="Telerobot — VR or leader-arm teleoperation for SO-ARM101")
    subparsers = parser.add_subparsers(dest="command")

    # --- run (default) ---
    run_parser = subparsers.add_parser("run", help="Start the configured teleoperation loop")
    run_parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )

    # --- delete-episodes ---
    del_parser = subparsers.add_parser("delete-episodes", help="Delete episodes from a dataset")
    del_parser.add_argument(
        "--repo-id", required=True,
        help="Repository ID of the dataset (e.g. 'user/dataset-name')",
    )
    del_parser.add_argument(
        "--episodes", required=True,
        help='Episode indices to delete as a JSON list, e.g. "[0, 2, 5]"',
    )
    del_parser.add_argument(
        "--root", default=None,
        help="Optional root directory override for the dataset",
    )
    del_parser.add_argument(
        "--push-to-hub", action="store_true", default=False,
        help="Push the updated dataset to the Hugging Face Hub after deletion",
    )

    args = parser.parse_args()

    # Default to "run" when no subcommand is given (backward-compatible)
    if args.command is None:
        args = run_parser.parse_args()
        args.command = "run"

    if args.command == "delete-episodes":
        episode_indices = json.loads(args.episodes)
        if not isinstance(episode_indices, list) or not all(isinstance(i, int) for i in episode_indices):
            parser.error("--episodes must be a JSON list of integers, e.g. '[0, 2, 5]'")
        delete_episodes_from_dataset(
            repo_id=args.repo_id,
            episode_indices=episode_indices,
            root=args.root,
            push_to_hub=args.push_to_hub,
            logger=logger,
        )
        return

    # --- run command ---
    #This sections of the code is responsible for loading the robot configuration, setting up the web servers for VR and teleoperation, building the controller, and initializing the dataset if available. 
    config_path = getattr(args, "config", DEFAULT_CONFIG_PATH)

    #This line add the config.yaml from config.pu using the load_robot funcation. 
    duo_robot, cfg = load_robot(config_path)
    leader_device = build_leader_teleoperator(cfg)


    camera_server = setup_webxr_server(
    duo_robot,
    logger,
    camera_image_settings={
        name: {
            "gamma": cam.vr_gamma,
            "gain": cam.vr_gain,
            "brightness": cam.vr_brightness,
        }
        for name, cam in cfg.cameras.items()
    },
    dataset_configured=(
        DATASET_AVAILABLE
        and cfg.dataset is not None
    ),
    teleoperation_mode=cfg.teleoperation.mode,
    )
    teleop_device = setup_websocket_server()

    controller = build_controller(duo_robot, cfg)

    dataset = None

    if DATASET_AVAILABLE:
        dataset = setup_dataset(
            duo_robot,
            cfg,
            logger,
        )

    # Connect to the robot
    duo_robot.connect()
    if leader_device is not None:
        leader_device.connect()

    # Init rerun viewer (optional)
    if cfg.use_rerun:
        init_rerun(session_name="vr_lerobot_teleop")

    if not duo_robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or web control server is not connected!")
    if leader_device is not None and not leader_device.is_connected:
        raise ValueError("Leader arm is not connected!")

    controller.capture_initial_observations()

    recording = False
    finalized_dataset = False
    push_to_hub = cfg.dataset.push_to_hub if cfg.dataset else False

    def publish_runtime_status():
        teleop_device.send_runtime_status(
            control_mode=cfg.teleoperation.mode,
            dataset_configured=dataset is not None,
            recording=recording,
            finalized=finalized_dataset,
            episode_count=(dataset.num_episodes if dataset is not None else 0),
        )

    publish_runtime_status()

    log_message(logger, f"🎮 Teleoperation mode: {cfg.teleoperation.mode}")
    if cfg.teleoperation.mode == "vr":
        log_message(logger, "Starting teleop loop. Connect your VR headset to teleoperate the robot...")
    else:
        log_message(logger, "Starting teleop loop. Move the leader arm to command the follower...")
    loop_count = 0
    last_action_str = "none"
    camera_read_warned = set()
    leader_action_validated = cfg.teleoperation.mode != "leader"

    # Here the code then enters a loop to handle VR observations, control the robot, stream camera frames, and manage dataset recording based on user actions.
    
    try:
        while True:
            t0 = time.perf_counter()
            t_control = t_rerun = t_dataset = None  # TODO: Remove timing debug

            # WebSocket messages always carry recording/reset actions. In VR mode
            # they also carry controller poses; leader mode ignores those poses.
            web_obs = teleop_device.last_observation
            camera_frames = {}  # Populated by process_vr_observation if available

            if web_obs is None:
                action_str = "none"
            else:
                raw_action_str = web_obs.get("action", "none")

                if raw_action_str == last_action_str and raw_action_str != "none":
                    action_str = "none"
                else:
                    action_str = raw_action_str

                last_action_str = raw_action_str

            if action_str == "recalibrate" and cfg.teleoperation.mode == "vr":
                controller.recalibrate()
                teleop_device.send_transform_status("collecting")

            if action_str == 'reset' and not controller.has_initial_position:
                controller.reset()
            elif action_str == 'start_episode' and not recording:
                # Begin a new recording episode (no-op if already recording)
                log_message(logger, f"🔴 Recording episode {dataset.num_episodes if dataset is not None else '?'}...")
                recording = True
                finalized_dataset = False
                publish_runtime_status()
            elif action_str == 'stop_episode' and recording:
                if cfg.teleoperation.mode == "vr":
                    controller.reset()
                # End the current recording episode
                end_active_episode(dataset, logger)
                recording = False
                publish_runtime_status()
            elif action_str == 'save_dataset' and not finalized_dataset:
                # Finalize and save the entire dataset
                finalize_dataset(dataset, push_to_hub, logger)
                recording = False
                finalized_dataset = True
                publish_runtime_status()
            else:
                result = None
                if cfg.teleoperation.mode == "leader":
                    # The leader produces the same named follower-joint action
                    # dictionary that the VR IK path produces downstream.
                    action = leader_device.get_action()
                    if not leader_action_validated:
                        expected_keys = set(duo_robot.action_features)
                        actual_keys = set(action)
                        if actual_keys != expected_keys:
                            raise RuntimeError(
                                "Leader/follower action features do not match. "
                                f"Expected {sorted(expected_keys)}, got {sorted(actual_keys)}."
                            )
                        leader_action_validated = True
                    obs = duo_robot.get_observation()
                    duo_robot.send_action(action)
                    controller.has_initial_position = False
                    result = (obs, action)
                elif web_obs is not None:
                    was_collecting_pose = getattr(
                        controller, "awaiting_recalibration", False
                    )
                    result = controller.process_vr_observation(web_obs)
                    if was_collecting_pose and not getattr(
                        controller, "awaiting_recalibration", False
                    ):
                        teleop_device.send_transform_status("finished")

                t_control = time.perf_counter()  # TODO: Remove timing debug

                if result is not None:
                    obs, action = result

                    # Collect camera frames from observation (avoids double camera read)
                    for cam_name in duo_robot.cameras:
                        if cam_name in obs:
                            camera_frames[cam_name] = obs[cam_name]

                    if cfg.use_rerun:
                        log_rerun_data(observation=obs, action=action)

                    t_rerun = time.perf_counter()  # TODO: Remove timing debug

                    if recording:
                        record_step(dataset, cfg, obs, action)

            # Always stream cameras — reuse obs frames when available, otherwise read directly
            for cam_name, cam in duo_robot.cameras.items():
                frame = camera_frames.get(cam_name)
                if frame is None:
                    try:
                        frame = cam.read_latest()
                    except Exception as exc:
                        if cam_name not in camera_read_warned:
                            logger.warning("Camera %s could not be read: %s", cam_name, exc)
                            camera_read_warned.add(cam_name)
                        continue

                if frame is not None:
                    camera_read_warned.discard(cam_name)
                    camera_server.update_camera_frame(cam_name, frame)
            t_camera = time.perf_counter()  # TODO: Remove timing debug

            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - t0), 0.0))

            loop_count += 1
            maybe_log_loop_timing(
                logger,
                loop_count,
                t0,
                t_control,
                t_rerun,
                t_camera,
            )
    except KeyboardInterrupt:
        log_message(logger, "\nStopping teleop...")

    finally:

        if leader_device is not None and leader_device.is_connected:
            leader_device.disconnect()

        if DATASET_AVAILABLE and not finalized_dataset:

            finalize_dataset(
                dataset,
                push_to_hub,
                logger,
            )

if __name__ == "__main__":
    main()

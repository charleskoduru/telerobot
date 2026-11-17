"""Startup/shutdown and native simulated RGB-D recording without GPU/hardware."""

import asyncio
from contextlib import ExitStack
import importlib.util
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from telerobot.config import DatasetConfig, load_config
from telerobot.dataset.dataset import setup_dataset, record_step, end_active_episode, finalize_dataset
from telerobot.server.webxr_server import WebXRServer
from telerobot.server.websocket_server import WebSocketServer
from telerobot.simulation.so101_robot import SimulatedSO101, JOINT_NAMES

PROJECT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def load_cli():
    # IK is irrelevant to lifecycle tests and requires optional physical-arm
    # packages. Keep the CLI and all server/dataset imports otherwise real.
    controller = ModuleType("telerobot.controller")
    controller.build_controller = MagicMock()
    with patch.dict(sys.modules, {"telerobot.controller": controller}):
        spec = importlib.util.spec_from_file_location("startup_test_cli", PROJECT / "src/telerobot/cli.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
    return cli


class StartupTests(unittest.TestCase):
    def run_cli(self, failure=None, mode="simulation", control="leader"):
        cli = load_cli()
        events = []
        robot = MagicMock()
        robot.cameras = {}
        robot.action_features = {"joint.pos": float}
        robot.is_connected = False

        def connect_robot():
            events.append("robot.connect")
            if failure == "viewer":
                raise RuntimeError("display failed")
            robot.is_connected = True

        robot.connect.side_effect = connect_robot
        robot.disconnect.side_effect = lambda: events.append("robot.disconnect")
        robot.set_joint_state.side_effect = lambda action: events.append("pose")
        robot.step.side_effect = KeyboardInterrupt
        leader = MagicMock()
        leader.is_connected = False
        leader.get_action.return_value = {"joint.pos": 0.0}

        def connect_leader():
            events.append("leader.connect")
            if failure == "calibration":
                raise KeyboardInterrupt
            leader.is_connected = True

        leader.connect.side_effect = connect_leader
        leader.disconnect.side_effect = lambda: events.append("leader.disconnect")
        controller = MagicMock()
        controller.has_initial_position = False
        controller.filter_joint_action.side_effect = lambda obs, action: action
        controller.process_vr_observation.return_value = None
        dataset = MagicMock()
        dataset.has_pending_frames.return_value = False
        dataset.num_episodes = 0
        dataset._telerobot_depth = None
        camera_server = MagicMock()
        camera_server.stop.side_effect = lambda: events.append("webxr.stop")
        ws = MagicMock()
        ws.is_connected = True
        ws.last_observation = None
        ws.stop.side_effect = lambda: events.append("ws.stop")
        cfg = SimpleNamespace(
            teleoperation=SimpleNamespace(mode=mode),
            dataset=SimpleNamespace(push_to_hub=False),
            cameras={}, use_rerun=False, fps=30,
        )

        def dataset_setup(*args):
            events.append("dataset.setup")
            if failure == "dataset":
                raise ValueError("schema mismatch")
            return dataset

        def webxr_setup(*args, **kwargs):
            events.append("webxr.start")
            self.assertTrue(kwargs["dataset_configured"])
            return camera_server

        def ws_setup():
            events.append("ws.start")
            if failure == "websocket":
                raise RuntimeError("port in use")
            return ws

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["telerobot"]))
            overrides = dict(
                load_robot=MagicMock(return_value=(robot, cfg)),
                build_leader_teleoperator=MagicMock(return_value=leader if control == "leader" else None),
                get_control_mode=MagicMock(return_value=control),
                build_controller=MagicMock(return_value=controller),
                setup_dataset=MagicMock(side_effect=dataset_setup),
                setup_webxr_server=MagicMock(side_effect=webxr_setup),
                setup_websocket_server=MagicMock(side_effect=ws_setup),
                finalize_dataset=MagicMock(side_effect=lambda *args: events.append("dataset.finalize")),
                precise_sleep=MagicMock(),
                DATASET_AVAILABLE=True,
            )
            stack.enter_context(patch.multiple(cli, **overrides))
            if failure in {"viewer", "websocket"}:
                with self.assertRaises(RuntimeError):
                    cli.main()
            elif failure == "dataset":
                with self.assertRaises(ValueError):
                    cli.main()
            else:
                cli.main()
        return events

    def test_calibration_precedes_viewer_and_sim_dataset_enabled(self):
        events = self.run_cli()
        self.assertEqual(events[:6], ["leader.connect", "robot.connect", "pose", "dataset.setup", "webxr.start", "ws.start"])
        for close in ["dataset.finalize", "ws.stop", "webxr.stop", "leader.disconnect", "robot.disconnect"]:
            self.assertIn(close, events)

    def test_calibration_interrupt_cleans_partial_leader_without_opening_viewer(self):
        events = self.run_cli(failure="calibration")
        self.assertNotIn("robot.connect", events)
        self.assertIn("leader.disconnect", events)

    def test_viewer_failure_closes_partial_robot_and_leader(self):
        events = self.run_cli(failure="viewer")
        self.assertNotIn("webxr.start", events)
        self.assertIn("robot.disconnect", events)
        self.assertIn("leader.disconnect", events)

    def test_dataset_failure_closes_robot_without_starting_servers(self):
        events = self.run_cli(failure="dataset")
        self.assertNotIn("webxr.start", events)
        self.assertIn("robot.disconnect", events)

    def test_websocket_failure_closes_existing_webxr_and_dataset(self):
        events = self.run_cli(failure="websocket")
        self.assertIn("webxr.stop", events)
        self.assertIn("dataset.finalize", events)
        self.assertIn("robot.disconnect", events)

    def test_vr_simulation_and_physical_leader_paths(self):
        for mode, control in [("simulation", "vr"), ("leader", "leader")]:
            events = self.run_cli(mode=mode, control=control)
            self.assertIn("dataset.setup", events)
            self.assertIn("webxr.stop", events)
            if control == "vr":
                self.assertNotIn("leader.connect", events)


class ServerLifecycleTests(unittest.TestCase):
    def test_webxr_real_http_start_stop_releases_port(self):
        import socket
        server = WebXRServer(host="127.0.0.1", port=0)
        server.add_camera("gripperCam")
        server.connect()
        port = next(iter(server._runner.sites))._server.sockets[0].getsockname()[1]
        server.stop()
        self.assertFalse(server._thread.is_alive())
        self.assertTrue(server._loop.is_closed())
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
        server.stop()  # Idempotent.

    def test_webxr_startup_failure_is_reported_and_thread_exits(self):
        server = WebXRServer(host="127.0.0.1", port=0)
        async def fail():
            raise OSError("address already in use")
        with patch.object(server, "start_server", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                server.connect()
        self.assertFalse(server._thread.is_alive())

    def test_webxr_bind_failure_cleans_runner(self):
        first = WebXRServer(host="127.0.0.1", port=0)
        first.connect()
        try:
            port = next(iter(first._runner.sites))._server.sockets[0].getsockname()[1]
            second = WebXRServer(host="127.0.0.1", port=port)
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                second.connect()
            self.assertFalse(second._thread.is_alive())
            self.assertIsNone(second._runner)
        finally:
            first.stop()

    def test_websocket_real_start_stop_and_client_close(self):
        import websockets
        original = websockets.serve
        async def local_serve(handler, *args, **kwargs):
            return await original(handler, "127.0.0.1", 0)
        server = WebSocketServer(use_ssl=False)
        with patch("telerobot.server.websocket_server.websockets.serve", side_effect=local_serve):
            server.connect()
            port = server._server.sockets[0].getsockname()[1]
            async def client():
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send('{"action":"none"}')
                    await asyncio.to_thread(server.stop)
                    await ws.wait_closed()
            asyncio.run(client())
        self.assertFalse(server.is_connected)
        self.assertFalse(server._thread.is_alive())
        server.stop()

    def test_websocket_startup_failure_is_reported_not_hung(self):
        server = WebSocketServer(use_ssl=False)
        with patch("telerobot.server.websocket_server.websockets.serve", side_effect=OSError("port occupied")):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                server.connect()
        self.assertFalse(server._thread.is_alive())


class SimulatedRecordingTests(unittest.TestCase):
    def make_robot(self, directory):
        cfg = load_config(PROJECT / "config.yaml")
        cfg.cameras = {"gripperCam": cfg.cameras["gripperCam"]}
        camcfg = cfg.cameras["gripperCam"]
        camcfg.width = camcfg.height = camcfg.depth_width = camcfg.depth_height = 32
        cfg.arms["right"].cameras = ["gripperCam"]
        cfg.dataset = DatasetConfig(repo_id="test/simulation_rgbd", single_task="pick forceps", root=str(Path(directory) / "dataset"))
        robot = SimulatedSO101(cfg, "right")
        component = MagicMock()
        color = np.full((32, 32, 4), 0.25, dtype=np.float32)
        position = np.zeros((32, 32, 4), dtype=np.float32)
        position[..., 2] = -0.125
        position[0, 0, 3] = 1  # Invalid/background.
        component.get_picture.side_effect = lambda name: color if name == "Color" else position
        camera = robot.cameras["gripperCam"]
        camera.bind(scene=MagicMock(), camera_component=component, model="d405", near_m=0.04, far_m=2, fx=30, fy=30, cx=15.5, cy=15.5, attach_to="gripper_link")
        robot._connected = True
        joints = []
        for name in JOINT_NAMES:
            joint = MagicMock()
            joint.name = name
            joint.get_limits.return_value = np.array([[-1, 1]])
            joints.append(joint)
        robot._joints = {joint.name: joint for joint in joints}
        robot.articulation = MagicMock()
        robot.articulation.get_active_joints.return_value = joints
        robot.articulation.get_qpos.return_value = np.zeros(6)
        return robot, cfg

    def test_real_simulated_camera_records_native_tiff_and_rgb(self):
        with TemporaryDirectory() as directory:
            robot, cfg = self.make_robot(directory)
            dataset = setup_dataset(robot, cfg, LOGGER)
            for _ in range(4):
                obs = robot.get_observation()
                # Preview captures another frame before recording; depth must
                # still be taken from the exact observation's retained packet.
                robot.cameras["gripperCam"].read_latest()
                record_step(dataset, cfg, obs, {key: 0.0 for key in robot.action_features})
            end_active_episode(dataset, LOGGER)
            finalize_dataset(dataset, False, LOGGER)
            loaded = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, video_backend="pyav")
            self.assertEqual(loaded.num_episodes, 1)
            self.assertEqual(loaded.num_frames, 4)
            self.assertEqual(loaded.features["observation.images.gripperCam"]["dtype"], "video")
            key = "observation.images.gripperCam_depth"
            self.assertEqual(loaded[0][key].shape, (1, 32, 32))
            self.assertEqual(loaded[0][key][0, 1, 1].item(), 125)
            self.assertEqual(loaded[0][key][0, 0, 0].item(), 0)
            self.assertEqual(loaded.reader.hf_dataset.with_format(None)[0][key].format, "TIFF")
            self.assertFalse(list(Path(cfg.dataset.root).rglob("*.npy")))
            robot.disconnect()

    def test_viewer_error_has_display_guidance_and_partial_scene_can_close(self):
        with TemporaryDirectory() as directory:
            robot, cfg = self.make_robot(directory)
            robot._connected = False
            scene = MagicMock()
            scene.create_urdf_loader.return_value.load.return_value = robot.articulation
            scene.create_viewer.side_effect = RuntimeError("Renderer does not support display")
            sapien = ModuleType("sapien")
            sapien.Scene = MagicMock(return_value=scene)
            sapien.physx = MagicMock()
            with patch.dict(sys.modules, {"sapien": sapien}), patch.object(robot, "_build_simulated_cameras"):
                with self.assertRaisesRegex(RuntimeError, "Docker DISPLAY/X11"):
                    robot.connect()
            self.assertIs(robot.scene, scene)
            robot.disconnect()
            self.assertIsNone(robot.scene)
            self.assertFalse(robot.cameras["gripperCam"].is_connected)

    def test_viewer_false_skips_window_creation(self):
        with TemporaryDirectory() as directory:
            robot, cfg = self.make_robot(directory)
            robot._connected = False
            cfg.teleoperation.simulation.viewer = False
            scene = MagicMock()
            scene.create_urdf_loader.return_value.load.return_value = robot.articulation
            sapien = ModuleType("sapien")
            sapien.Scene = MagicMock(return_value=scene)
            sapien.physx = MagicMock()
            with patch.dict(sys.modules, {"sapien": sapien}), patch.object(robot, "_build_simulated_cameras"):
                robot.connect()
            scene.create_viewer.assert_not_called()
            self.assertTrue(robot.is_connected)
            robot.disconnect()

    def test_partial_sim_connect_cleanup_and_repeated_disconnect(self):
        with TemporaryDirectory() as directory:
            robot, cfg = self.make_robot(directory)
            robot._connected = False  # Failure occurred before connect finished.
            robot.viewer = MagicMock()
            robot.viewer.close.side_effect = RuntimeError("display closed")
            with self.assertRaisesRegex(RuntimeError, "display closed"):
                robot.disconnect()
            self.assertIsNone(robot.scene)
            self.assertIsNone(robot.viewer)
            self.assertFalse(robot.cameras["gripperCam"].is_connected)
            robot.disconnect()


if __name__ == "__main__":
    unittest.main()

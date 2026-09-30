"""Hardware-free regression tests. Run: python -m unittest discover -s tests -v."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from telerobot.config import CameraConfig, load_config
from telerobot.cameras.realsense import PairedRealSenseCamera, RGBDPacket

# Depth sidecars need only NumPy, not a robot/LeRobot installation.
spec = importlib.util.spec_from_file_location("depth_tests", PROJECT / "src/telerobot/dataset/depth.py")
depth_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_module)
DepthRecorder = depth_module.DepthRecorder


def camera(serial="123456789012"):
    return PairedRealSenseCamera(CameraConfig(type="realsense", serial_number=serial, use_depth=True,
                                             width=2, height=2, depth_width=2, depth_height=2))


def packet(sequence=1):
    rgb = np.full((2, 2, 3), sequence, dtype=np.uint8)
    values = np.array([[0, 1], [1000, 65535]], dtype=np.uint16)
    return RGBDPacket(rgb, values, {"capture_sequence": sequence,
                                   "host_monotonic_ns": time.monotonic_ns(),
                                   "color_timestamp_ms": sequence * 33.3,
                                   "depth_timestamp_ms": sequence * 33.3},
                      {"depth_scale_m": 0.001, "aligned_to_color": True})


class ConfigTests(unittest.TestCase):
    def test_original_config_remains_rgb_only(self):
        config = load_config(PROJECT / "config.yaml")
        self.assertEqual(len(config.cameras), 3)
        self.assertTrue(all(cam.type == "opencv" and not cam.use_depth for cam in config.cameras.values()))

    def test_realsense_example_parses(self):
        config = load_config(PROJECT / "examples/config/realsense.yaml")
        self.assertEqual(config.cameras["bevCam"].serial_number, "222222222222")
        self.assertTrue(config.cameras["bevCam"].use_depth)
        self.assertFalse(config.cameras["baseCam"].use_depth)

    def test_invalid_serial_and_opencv_depth_are_rejected(self):
        config = yaml.safe_load((PROJECT / "examples/config/realsense.yaml").read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            config["cameras"]["gripperCam"]["serial_number"] = 111111111111
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "quotes"):
                load_config(path)
            config["cameras"]["gripperCam"]["serial_number"] = "111111111111"
            config["cameras"]["baseCam"]["use_depth"] = True
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "requires type"):
                load_config(path)


class PairingTests(unittest.TestCase):
    def test_preview_does_not_replace_observation_depth(self):
        cam = camera()
        cam._connected = True
        first = packet(1)
        cam._latest = first
        rgb = cam.async_read()
        cam._latest = packet(2)
        self.assertIs(cam.read_latest(), cam._latest.rgb)
        self.assertIs(cam.packet_for_rgb(rgb), first)
        with self.assertRaisesRegex(RuntimeError, "match depth"):
            cam.packet_for_rgb(rgb.copy())

    def test_old_capture_and_duplicate_reads_timeout(self):
        cam = camera()
        cam._connected = True
        cam._latest = packet()
        cam.async_read()
        with self.assertRaises(TimeoutError):
            cam.async_read(timeout_ms=1)
        cam._last_sequence = -1
        cam._latest.metadata["host_monotonic_ns"] -= 2_000_000_000
        with self.assertRaises(TimeoutError):
            cam.read_latest()


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "meta").mkdir()
        self.cam = camera()
        self.recorder = DepthRecorder(self.root, {"bevCam": self.cam}, 30, 0)

    def tearDown(self):
        self.recorder.close_worker()
        self.directory.cleanup()

    def add(self, episode, sequence=1):
        snapshot = packet(sequence)
        self.cam._observed.append(snapshot)
        prepared = self.recorder.prepare({"bevCam": snapshot.rgb}, episode)
        self.recorder.submit(prepared)
        return snapshot

    def test_lossless_pixels_alignment_commit_and_resume(self):
        first = self.add(0)
        self.add(0, 2)
        self.recorder.flush()
        self.assertFalse((self.root / "depth/episode_000000").exists())
        self.recorder.commit_episode(0)
        saved = np.load(self.root / "depth/episode_000000/bevCam/frame_000000.npy", allow_pickle=False)
        np.testing.assert_array_equal(saved, first.depth)
        self.assertEqual(saved.dtype, np.uint16)
        rows = [json.loads(line) for line in (self.root / "depth/episode_000000/frames.jsonl").read_text().splitlines()]
        self.assertEqual([row["frame_index"] for row in rows], [0, 1])
        self.assertEqual(rows[1]["timestamp"], 1 / 30)
        resumed = DepthRecorder(self.root, {"bevCam": self.cam}, 30, 1)
        resumed.close_worker()
        self.add(1, 3)
        self.recorder.commit_episode(1)
        self.assertTrue((self.root / "depth/episode_000001/episode.json").exists())

    def test_multiple_cameras_and_dual_arm_keys(self):
        second = camera("987654321012")
        root = self.root / "multi"
        (root / "meta").mkdir(parents=True)
        recorder = DepthRecorder(root, {"left_bevCam": self.cam, "right_gripperCam": second}, 30, 0)
        a, b = packet(1), packet(2)
        self.cam._observed.append(a)
        second._observed.append(b)
        try:
            recorder.submit(recorder.prepare({"left_bevCam": a.rgb, "right_gripperCam": b.rgb}, 0))
            recorder.commit_episode(0)
            row = json.loads((root / "depth/episode_000000/frames.jsonl").read_text())
            self.assertEqual(len(row["cameras"]), 2)
            self.assertEqual(row["cameras"]["right_gripperCam"]["capture_sequence"], 2)
        finally:
            recorder.close_worker()

    def test_incomplete_episode_is_not_overwritten(self):
        self.add(0)
        self.recorder.flush()
        with self.assertRaisesRegex(RuntimeError, "Unfinished"):
            DepthRecorder(self.root, {"bevCam": self.cam}, 30, 0)

    def test_wrong_rgb_and_disk_failure_are_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "match depth"):
            self.recorder.prepare({"bevCam": packet().rgb}, 0)
        with patch.object(depth_module.np, "save", side_effect=OSError("disk full")):
            self.add(0)
            with self.assertRaisesRegex(RuntimeError, "Depth writing failed"):
                self.recorder.flush()
        self.assertTrue((self.root / "depth/episode_000000.incomplete").exists())
        self.assertFalse((self.root / "depth/episode_000000").exists())

    def test_overload_fails_before_accepting_rgb(self):
        with patch.object(self.recorder._queue, "full", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "cannot keep up"):
                self.recorder.prepare({}, 0)

    def test_deletion_reindexes_depth(self):
        for episode in range(3):
            self.add(episode, episode + 1)
            self.recorder.commit_episode(episode)
        destination = self.root / "rewritten"
        (destination / "meta").mkdir(parents=True)
        depth_module.copy_depth_after_deletion(self.root, destination, [1], 3)
        path = destination / "depth/episode_000001"
        self.assertEqual(json.loads((path / "episode.json").read_text())["episode_index"], 1)
        row = json.loads((path / "frames.jsonl").read_text())
        self.assertEqual(row["episode_index"], 1)
        self.assertEqual(row["cameras"]["bevCam"]["capture_sequence"], 3)
        self.assertFalse((destination / "depth/episode_000002").exists())


class DatasetIntegrationTests(unittest.TestCase):
    """Exercise Telerobot dataset lifecycle with a fake external LeRobot writer."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cam = camera()
        self.cfg = SimpleNamespace(
            fps=30, arms={"right": object()},
            dataset=SimpleNamespace(root=str(self.root), repo_id="test/rgbd", single_task="pick"))
        self.robot = SimpleNamespace(cameras={"bevCam": self.cam}, action_features={},
                                     observation_features={}, name="so_follower")
        self.logger = MagicMock()
        self.storage = {"episodes": 0, "frames": 0}
        storage = self.storage
        root = self.root

        class FakeDataset:
            def __init__(inner, **kwargs):
                inner.root = root
                inner.repo_id = "test/rgbd"
                inner.features = {}
                inner.frames = []
                inner.finalized = False

            @classmethod
            def create(cls, **kwargs):
                (root / "meta").mkdir(exist_ok=True)
                (root / "meta/info.json").write_text('{"total_frames": 0}')
                return cls(**kwargs)

            @classmethod
            def resume(cls, **kwargs):
                return cls(**kwargs)

            @property
            def num_episodes(inner):
                return storage["episodes"]

            @property
            def num_frames(inner):
                return storage["frames"]

            def has_pending_frames(inner):
                return bool(inner.frames)

            def add_frame(inner, frame):
                inner.frames.append(frame)

            def save_episode(inner):
                storage["episodes"] += 1
                storage["frames"] += len(inner.frames)
                inner.frames.clear()

            def finalize(inner):
                inner.finalized = True

            def push_to_hub(inner, **kwargs):
                pass

        def module(name, **attrs):
            result = ModuleType(name)
            result.__dict__.update(attrs)
            return result

        modules = {
            "telerobot.dataset.depth": depth_module,
            "lerobot.datasets.dataset_tools": module("tools", delete_episodes=MagicMock()),
            "lerobot.datasets.lerobot_dataset": module("dataset", LeRobotDataset=FakeDataset),
            "lerobot.utils.constants": module("constants", HF_LEROBOT_HOME=root, ACTION="action", OBS_STR="observation"),
            "lerobot.datasets.pipeline_features": module("features", aggregate_pipeline_dataset_features=lambda **kwargs: {}, create_initial_features=lambda **kwargs: {}),
            "lerobot.utils.feature_utils": module("utils", build_dataset_frame=lambda features, values, prefix: {prefix: values}, combine_feature_dicts=lambda *values: {}),
            "lerobot.processor": module("processor", make_default_processors=lambda: (None, None, None)),
        }
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("dataset_integration", PROJECT / "src/telerobot/dataset/dataset.py")
            self.api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.api)
        self.dataset = self.api.setup_dataset(self.robot, self.cfg, self.logger)

    def tearDown(self):
        self.dataset._telerobot_depth.close_worker()
        self.directory.cleanup()

    def add(self, sequence=1):
        value = packet(sequence)
        self.cam._observed.append(value)
        self.api.record_step(self.dataset, self.cfg, {"bevCam": value.rgb}, {"joint": 1.0})

    def test_record_save_checkpoint_continue(self):
        self.add()
        self.add(2)
        self.api.end_active_episode(self.dataset, self.logger)
        self.assertEqual(self.dataset.num_episodes, 1)
        self.assertEqual(json.loads((self.root / "depth/episode_000000/episode.json").read_text())["num_frames"], 2)
        previous = self.dataset
        self.dataset, success = self.api.checkpoint_dataset(self.dataset, self.cfg, False, self.logger)
        self.assertTrue(success)
        self.assertTrue(previous.finalized)
        self.assertIs(self.dataset._telerobot_depth, previous._telerobot_depth)
        self.add(3)
        self.api.end_active_episode(self.dataset, self.logger)
        self.assertEqual(self.dataset.num_episodes, 2)
        self.assertTrue((self.root / "depth/episode_000001/episode.json").exists())

    def test_rgb_save_failure_keeps_depth_incomplete(self):
        self.add()
        with patch.object(self.dataset, "save_episode", side_effect=OSError("RGB save failed")):
            with self.assertRaises(OSError):
                self.api.end_active_episode(self.dataset, self.logger)
        self.assertTrue((self.root / "depth/episode_000000.incomplete").exists())
        self.assertFalse((self.root / "depth/episode_000000").exists())
        with self.assertRaises(RuntimeError):
            self.api.finalize_dataset(self.dataset, False, self.logger)

    def test_hub_sync_removes_stale_depth_before_tagging(self):
        self.add()
        self.api.end_active_episode(self.dataset, self.logger)
        events = []
        hub = MagicMock()
        hub.upload_folder.side_effect = lambda **kwargs: events.append(("depth", kwargs))
        module = ModuleType("huggingface_hub")
        module.HfApi = lambda: hub
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            with patch.object(self.dataset, "push_to_hub", side_effect=lambda **kwargs: events.append(("rgb", kwargs))):
                self.api._push_with_depth(self.dataset)
        self.assertEqual([name for name, _ in events], ["depth", "rgb"])
        self.assertEqual(events[0][1]["delete_patterns"], ["depth/**"])
        self.assertIn("meta/depth.json", events[0][1]["allow_patterns"])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the native TIFF depth converter (no camera required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]

# The pixel conversion helper itself does not need Hugging Face Datasets or
# LeRobot. Supply small import stubs so this test also runs in the lightweight
# Telerobot development environment.
if "datasets" not in sys.modules:
    sys.modules["datasets"] = ModuleType("datasets")

if "lerobot" not in sys.modules:
    lerobot = ModuleType("lerobot")
    lerobot_datasets = ModuleType("lerobot.datasets")
    lerobot_dataset = ModuleType("lerobot.datasets.lerobot_dataset")

    class LeRobotDataset:  # pragma: no cover - import-only test stub
        pass

    lerobot_dataset.LeRobotDataset = LeRobotDataset
    sys.modules["lerobot"] = lerobot
    sys.modules["lerobot.datasets"] = lerobot_datasets
    sys.modules["lerobot.datasets.lerobot_dataset"] = lerobot_dataset

sys.path.insert(0, str(PROJECT / "tools"))
spec = importlib.util.spec_from_file_location(
    "convert_depth_to_tiff",
    PROJECT / "tools" / "convert_depth_to_tiff.py",
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class NativeDepthConversionTests(unittest.TestCase):
    def test_d405_sensor_units_become_uint16_millimetres(self):
        raw = np.array([[0, 1, 1251, 5000]], dtype=np.uint16)
        actual = converter.depth_sensor_units_to_mm(raw, depth_scale_m=0.0001)

        self.assertEqual(actual.dtype, np.uint16)
        self.assertEqual(actual.shape, (1, 4, 1))
        np.testing.assert_array_equal(
            actual[..., 0],
            np.array([[0, 0, 125, 500]], dtype=np.uint16),
        )

    def test_standard_one_mm_scale_is_preserved(self):
        raw = np.array([[0, 1, 1000, 65535]], dtype=np.uint16)
        actual = converter.depth_sensor_units_to_mm(raw, depth_scale_m=0.001)
        np.testing.assert_array_equal(actual[..., 0], raw)

    def test_invalid_shape_and_overflow_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "2-D uint16"):
            converter.depth_sensor_units_to_mm(np.zeros((2, 2, 1), np.uint16), 0.001)
        with self.assertRaises(OverflowError):
            converter.depth_sensor_units_to_mm(
                np.array([[65535]], dtype=np.uint16),
                depth_scale_m=0.01,
            )

    def test_uint16_tiff_round_trip_is_exact(self):
        frame = np.array([[0, 125, 500, 4000]], dtype=np.uint16)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "frame-000000.tiff"
            Image.fromarray(frame).save(path, compression="raw")
            decoded = np.asarray(Image.open(path))
        self.assertEqual(decoded.dtype, np.uint16)
        np.testing.assert_array_equal(decoded, frame)


if __name__ == "__main__":
    unittest.main()

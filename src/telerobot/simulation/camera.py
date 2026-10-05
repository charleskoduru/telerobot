"""SAPIEN camera adapters that mimic Telerobot's RGB/RGB-D camera surface."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


# RealSense D400-series color-camera nominal fields of view. These values are
# only optical starting points for simulation; the physical camera's calibrated
# intrinsics are still the source of truth for final sim-to-real matching.
CAMERA_MODEL_PROFILES: dict[str, dict[str, float]] = {
    "d405": {
        "color_hfov_deg": 84.0,
        "color_vfov_deg": 58.0,
        "default_near_m": 0.04,
        "default_far_m": 2.0,
    },
    "d415": {
        "color_hfov_deg": 69.0,
        "color_vfov_deg": 42.0,
        "default_near_m": 0.03,
        "default_far_m": 3.0,
    },
    "d435i": {
        "color_hfov_deg": 69.0,
        "color_vfov_deg": 42.0,
        "default_near_m": 0.03,
        "default_far_m": 3.0,
    },
}


@dataclass(frozen=True)
class SimRGBDPacket:
    """Synchronized simulated RGB/depth sample.

    Depth is stored as uint16 millimeters to mirror the RealSense Z16 packets
    used by the physical Telerobot recording path. A value of 0 is invalid.
    """

    rgb: np.ndarray
    depth: np.ndarray | None
    metadata: dict[str, Any]
    calibration: dict[str, Any]


def look_at_pose(sapien_module, eye, target, up):
    """Build a SAPIEN camera pose (+X forward, +Y left, +Z up)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("Camera position and look_at target must be different.")
    forward /= norm

    up_norm = np.linalg.norm(up)
    if up_norm < 1e-9:
        raise ValueError("Camera up vector must be non-zero.")
    up /= up_norm

    left = np.cross(up, forward)
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-6:
        # Pick a deterministic fallback when up is parallel to the view axis.
        fallback = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(fallback, forward))) > 0.95:
            fallback = np.array([1.0, 0.0, 0.0])
        left = np.cross(fallback, forward)
        left_norm = np.linalg.norm(left)
    left /= left_norm
    camera_up = np.cross(forward, left)
    camera_up /= np.linalg.norm(camera_up)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack([forward, left, camera_up], axis=1)
    transform[:3, 3] = eye
    return sapien_module.Pose(transform)


def intrinsics_from_fov(width: int, height: int, hfov_deg: float, vfov_deg: float):
    """Return pinhole intrinsics matching independent horizontal/vertical FOVs."""
    hfov = np.deg2rad(float(hfov_deg))
    vfov = np.deg2rad(float(vfov_deg))
    fx = float(width) / (2.0 * np.tan(hfov / 2.0))
    fy = float(height) / (2.0 * np.tan(vfov / 2.0))
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return fx, fy, cx, cy


class SimulatedCamera:
    """LeRobot-style RGB camera wrapper around a SAPIEN RenderCameraComponent."""

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.width = int(config.width)
        self.height = int(config.height)
        self.fps = int(config.fps)

        self._scene = None
        self._camera = None
        self._connected = False
        self._latest_packet: SimRGBDPacket | None = None
        self._observed = deque(maxlen=32)
        self._sequence = 0
        self._calibration: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def bind(
        self,
        *,
        scene,
        camera_component,
        model: str,
        near_m: float,
        far_m: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        attach_to: str,
    ) -> None:
        self._scene = scene
        self._camera = camera_component
        self._connected = True
        self._latest_packet = None
        self._observed.clear()
        self._sequence = 0
        self._calibration = {
            "device_name": f"Simulated Intel RealSense {model.upper()}",
            "model": model,
            "simulated": True,
            "depth_scale_m": 0.001 if self.config.use_depth else None,
            "aligned_to_color": bool(self.config.use_depth and self.config.align_depth),
            "rgb_format": "RGB8",
            "depth_format": "Z16" if self.config.use_depth else None,
            "invalid_depth_value": 0,
            "attach_to": attach_to,
            "near_m": float(near_m),
            "far_m": float(far_m),
            "color_intrinsics": {
                "width": self.width,
                "height": self.height,
                "fx": float(fx),
                "fy": float(fy),
                "ppx": float(cx),
                "ppy": float(cy),
                "distortion_model": "pinhole",
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        }
        if self.config.use_depth:
            # The first simulation pass renders depth directly from the same
            # optical camera, so it is intrinsically registered to RGB.
            self._calibration["stored_depth_intrinsics"] = dict(
                self._calibration["color_intrinsics"]
            )

    def connect(self, *args, **kwargs) -> None:
        del args, kwargs
        if not self._connected:
            raise RuntimeError(
                f"Simulated camera '{self.name}' is created when the SAPIEN robot connects."
            )

    def _capture(self) -> SimRGBDPacket:
        if not self._connected or self._scene is None or self._camera is None:
            raise RuntimeError(f"Simulated camera '{self.name}' is not connected.")

        self._scene.update_render()
        self._camera.take_picture()

        color = np.asarray(self._camera.get_picture("Color"))
        rgb = color[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
        rgb = np.ascontiguousarray(rgb)

        depth_z16 = None
        if self.config.use_depth:
            position = np.asarray(self._camera.get_picture("Position"))
            depth_m = -position[..., 2]
            valid = (
                np.isfinite(depth_m)
                & (position[..., 3] < 1.0)
                & (depth_m >= float(self._calibration["near_m"]))
                & (depth_m <= float(self._calibration["far_m"]))
            )
            depth_z16 = np.zeros(depth_m.shape, dtype=np.uint16)
            depth_mm = np.rint(depth_m[valid] * 1000.0)
            depth_z16[valid] = np.clip(depth_mm, 1, 65535).astype(np.uint16)
            depth_z16 = np.ascontiguousarray(depth_z16)

        rgb.setflags(write=False)
        if depth_z16 is not None:
            depth_z16.setflags(write=False)

        self._sequence += 1
        packet = SimRGBDPacket(
            rgb=rgb,
            depth=depth_z16,
            metadata={
                "capture_sequence": self._sequence,
                "host_monotonic_ns": time.monotonic_ns(),
                "host_time_ns": time.time_ns(),
                "simulated": True,
            },
            calibration=dict(self._calibration),
        )
        self._latest_packet = packet
        self._observed.append(packet)
        return packet

    def async_read(self, timeout_ms=200):
        del timeout_ms
        return self._capture().rgb

    def read(self, *args, **kwargs):
        del args, kwargs
        return self._capture().rgb

    def read_latest(self, *args, **kwargs):
        del args, kwargs
        return self._capture().rgb

    def read_depth_latest(self) -> np.ndarray | None:
        """Capture and return aligned Z16 depth for RGB-D cameras."""
        return self._capture().depth

    def packet_for_rgb(self, rgb: np.ndarray) -> SimRGBDPacket:
        """Return the synchronized depth packet corresponding to an RGB ndarray."""
        for packet in reversed(self._observed):
            if packet.rgb is rgb:
                return packet
        raise RuntimeError(
            f"Cannot match simulated depth to RGB observation for camera '{self.name}'."
        )

    @property
    def calibration(self) -> dict[str, Any]:
        return dict(self._calibration)

    def disconnect(self) -> None:
        self._scene = None
        self._camera = None
        self._connected = False
        self._latest_packet = None
        self._observed.clear()

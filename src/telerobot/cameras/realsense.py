"""One RealSense pipeline per device, with paired RGB and Z16 depth snapshots."""
from collections import deque
from dataclasses import dataclass
import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RGBDPacket:
    rgb: np.ndarray
    depth: np.ndarray | None
    metadata: dict
    calibration: dict


def _intrinsics(profile):
    intr = profile.as_video_stream_profile().get_intrinsics()
    return {"width": intr.width, "height": intr.height, "fx": intr.fx,
            "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "distortion_model": str(intr.model), "coeffs": list(intr.coeffs)}


class PairedRealSenseCamera:
    """LeRobot-compatible RGB camera; recording retrieves the exact matching packet.

    Only the capture thread reads the pipeline. Every RGB read retains its
    matching depth packet, including LeRobot's read_latest observation path.
    Depth remains uint16.
    """
    def __init__(self, config):
        self.config = config
        self.width, self.height, self.fps = config.width, config.height, config.fps
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._observed = deque(maxlen=32)
        self._thread = None
        self._pipeline = None
        self._error = None
        self._connected = False
        self._last_sequence = -1
        self._sequence = 0

    @property
    def is_connected(self):
        return self._connected

    def connect(self, warmup=True):
        if self.is_connected:
            raise RuntimeError("RealSense camera is already connected.")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("Install RealSense support: python -m pip install pyrealsense2") from exc
        self._rs = rs
        self._stop.clear()
        self._error = None
        self._latest = None
        self._observed.clear()
        self._last_sequence = -1
        self._sequence = 0
        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(self.config.serial_number)
        stream_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.config.use_depth:
            stream_config.enable_stream(rs.stream.depth, self.config.depth_width,
                                        self.config.depth_height, rs.format.z16, self.fps)
        try:
            profile = pipeline.start(stream_config)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot start RealSense {self.config.serial_number}. Check USB access, "
                "close other camera apps, and check supported RGB/depth resolutions and FPS."
            ) from exc
        self._pipeline = pipeline
        self._profile = profile
        try:
            self._align = rs.align(rs.stream.color) if self.config.use_depth and self.config.align_depth else None
            self._scale = (float(profile.get_device().first_depth_sensor().get_depth_scale())
                           if self.config.use_depth else None)
            color_profile = profile.get_stream(rs.stream.color)
            self._calibration = {
                "serial_number": self.config.serial_number,
                "device_name": profile.get_device().get_info(rs.camera_info.name),
                "depth_scale_m": self._scale,
                "aligned_to_color": bool(self._align),
                "color_intrinsics": _intrinsics(color_profile),
                "rgb_format": "RGB8", "depth_format": "Z16", "invalid_depth_value": 0,
            }
            if self.config.use_depth:
                depth_profile = profile.get_stream(rs.stream.depth)
                extr = depth_profile.get_extrinsics_to(color_profile)
                self._calibration.update({
                    "native_depth_intrinsics": _intrinsics(depth_profile),
                    "depth_to_color_extrinsics": {"rotation": list(extr.rotation),
                                                  "translation_m": list(extr.translation)},
                })
            self._connected = True
            self._thread = threading.Thread(target=self._capture_loop, name="realsense-capture", daemon=True)
            self._thread.start()
            # Always validate an initial packet before handing the camera to the robot.
            self._wait_packet(5000)
        except Exception:
            self.disconnect()
            raise

    def _capture_loop(self):
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if self._align is not None:
                    frames = self._align.process(frames)
                color = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self.config.use_depth else None
                if not color or (self.config.use_depth and not depth_frame):
                    continue
                rgb = np.asanyarray(color.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None
                # Keep snapshots immutable while background disk writes are in flight.
                rgb.setflags(write=False)
                if depth is not None:
                    depth.setflags(write=False)
                calibration = dict(self._calibration)
                if depth_frame:
                    calibration["stored_depth_intrinsics"] = _intrinsics(depth_frame.profile)
                self._sequence += 1
                metadata = {
                    "capture_sequence": self._sequence,
                    "host_monotonic_ns": time.monotonic_ns(),
                    "host_time_ns": time.time_ns(),
                    "color_frame_number": color.get_frame_number(),
                    "color_timestamp_ms": color.get_timestamp(),
                    "color_timestamp_domain": str(color.get_frame_timestamp_domain()),
                }
                if depth_frame:
                    metadata.update({"depth_frame_number": depth_frame.get_frame_number(),
                                     "depth_timestamp_ms": depth_frame.get_timestamp(),
                                     "depth_timestamp_domain": str(depth_frame.get_frame_timestamp_domain())})
                packet = RGBDPacket(rgb, depth, metadata, calibration)
                with self._condition:
                    self._latest = packet
                    self._error = None
                    self._condition.notify_all()
            except Exception as exc:
                if self._stop.is_set():
                    break
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                # Prevent a busy loop after a USB disconnect; each new read can retry.
                self._stop.wait(0.05)

    def _wait_packet(self, timeout_ms, new=False):
        deadline = time.monotonic() + timeout_ms / 1000
        with self._condition:
            while True:
                if not self.is_connected:
                    raise RuntimeError("RealSense camera is disconnected.")
                packet = self._latest
                if (packet is not None and self._error is None
                        and time.monotonic_ns() - packet.metadata["host_monotonic_ns"] < 1_000_000_000
                        and (not new or packet.metadata["capture_sequence"] != self._last_sequence)):
                    return packet
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RealSense {self.config.serial_number}: no fresh RGB/depth frames.") from self._error
                self._condition.wait(remaining)

    def async_read(self, timeout_ms=200):
        packet = self._wait_packet(timeout_ms, new=True)
        self._last_sequence = packet.metadata["capture_sequence"]
        self._remember_packet(packet)
        return packet.rgb

    def read(self, *args, **kwargs):
        return self.async_read(timeout_ms=kwargs.get("timeout_ms", 1000))

    def read_latest(self, *args, **kwargs):
        # This project's preview expects the image alone, not (image, timestamp).
        packet = self._wait_packet(200)
        self._remember_packet(packet)
        return packet.rgb

    def _remember_packet(self, packet):
        with self._condition:
            # Repeated preview reads of one capture must not consume the history.
            if not self._observed or self._observed[-1] is not packet:
                self._observed.append(packet)

    def packet_for_rgb(self, rgb):
        with self._condition:
            for packet in reversed(self._observed):
                if packet.rgb is rgb:
                    return packet
        raise RuntimeError("Cannot match depth to the RGB observation; refusing an unsynchronized recording.")

    def disconnect(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._thread = None
        self._connected = False
        with self._condition:
            self._condition.notify_all()

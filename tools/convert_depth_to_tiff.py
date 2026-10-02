#!/usr/bin/env python3
"""Build a LeRobot v3 RGB-D dataset with lossless uint16 TIFF depth images.

The source dataset is never modified. RGB observations, robot state, and
actions are copied into a new dataset. Telerobot ``.npy`` depth sidecars are
converted from RealSense sensor units into integer millimetres and written as
one-channel TIFF images under ``observation.images.<camera>_depth``.

Requires LeRobot >= 0.6.0. The target depth feature is:

* dtype: ``image``
* shape: ``(height, width, 1)``
* info.is_depth_map: ``true``
* info.depth_unit: ``mm``

RGB streams remain video features. Depth stays as lossless TIFF rather than
being quantized into a depth video or converted to an RGB-looking image.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from packaging.version import Version

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from convert_depth_for_act import (
    AUTOMATIC_FEATURES,
    as_numpy,
    depth_episode_info,
    sanitize_source_in_memory,
    scalar,
    task_lookup,
    visual_to_hwc_uint8,
)


MIN_LEROBOT_VERSION = Version("0.6.0")
DEFAULT_SOURCE_REPO = "charlieK123/depth_dataSet"
DEFAULT_TARGET_REPO = "charlieK123/depth_dataSet_native_tiff_rgbd"


def require_supported_lerobot() -> str:
    """Fail early when TIFF/depth-aware LeRobot APIs are unavailable."""
    try:
        installed = version("lerobot")
    except PackageNotFoundError as exc:
        raise RuntimeError("LeRobot is not installed in this Python environment.") from exc
    if Version(installed) < MIN_LEROBOT_VERSION:
        raise RuntimeError(
            f"LeRobot >= {MIN_LEROBOT_VERSION} is required for native TIFF depth; "
            f"found {installed}."
        )
    return installed


def depth_sensor_units_to_mm(raw_depth: np.ndarray, depth_scale_m: float) -> np.ndarray:
    """Convert one RealSense depth frame to ``uint16`` millimetres, HWC.

    ``raw_depth`` is stored in camera-specific sensor units. For example, a
    D405 scale of 0.0001 m/unit means raw value 1251 represents 125.1 mm and is
    rounded to 125 mm in the requested uint16-millimetre representation.
    Invalid zero pixels remain zero. Values that cannot be represented exactly
    in uint16 millimetres cause an error instead of being silently clipped.
    """
    raw_depth = np.asarray(raw_depth)
    if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise ValueError(
            f"Expected a 2-D uint16 depth frame, got {raw_depth.dtype} {raw_depth.shape}"
        )
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError(f"Invalid RealSense depth scale: {depth_scale_m}")

    depth_mm_float = raw_depth.astype(np.float64) * float(depth_scale_m) * 1000.0
    valid = raw_depth != 0
    if valid.any():
        maximum = float(depth_mm_float[valid].max())
        if maximum > np.iinfo(np.uint16).max + 0.5:
            raise OverflowError(
                f"Depth value {maximum:.3f} mm exceeds uint16 capacity; "
                "use float32 metres instead of silently clipping."
            )

    depth_mm = np.rint(depth_mm_float).astype(np.uint16)
    depth_mm[~valid] = 0
    return np.ascontiguousarray(depth_mm[..., None])


def native_tiff_features(
    source: LeRobotDataset,
    rgb_key: str,
    depth_key: str,
) -> dict[str, dict]:
    """Copy source schemas and add a native one-channel depth image feature."""
    if rgb_key not in source.features:
        raise KeyError(
            f"Source RGB feature {rgb_key!r} was not found. "
            f"Available features: {sorted(source.features)}"
        )

    features = {
        key: deepcopy(value)
        for key, value in source.features.items()
        if key not in AUTOMATIC_FEATURES
    }

    # Codec details describe existing files and cannot be copied to newly
    # encoded streams. Keep source visual modalities as ordinary RGB videos.
    for key, feature in list(features.items()):
        if feature.get("dtype") in ("image", "video"):
            height, width, channels = tuple(feature["shape"])
            if channels != 3:
                raise ValueError(
                    f"Source visual {key!r} has shape {feature['shape']}; expected RGB HWC."
                )
            features[key] = {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": feature.get("names", ["height", "width", "channels"]),
            }

    height, width, _ = tuple(features[rgb_key]["shape"])
    features[depth_key] = {
        "dtype": "image",
        "shape": (height, width, 1),
        "names": ["height", "width", "channels"],
        "info": {
            "is_depth_map": True,
            "depth_unit": "mm",
        },
    }
    return features


def _validated_source(args: argparse.Namespace):
    source_kwargs: dict[str, Any] = {"repo_id": args.source_repo}
    if args.source_root is not None:
        source_kwargs["root"] = args.source_root
    source_all = LeRobotDataset(**source_kwargs)
    source_root = Path(source_all.root)

    if source_all.meta.episodes is None:
        raise ValueError("The source dataset has no episode metadata.")
    metadata_rows = [
        source_all.meta.episodes[index]
        for index in range(len(source_all.meta.episodes))
    ]
    original_ids = [int(row["episode_index"]) for row in metadata_rows]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError("meta/episodes contains duplicate episode_index values")

    if source_all.num_episodes != len(metadata_rows):
        print(
            "WARNING: info.json reports "
            f"{source_all.num_episodes} episodes, but meta/episodes contains "
            f"{len(metadata_rows)}. Converting only metadata-backed episodes."
        )
    print(f"Valid source episode IDs: {original_ids}")

    source, original_ids, expected_lengths = sanitize_source_in_memory(
        source_all, metadata_rows
    )
    print(
        f"Sanitized source: {len(original_ids)} episodes, "
        f"{len(source)} frames"
    )
    return source, source_root, original_ids, expected_lengths


def convert(args: argparse.Namespace) -> None:
    installed = require_supported_lerobot()
    if args.source_repo == args.target_repo:
        raise ValueError("Source and target repo IDs must be different.")

    source, source_root, original_episode_ids, expected_lengths = _validated_source(args)
    rgb_key = f"observation.images.{args.camera}"
    depth_key = f"observation.images.{args.camera}_depth"
    features = native_tiff_features(source, rgb_key, depth_key)

    # Validate every NPY sequence and calibration before creating the target.
    depth_episodes: dict[int, tuple[Path, float]] = {}
    for logical_index, original_episode_index in enumerate(original_episode_ids):
        length = expected_lengths[logical_index]
        depth_episodes[logical_index] = depth_episode_info(
            source_root,
            original_episode_index,
            args.camera,
            length,
        )
        if original_episode_index != logical_index:
            print(
                f"Mapping target episode {logical_index:06d} from source "
                f"episode {original_episode_index:06d}"
            )

    expected_total = sum(expected_lengths.values())
    if expected_total != len(source):
        raise ValueError(
            f"Episode metadata totals {expected_total} frames, but dataset has {len(source)}"
        )

    target_kwargs: dict[str, Any] = {
        "repo_id": args.target_repo,
        "fps": source.fps,
        "features": features,
        "robot_type": source.meta.robot_type,
        # RGB remains video; the depth feature itself is dtype=image and is
        # therefore retained as TIFF by LeRobot 0.6.x.
        "use_videos": True,
        "image_writer_threads": args.image_writer_threads,
    }
    if args.target_root is not None:
        target_kwargs["root"] = args.target_root
    target = LeRobotDataset.create(**target_kwargs)

    tasks = task_lookup(source)
    source_visual_keys = {
        key
        for key, feature in features.items()
        if feature.get("dtype") in ("image", "video") and key != depth_key
    }

    current_episode: int | None = None
    saved_episodes = 0
    frames_in_episode = 0
    for dataset_index in range(len(source)):
        sample = source[dataset_index]
        episode_index = scalar(sample["episode_index"])
        frame_index = scalar(sample["frame_index"])

        if episode_index not in expected_lengths:
            raise ValueError(f"Reader returned stale episode {episode_index}")
        if current_episode is None:
            current_episode = episode_index
        elif episode_index != current_episode:
            if episode_index <= current_episode:
                raise ValueError(
                    f"Unexpected episode ordering: {current_episode} -> {episode_index}"
                )
            target.save_episode(parallel_encoding=False)
            print(f"Saved target episode {saved_episodes} ({frames_in_episode} frames)")
            saved_episodes += 1
            current_episode = episode_index
            frames_in_episode = 0

        if frame_index != frames_in_episode:
            raise ValueError(
                f"Episode {episode_index}: expected frame {frames_in_episode}, got {frame_index}"
            )

        frame: dict[str, Any] = {}
        for key in features:
            if key == depth_key:
                continue
            if key not in sample:
                raise KeyError(f"Frame {dataset_index} is missing source feature {key!r}")
            value = sample[key]
            frame[key] = (
                visual_to_hwc_uint8(value)
                if key in source_visual_keys
                else as_numpy(value)
            )

        camera_dir, depth_scale_m = depth_episodes[episode_index]
        raw_path = camera_dir / f"frame_{frame_index:06d}.npy"
        raw_depth = np.load(raw_path, allow_pickle=False)
        depth_mm = depth_sensor_units_to_mm(raw_depth, depth_scale_m)

        expected_shape = tuple(features[depth_key]["shape"])
        if depth_mm.shape != expected_shape:
            raise ValueError(
                f"{raw_path}: converted shape {depth_mm.shape} does not match "
                f"{expected_shape}. Was depth aligned to RGB?"
            )
        frame[depth_key] = depth_mm

        task_index = scalar(sample["task_index"])
        if task_index not in tasks:
            raise KeyError(f"Unknown task_index {task_index} at frame {dataset_index}")
        frame["task"] = tasks[task_index]

        target.add_frame(frame)
        frames_in_episode += 1

    if frames_in_episode:
        target.save_episode(parallel_encoding=False)
        print(f"Saved target episode {saved_episodes} ({frames_in_episode} frames)")

    target.finalize()
    depth_info = target.features[depth_key]
    if depth_info["dtype"] != "image" or tuple(depth_info["shape"]) != (480, 640, 1):
        raise RuntimeError(f"Unexpected target depth schema: {depth_info}")

    print(f"Conversion complete: {target.root}")
    print(f"LeRobot: {installed}")
    print(f"Feature: {depth_key}")
    print("Storage: lossless uint16 TIFF, millimetres, one channel")
    print(f"Frames: {target.num_frames}; episodes: {target.num_episodes}")

    if args.push_to_hub:
        print(f"Uploading private dataset {args.target_repo} ...")
        target.push_to_hub(private=True)
        print("Upload complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Telerobot RealSense .npy sidecars into native uint16-mm "
            "TIFF depth images for LeRobot >= 0.6.0."
        )
    )
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--camera", default="gripperCam")
    parser.add_argument("--image-writer-threads", type=int, default=2)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())

#!/usr/bin/env python3
"""Convert Telerobot uint16 depth sidecars into an ACT visual feature.

The source dataset is left untouched.  The output is a new LeRobot dataset
containing all original observations/actions plus a three-channel uint8 depth
view named ``observation.images.<camera>_depth``.

Depth encoding is deliberately fixed across the whole dataset:

* raw value 0 -> RGB [0, 0, 0] (invalid measurement)
* valid near/far range -> grayscale values 1..255
* values outside the range are clipped

Use the exact same encoding at policy rollout time.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import datasets
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


AUTOMATIC_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def scalar(value: Any) -> int:
    """Return an integer from a Python, NumPy, or single-value Torch scalar."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def as_numpy(value: Any) -> Any:
    """Detach Torch tensors while leaving ordinary scalar values unchanged."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value.copy() if isinstance(value, np.ndarray) else value


def visual_to_hwc_uint8(value: Any) -> np.ndarray:
    """Convert a decoded LeRobot visual tensor/array to HWC uint8."""
    image = as_numpy(value)
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D visual frame, got shape {image.shape}")

    # LeRobot readers normally return CHW Torch tensors; writers expect HWC.
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)

    if np.issubdtype(image.dtype, np.floating):
        # Decoded RGB is normally float in [0, 1].
        if image.size and float(np.nanmax(image)) <= 1.5:
            image = image * 255.0
        image = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def encode_depth(
    raw_depth: np.ndarray,
    depth_scale_m: float,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Map uint16 metric depth to an unambiguous three-channel uint8 image."""
    if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise ValueError(
            f"Expected a 2-D uint16 depth frame, got {raw_depth.dtype} {raw_depth.shape}"
        )

    valid = raw_depth != 0
    depth_m = raw_depth.astype(np.float32) * np.float32(depth_scale_m)
    encoded = np.zeros(raw_depth.shape, dtype=np.uint8)
    normalized = np.clip((depth_m[valid] - near_m) / (far_m - near_m), 0.0, 1.0)
    encoded[valid] = 1 + np.rint(254.0 * normalized).astype(np.uint8)
    return np.repeat(encoded[..., None], 3, axis=-1)


def task_lookup(dataset: LeRobotDataset) -> dict[int, str]:
    tasks = dataset.meta.tasks
    if tasks is None or len(tasks) == 0:
        raise ValueError("The source dataset has no task descriptions.")
    return {
        int(row["task_index"]): str(task)
        for task, row in tasks.iterrows()
    }


def episode_length(dataset: LeRobotDataset, episode_index: int) -> int:
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError("The source dataset has no episode metadata.")
    return int(episodes[episode_index]["length"])


def depth_episode_info(
    dataset_root: Path,
    episode_index: int,
    camera: str,
    expected_frames: int,
) -> tuple[Path, float]:
    episode_dir = dataset_root / "depth" / f"episode_{episode_index:06d}"
    marker = episode_dir / "episode.json"
    calibration_path = episode_dir / "calibration.json"
    camera_dir = episode_dir / camera

    if not marker.exists():
        raise FileNotFoundError(f"Missing completed depth marker: {marker}")
    metadata = json.loads(marker.read_text())
    if not metadata.get("complete", False):
        raise ValueError(f"Depth episode is not marked complete: {marker}")
    if int(metadata["num_frames"]) != expected_frames:
        raise ValueError(
            f"Episode {episode_index}: RGB/data has {expected_frames} frames, "
            f"but depth marker says {metadata['num_frames']}"
        )

    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing calibration: {calibration_path}")
    calibration = json.loads(calibration_path.read_text())
    if camera not in calibration:
        raise KeyError(f"Camera {camera!r} is absent from {calibration_path}")
    depth_scale_m = float(calibration[camera]["depth_scale_m"])
    if depth_scale_m <= 0:
        raise ValueError(f"Invalid depth scale {depth_scale_m} in {calibration_path}")

    frames = sorted(camera_dir.glob("frame_*.npy"))
    if len(frames) != expected_frames:
        raise ValueError(
            f"Episode {episode_index}: expected {expected_frames} {camera} depth files, "
            f"found {len(frames)}"
        )
    for frame_index, path in enumerate(frames):
        expected_name = f"frame_{frame_index:06d}.npy"
        if path.name != expected_name:
            raise ValueError(
                f"Episode {episode_index}: depth sequence has a gap; "
                f"expected {expected_name}, found {path.name}"
            )

    return camera_dir, depth_scale_m


def output_features(source: LeRobotDataset, rgb_key: str, depth_key: str) -> dict:
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

    # Video codec details describe the source files and must not be copied to a
    # newly encoded dataset. Keep only the logical visual schema.
    for key, feature in list(features.items()):
        if feature.get("dtype") in ("image", "video"):
            features[key] = {
                "dtype": "video",
                "shape": tuple(feature["shape"]),
                "names": feature.get("names", ["height", "width", "channels"]),
            }

    rgb_feature = features[rgb_key]
    features[depth_key] = {
        "dtype": "video",
        "shape": tuple(rgb_feature["shape"]),
        "names": rgb_feature.get("names", ["height", "width", "channels"]),
    }
    return features


def sanitize_source_in_memory(
    source: LeRobotDataset,
    metadata_rows: list[dict],
) -> tuple[LeRobotDataset, list[int], dict[int, int]]:
    """Drop stale data rows and reindex retained episodes without touching disk.

    A partially completed episode deletion can leave data Parquet rows with
    original IDs (for example 0..56) while meta/episodes retains only the valid
    rows (for example 0..9 and 17..56). LeRobot assumes episode IDs are dense
    row positions, so make that view true only in memory for conversion.
    """
    original_ids = [int(row["episode_index"]) for row in metadata_rows]
    id_map = {original_id: logical_id for logical_id, original_id in enumerate(original_ids)}
    expected_lengths = {
        logical_id: int(row["length"])
        for logical_id, row in enumerate(metadata_rows)
    }

    reader = source.reader
    if reader is None or not hasattr(reader, "hf_dataset"):
        raise RuntimeError("This converter requires LeRobot's standard DatasetReader.")

    loaded = source.hf_dataset
    transform = loaded.format["format_kwargs"].get("transform")
    # ``with_transform(None)`` leaves a callable-null transform in some
    # Hugging Face datasets releases. ``with_format(None)`` reliably returns
    # an unformatted copy suitable for Arrow filtering/mapping.
    raw = loaded.with_format(None)
    raw_episode_ids = np.asarray(raw["episode_index"], dtype=np.int64)
    keep_mask = np.isin(raw_episode_ids, np.asarray(original_ids, dtype=np.int64))
    keep_indices = np.flatnonzero(keep_mask).tolist()
    clean = raw.select(keep_indices)

    # Validate every original episode before changing any IDs.
    selected_ids = np.asarray(clean["episode_index"], dtype=np.int64)
    unique_ids, counts = np.unique(selected_ids, return_counts=True)
    actual_counts = {int(ep): int(count) for ep, count in zip(unique_ids, counts, strict=True)}
    for logical_id, row in enumerate(metadata_rows):
        original_id = int(row["episode_index"])
        expected = expected_lengths[logical_id]
        actual = actual_counts.get(original_id, 0)
        if actual != expected:
            raise ValueError(
                f"Source data episode {original_id} has {actual} rows; metadata expects {expected}."
            )

    def reindex_row(row: dict, new_index: int) -> dict:
        return {
            "episode_index": id_map[int(row["episode_index"])],
            "index": new_index,
        }

    clean = clean.map(
        reindex_row,
        with_indices=True,
        desc="Removing stale episodes and reindexing valid rows",
    )
    if transform is not None:
        clean.set_transform(transform)

    # Preserve all episode metadata (video file/timestamp locations and stats),
    # changing only identity and global frame bounds for the dense in-memory view.
    clean_episode_rows = []
    dataset_from_index = 0
    for logical_id, source_row in enumerate(metadata_rows):
        row = dict(source_row)
        length = expected_lengths[logical_id]
        row["episode_index"] = logical_id
        row["dataset_from_index"] = dataset_from_index
        row["dataset_to_index"] = dataset_from_index + length
        dataset_from_index += length
        clean_episode_rows.append(row)

    source.meta.episodes = datasets.Dataset.from_list(clean_episode_rows)
    source.meta.info.total_episodes = len(clean_episode_rows)
    source.meta.info.total_frames = dataset_from_index
    source.episodes = None
    reader.episodes = None
    reader.hf_dataset = clean
    if hasattr(reader, "_absolute_to_relative_idx"):
        reader._absolute_to_relative_idx = None
    if hasattr(reader, "_column_views"):
        reader._column_views.clear()
    if hasattr(reader, "_column_views_source"):
        reader._column_views_source = None

    return source, original_ids, expected_lengths


def convert(args: argparse.Namespace) -> None:
    if args.near_m < 0 or args.far_m <= args.near_m:
        raise ValueError("Require 0 <= --near-m < --far-m")
    if args.depth_offset != 0 and args.depth_offset_start is None:
        raise ValueError("--depth-offset requires --depth-offset-start")
    if args.depth_offset_start is not None and args.depth_offset_start < 0:
        raise ValueError("--depth-offset-start must be non-negative")
    if args.source_repo == args.target_repo:
        raise ValueError("Source and target repo IDs must be different.")

    source_kwargs = {"repo_id": args.source_repo}
    if args.source_root is not None:
        source_kwargs["root"] = args.source_root
    source_all = LeRobotDataset(**source_kwargs)
    source_root = Path(source_all.root)

    # The rows and episode_index values in meta/episodes are authoritative. A
    # deletion can leave stale Parquet rows and stale totals in info.json while
    # correctly retaining original IDs such as 0..9, 17..56 here.
    if source_all.meta.episodes is None:
        raise ValueError("The source dataset has no episode metadata.")
    metadata_rows: list[dict] = [
        source_all.meta.episodes[index]
        for index in range(len(source_all.meta.episodes))
    ]
    valid_episode_ids = [int(row["episode_index"]) for row in metadata_rows]
    expected_lengths = {
        int(row["episode_index"]): int(row["length"])
        for row in metadata_rows
    }
    if len(expected_lengths) != len(metadata_rows):
        raise ValueError("meta/episodes contains duplicate episode_index values")

    if source_all.num_episodes != len(metadata_rows):
        print(
            "WARNING: info.json reports "
            f"{source_all.num_episodes} episodes, but meta/episodes contains "
            f"{len(metadata_rows)}. Converting only metadata-backed episode IDs."
        )

    print(f"Valid source episode IDs: {valid_episode_ids}")
    source, original_episode_ids, expected_lengths = sanitize_source_in_memory(
        source_all, metadata_rows
    )
    expected_total_frames = sum(expected_lengths.values())
    print(
        f"Sanitized in-memory source: {len(original_episode_ids)} episodes, "
        f"{len(source)} frames"
    )

    rgb_key = f"observation.images.{args.camera}"
    depth_key = f"observation.images.{args.camera}_depth"
    features = output_features(source, rgb_key, depth_key)

    # Complete validation happens before target creation, so a missing frame
    # cannot leave behind a dataset that merely looks complete.
    depth_episodes: dict[int, tuple[Path, float]] = {}
    total_depth_frames = 0
    if args.depth_offset != 0:
        raise ValueError(
            "Do not use --depth-offset: the converter now reads original episode IDs "
            "directly from meta/episodes. Remove both depth-offset arguments."
        )

    for logical_index, original_episode_index in enumerate(original_episode_ids):
        length = expected_lengths[logical_index]
        depth_episodes[logical_index] = depth_episode_info(
            source_root, original_episode_index, args.camera, length
        )
        if original_episode_index != logical_index:
            print(
                f"Mapping target episode {logical_index:06d} from source RGB/data/depth "
                f"episode {original_episode_index:06d}"
            )
        total_depth_frames += length

    if total_depth_frames != len(source):
        raise ValueError(
            f"Episode metadata totals {total_depth_frames} frames, but dataset has {len(source)}"
        )

    target_kwargs = {
        "repo_id": args.target_repo,
        "fps": source.fps,
        "features": features,
        "robot_type": source.meta.robot_type,
        "use_videos": True,
        # A small thread count suits systems with limited host RAM.
        "image_writer_threads": args.image_writer_threads,
    }
    if args.target_root is not None:
        target_kwargs["root"] = args.target_root
    target = LeRobotDataset.create(**target_kwargs)

    tasks = task_lookup(source)
    visual_keys = {
        key for key, feature in features.items() if feature.get("dtype") in ("image", "video")
    }

    current_episode: int | None = None
    converted_episode_index = 0
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
            print(
                f"Saved target episode {converted_episode_index} from source episode "
                f"{current_episode} ({frames_in_episode} frames)"
            )
            converted_episode_index += 1
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
            frame[key] = visual_to_hwc_uint8(value) if key in visual_keys else as_numpy(value)

        camera_dir, depth_scale_m = depth_episodes[episode_index]
        raw_path = camera_dir / f"frame_{frame_index:06d}.npy"
        raw_depth = np.load(raw_path, allow_pickle=False)
        encoded_depth = encode_depth(
            raw_depth,
            depth_scale_m=depth_scale_m,
            near_m=args.near_m,
            far_m=args.far_m,
        )

        expected_shape = tuple(features[depth_key]["shape"])
        if encoded_depth.shape != expected_shape:
            raise ValueError(
                f"{raw_path}: converted shape {encoded_depth.shape} does not match "
                f"RGB feature shape {expected_shape}. Was depth aligned to RGB?"
            )
        frame[depth_key] = encoded_depth

        task_index = scalar(sample["task_index"])
        if task_index not in tasks:
            raise KeyError(f"Unknown task_index {task_index} at dataset frame {dataset_index}")
        frame["task"] = tasks[task_index]

        target.add_frame(frame)
        frames_in_episode += 1

    if frames_in_episode:
        target.save_episode(parallel_encoding=False)
        print(
            f"Saved target episode {converted_episode_index} from source episode "
            f"{current_episode} ({frames_in_episode} frames)"
        )

    target.finalize()
    print(f"Conversion complete: {target.root}")
    print(f"Feature created: {depth_key}")
    print(f"Frames: {target.num_frames}; episodes: {target.num_episodes}")

    if args.push_to_hub:
        print(f"Uploading private dataset {args.target_repo} ...")
        target.push_to_hub(private=True)
        print("Upload complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Telerobot depth .npy sidecars into an ACT image feature."
    )
    parser.add_argument("--source-repo", default="charlieK123/depth_dataSet")
    parser.add_argument("--target-repo", default="charlieK123/depth_dataSet_act_rgbd")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--camera", default="gripperCam")
    parser.add_argument("--near-m", type=float, default=0.07)
    parser.add_argument("--far-m", type=float, default=0.50)
    parser.add_argument(
        "--depth-offset-start",
        type=int,
        help=(
            "First RGB/data episode whose matching depth episode needs an offset. "
            "For example, use 10 when seven RGB episodes were deleted after episode 9."
        ),
    )
    parser.add_argument(
        "--depth-offset",
        type=int,
        default=0,
        help="Add this value to depth episode indices at/after --depth-offset-start.",
    )
    parser.add_argument("--image-writer-threads", type=int, default=2)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())

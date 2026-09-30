import json
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.processor import make_default_processors

from telerobot.logger import log_message
from telerobot.dataset.depth import DepthRecorder, recording_cameras, copy_depth_after_deletion


def _dataset_root(cfg) -> Path:
    """Resolve the on-disk path used for the configured dataset."""
    if cfg.dataset is None:
        raise ValueError("Dataset configuration is required.")
    return Path(cfg.dataset.root) if cfg.dataset.root is not None else HF_LEROBOT_HOME / cfg.dataset.repo_id


def _resume_dataset(cfg, logger) -> LeRobotDataset:
    """Re-open an existing dataset in write mode so more episodes can be appended."""
    if cfg.dataset is None:
        raise ValueError("Dataset configuration is required.")

    dataset_root = _dataset_root(cfg)

    # Current LeRobot versions expose resume() for write-mode append. Keep the
    # constructor fallback for older versions used by some Telerobot installs.
    resume_fn = getattr(LeRobotDataset, "resume", None)
    if callable(resume_fn):
        dataset = resume_fn(
            repo_id=cfg.dataset.repo_id,
            root=dataset_root,
            streaming_encoding=True,
        )
    else:
        dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_root,
            streaming_encoding=True,
        )

    log_message(
        logger,
        f"📂 Dataset writer ready: {cfg.dataset.repo_id} "
        f"({dataset.num_episodes} episodes so far)",
    )
    return dataset


def setup_dataset(robot, cfg, logger) -> LeRobotDataset | None:
    """Create a LeRobotDataset for episode recording, or None if not configured."""
    if cfg.dataset is None:
        return None

    teleop_action_processor, _robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    # Streaming encoding is always enabled with auto HW-accelerated codec (VideoToolbox on
    # macOS, NVENC on Linux with Nvidia, etc.) for near-instant save_episode() calls.
    VCODEC = "auto"

    dataset_root = _dataset_root(cfg)
    info_path = dataset_root / "meta" / "info.json"
    dataset_exists = info_path.exists()
    depth_cameras = recording_cameras(robot, cfg)
    if (dataset_root / "meta" / "depth.json").exists() and not depth_cameras:
        raise ValueError("This dataset records depth. Enable its RealSense depth cameras or use a new dataset.")
    if list((dataset_root / "depth").glob("episode_*.incomplete")):
        raise RuntimeError("Unfinished depth episode exists. Inspect/recover it before starting a new recording.")

    # If the dataset dir was initialised but no frames were ever recorded (0 total_frames),
    # LeRobotDataset can otherwise try to treat it as a readable/finalized dataset. Wipe only
    # this empty shell so a fresh writer can be created cleanly.
    if dataset_exists:
        with open(info_path) as f:
            info = json.load(f)
        if info.get("total_frames", 0) == 0:
            shutil.rmtree(dataset_root)
            dataset_exists = False

    if dataset_exists:
        dataset = _resume_dataset(cfg, logger)
    else:
        dataset = LeRobotDataset.create(
            repo_id=cfg.dataset.repo_id,
            fps=cfg.fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            streaming_encoding=True,
        )
        log_message(logger, f"📁 Dataset recording enabled: {cfg.dataset.repo_id}")

    if depth_cameras:
        dataset._telerobot_depth = DepthRecorder(dataset_root, depth_cameras, cfg.fps, dataset.num_episodes)
        log_message(logger, f"📏 Lossless depth recording enabled: {', '.join(depth_cameras)}")
    log_message(logger, f"🎬 Streaming video encoding active (vcodec={VCODEC})")
    return dataset


def end_active_episode(
    dataset: LeRobotDataset | None,
    logger,
) -> None:
    """Save and close the current episode if recording is active."""
    if dataset is None:
        return

    # New LeRobot API keeps the episode buffer inside DatasetWriter.
    # has_pending_frames() is the supported way to check for unsaved frames.
    if not dataset.has_pending_frames():
        log_message(logger, "⚠️ No frames recorded, skipping empty episode.")
        return

    depth = getattr(dataset, "_telerobot_depth", None)
    if depth is not None:
        depth.flush()
    dataset.save_episode()
    if depth is not None:
        depth.commit_episode(dataset.num_episodes - 1)

    log_message(
        logger,
        f"✅ Episode {dataset.num_episodes - 1} saved "
        f"({dataset.num_frames} total frames)",
    )


def record_step(
    dataset: LeRobotDataset | None,
    cfg,
    obs,
    action,
) -> None:
    """Record a single observation/action step into the dataset."""
    if dataset is None:
        return

    observation_frame = build_dataset_frame(
        dataset.features, obs, prefix=OBS_STR
    )
    action_frame = build_dataset_frame(
        dataset.features, action, prefix=ACTION
    )
    frame = {
        **observation_frame,
        **action_frame,
        "task": cfg.dataset.single_task,
    }
    depth = getattr(dataset, "_telerobot_depth", None)
    packets = depth.prepare(obs, dataset.num_episodes) if depth is not None else None
    dataset.add_frame(frame)
    if depth is not None:
        depth.submit(packets)


def finalize_dataset(dataset: LeRobotDataset | None, push_to_hub: bool, logger) -> bool:
    """Finalize a dataset so all parquet/video files are valid, then optionally upload it.

    Returns True when the local finalize succeeded and either the Hub upload succeeded or no
    upload was requested. Hub upload errors are logged but do not invalidate the local dataset.
    """
    if dataset is None:
        return True

    if dataset.has_pending_frames():
        raise RuntimeError(
            "Cannot finalize while an episode still has unsaved frames. "
            "Stop/save the current episode first."
        )

    # LeRobotDataset.finalize() already flushes streaming video encoders, closes parquet
    # writers, writes footer metadata, and finalizes episode metadata. Calling the streaming
    # encoder's close() separately is redundant and can make lifecycle handling brittle.
    depth = getattr(dataset, "_telerobot_depth", None)
    if depth is not None and depth.pending:
        raise RuntimeError("Depth episode is incomplete; refusing to finalize/upload it.")
    dataset.finalize()
    log_message(logger, f"📊 Total episodes recorded: {dataset.num_episodes}")

    if not push_to_hub:
        return True

    if dataset.num_episodes == 0:
        log_message(logger, "⚠️ No episodes recorded — skipping push to Hub.")
        return True

    try:
        log_message(logger, f"🚀 Pushing dataset '{dataset.repo_id}' to Hugging Face Hub...")
        _push_with_depth(dataset)
        log_message(logger, f"✅ Dataset '{dataset.repo_id}' pushed successfully.")
        return True
    except Exception as e:
        log_message(logger, f"❌ Failed to push dataset to Hub: {e}")
        log_message(
            logger,
            "   Local data is finalized and safe. Check network/authentication, then retry the checkpoint.",
        )
        return False


def checkpoint_dataset(
    dataset: LeRobotDataset | None, cfg, push_to_hub: bool, logger
) -> tuple[LeRobotDataset | None, bool]:
    """Finalize/upload the current dataset and reopen it for more recording.

    Returns ``(resumed_dataset, upload_ok)``. A Hub/network failure does not prevent the
    local dataset from being reopened, so collection can continue safely after a failed
    upload. This function is intended to run in a background thread; callers must not add
    frames while it is running.
    """
    if dataset is None:
        return None, True

    if dataset.has_pending_frames():
        raise RuntimeError(
            "Cannot checkpoint while an episode is recording. Stop/save the episode first."
        )

    upload_ok = finalize_dataset(dataset, push_to_hub, logger)

    # finalize() intentionally closes the writer. Re-open the same local dataset in write
    # mode so the operator can immediately continue collecting episodes without restarting.
    resumed = _resume_dataset(cfg, logger)
    if hasattr(dataset, "_telerobot_depth"):
        resumed._telerobot_depth = dataset._telerobot_depth
    if upload_ok:
        log_message(logger, "✅ Dataset checkpoint complete; recording can continue.")
    else:
        log_message(
            logger,
            "⚠️ Local checkpoint is safe and recording can continue, but the Hub upload failed.",
        )
    return resumed, upload_ok


def delete_episodes_from_dataset(
    repo_id: str,
    episode_indices: list[int],
    root: str | None = None,
    push_to_hub: bool = False,
    logger=None,
) -> bool:
    """Delete specific episodes from a LeRobot dataset.

    The original dataset is replaced in-place: episodes are deleted into a
    temporary copy which then replaces the original directory.

    Args:
        repo_id: Repository ID of the dataset (e.g. "user/dataset-name").
        episode_indices: List of episode indices to delete.
        root: Optional root directory override. If None, uses the default
              HF_LEROBOT_HOME / repo_id location.
        push_to_hub: If True, push the updated dataset to the Hugging Face Hub.
        logger: Optional logger instance for status messages.
    """
    dataset_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_root}")

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    log_message(logger, f"📂 Loaded dataset: {repo_id} ({dataset.num_episodes} episodes, {dataset.num_frames} frames)")
    log_message(logger, f"🗑️  Deleting episodes: {episode_indices}")

    tmp_repo_id = f"{repo_id}_tmp_delete"
    tmp_root = dataset_root.parent / f"{dataset_root.name}_tmp_delete"

    try:
        new_dataset = delete_episodes(
            dataset=dataset,
            episode_indices=episode_indices,
            output_dir=tmp_root,
            repo_id=tmp_repo_id,
        )
        log_message(logger, f"✅ New dataset has {new_dataset.num_episodes} episodes, {new_dataset.num_frames} frames")

        # Preserve/reindex lossless depth alongside LeRobot's RGB rewrite.
        copy_depth_after_deletion(dataset_root, tmp_root, episode_indices, dataset.num_episodes)

        # Replace original with the new dataset
        shutil.rmtree(dataset_root)
        tmp_root.rename(dataset_root)
        log_message(logger, f"✅ Dataset at {dataset_root} updated in-place.")

        if push_to_hub:
            try:
                updated_dataset = LeRobotDataset(repo_id=repo_id, root=root)
                log_message(logger, f"🚀 Pushing updated dataset '{repo_id}' to Hugging Face Hub...")
                _push_with_depth(updated_dataset)
                log_message(logger, f"✅ Dataset '{repo_id}' pushed successfully.")
            except Exception as exc:
                # The local deletion is already complete and should not be rolled back
                # just because the network/Hub update failed.
                log_message(logger, f"❌ Failed to push updated dataset to Hub: {exc}")
                log_message(
                    logger,
                    "   Previous episode was deleted locally. Recording can continue; retry a checkpoint later.",
                )
                return False
        return True
    except Exception:
        # Clean up temp dir on failure
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise


def delete_previous_episode_checkpoint(
    dataset: LeRobotDataset | None,
    cfg,
    push_to_hub: bool,
    logger,
) -> tuple[LeRobotDataset | None, int | None, bool]:
    """Delete the most recently saved episode, then reopen for recording.

    This mirrors :func:`checkpoint_dataset`: the writer is finalized first so
    parquet/video files are valid, the last saved episode is removed, the
    optional Hub copy is updated, and finally the same dataset is reopened in
    write mode. It is intended to run in a background thread so camera/robot
    control stays responsive.

    Returns ``(resumed_dataset, deleted_episode_index, upload_ok)``.
    """
    if dataset is None:
        return None, None, True

    if dataset.has_pending_frames():
        raise RuntimeError(
            "Cannot delete the previous episode while an episode is recording. "
            "Stop/save the active episode first."
        )

    if dataset.num_episodes <= 0:
        log_message(logger, "⚠️ No saved episodes are available to delete.")
        return dataset, None, True

    deleted_episode_index = dataset.num_episodes - 1

    # Close all writers before dataset_tools rewrites the dataset directory.
    dataset.finalize()
    log_message(logger, f"🗑️ Deleting previous episode {deleted_episode_index}...")

    dataset_root = _dataset_root(cfg)
    upload_ok = delete_episodes_from_dataset(
        repo_id=cfg.dataset.repo_id,
        episode_indices=[deleted_episode_index],
        root=str(dataset_root),
        push_to_hub=push_to_hub,
        logger=logger,
    )

    resumed = _resume_dataset(cfg, logger)
    if hasattr(dataset, "_telerobot_depth"):
        resumed._telerobot_depth = dataset._telerobot_depth
    log_message(
        logger,
        f"✅ Previous episode {deleted_episode_index} deleted; "
        f"{resumed.num_episodes} saved episode(s) remain.",
    )
    return resumed, deleted_episode_index, upload_ok


def _push_with_depth(dataset):
    """Sync depth and remove stale depth episodes before LeRobot tags the upload."""
    root = Path(dataset.root)
    if (root / "meta" / "depth.json").exists():
        if list((root / "depth").glob("episode_*.incomplete")):
            raise RuntimeError("Cannot upload an unfinished depth episode.")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=dataset.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=dataset.repo_id, repo_type="dataset", folder_path=root,
                          allow_patterns=["depth/**", "meta/depth.json"],
                          delete_patterns=["depth/**"],
                          commit_message="Synchronize lossless depth episodes")
    dataset.push_to_hub(tags=["TeLeRobot"])

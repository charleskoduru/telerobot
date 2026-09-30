"""Lossless depth sidecars, indexed by LeRobot episode and frame_index."""
import json
from pathlib import Path
from queue import Full, Queue
import shutil
import threading

import numpy as np

from telerobot.cameras.realsense import PairedRealSenseCamera


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def recording_cameras(robot, cfg):
    """Map actual observation keys to cameras, including dual-arm prefixes."""
    output = {}
    if len(cfg.arms) == 1:
        output = dict(robot.cameras)
    else:
        for side in ("left", "right"):
            for name, cam in getattr(robot, f"{side}_arm").cameras.items():
                output[f"{side}_{name}"] = cam
    return {name: cam for name, cam in output.items()
            if isinstance(cam, PairedRealSenseCamera) and cam.config.use_depth}


class DepthRecorder:
    """A bounded disk writer; overload fails explicitly instead of dropping frames.

    Episode directories are marked incomplete until BOTH RGB and depth save.
    These files are companion data; they are not LeRobot policy image features.
    """
    def __init__(self, root, cameras, fps, start_episode):
        self.root = Path(root)
        self.cameras = cameras
        self.fps = fps
        self._queue = Queue(maxsize=32)
        self._error = None
        self._episode = None
        self._count = 0
        self._stage = None
        self._thread = None
        for name in cameras:
            if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
                raise ValueError(f"Camera name '{name}' must contain only letters, digits, underscores or hyphens.")
        depth_root = self.root / "depth"
        incomplete = sorted(depth_root.glob("episode_*.incomplete"))
        if incomplete:
            raise RuntimeError(f"Unfinished depth episode found: {incomplete[0]}. "
                               "Inspect/recover it or move it aside before recording more episodes.")
        schema = {name: {"serial_number": cam.config.serial_number,
                         "width": cam.width, "height": cam.height,
                         "depth_width": cam.config.depth_width,
                         "depth_height": cam.config.depth_height,
                         "aligned_to_color": cam.config.align_depth}
                  for name, cam in cameras.items()}
        meta_path = self.root / "meta" / "depth.json"
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            if existing["cameras"] != schema or existing["fps"] != fps:
                raise ValueError("Depth camera settings differ from this dataset. Use a new dataset repo_id/root.")
        else:
            _write_json(meta_path, {"format_version": 1, "format": "numpy_uint16_npy",
                                   "fps": fps, "cameras": schema,
                                   "first_depth_episode": start_episode,
                                   "match": "episode_index + frame_index",
                                   "meters": "stored_uint16 * calibration.depth_scale_m",
                                   "invalid_depth_value": 0})

    @property
    def pending(self):
        return self._episode is not None

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("Depth writing failed; incomplete files were kept for recovery.") from self._error

    def prepare(self, obs, episode_index):
        self._check_error()
        if self._queue.full():
            raise RuntimeError("Depth disk writer cannot keep up. Reduce camera FPS/resolution or use a faster disk.")
        packets = {}
        for name, cam in self.cameras.items():
            rgb = obs.get(name)
            # Some BiSOFollower versions expose top-level camera keys too.
            if rgb is None and name.startswith(("left_", "right_")):
                rgb = obs.get(name.split("_", 1)[1])
            if rgb is None:
                raise RuntimeError(f"RGB observation is missing for depth camera '{name}'.")
            packet = cam.packet_for_rgb(rgb)
            if packet.depth is None or packet.depth.dtype != np.uint16 or packet.depth.ndim != 2:
                raise ValueError(f"Camera '{name}' did not return a uint16 depth image.")
            packets[name] = packet
        if self._episode is None:
            final = self.root / "depth" / f"episode_{episode_index:06d}"
            stage = final.with_name(final.name + ".incomplete")
            if final.exists() or stage.exists():
                raise FileExistsError(f"Refusing to overwrite existing depth episode: {final}")
            stage.mkdir(parents=True)
            for name in packets:
                (stage / name).mkdir()
            _write_json(stage / "calibration.json", {name: packet.calibration for name, packet in packets.items()})
            self._episode, self._stage, self._count = episode_index, stage, 0
            self._thread = threading.Thread(target=self._write_loop, name="depth-disk-writer", daemon=True)
            self._thread.start()
        elif self._episode != episode_index:
            raise RuntimeError("The previous depth episode has not been committed.")
        return packets

    def submit(self, packets):
        # One producer; prepare() reserved availability before dataset.add_frame().
        try:
            self._queue.put_nowait((self._count, packets))
        except Full as exc:
            raise RuntimeError("Depth queue is full; recording stopped without silently dropping frames.") from exc
        self._count += 1

    def _write_loop(self):
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                if self._error is not None:
                    continue
                index, packets = payload
                row = {"episode_index": self._episode, "frame_index": index,
                       "timestamp": index / self.fps, "cameras": {}}
                for name, packet in packets.items():
                    relative = f"{name}/frame_{index:06d}.npy"
                    target = self._stage / relative
                    temporary = target.with_suffix(".npy.tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, packet.depth, allow_pickle=False)
                    temporary.replace(target)
                    row["cameras"][name] = {"file": relative, **packet.metadata}
                with (self._stage / "frames.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            except Exception as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def flush(self):
        self._queue.join()
        self._check_error()

    def commit_episode(self, saved_episode_index):
        self.flush()
        if saved_episode_index != self._episode:
            raise RuntimeError("RGB and depth episode indices disagree; incomplete depth retained.")
        _write_json(self._stage / "episode.json", {"episode_index": self._episode,
                                                  "num_frames": self._count,
                                                  "fps": self.fps, "complete": True})
        self.close_worker()
        final = self._stage.with_name(f"episode_{self._episode:06d}")
        if final.exists():
            raise FileExistsError(f"Depth target already exists: {final}")
        self._stage.rename(final)
        self._episode, self._stage, self._count = None, None, 0

    def close_worker(self):
        if self._thread is not None:
            self._queue.put(None)
            self._queue.join()
            self._thread.join()
            self._thread = None


def copy_depth_after_deletion(source_root, output_root, deleted_indices, total_episodes):
    """Preserve companion files and reindex them before replacing the dataset."""
    source_root, output_root = Path(source_root), Path(output_root)
    depth_root = source_root / "depth"
    if list(depth_root.glob("episode_*.incomplete")):
        raise RuntimeError("Recover unfinished depth episodes before deleting/reindexing episodes.")
    deleted = set(deleted_indices)
    mapping = {old: new for new, old in enumerate(i for i in range(total_episodes) if i not in deleted)}
    for old, new in mapping.items():
        source = depth_root / f"episode_{old:06d}"
        if not source.exists():
            continue  # RGB-only episodes may precede first_depth_episode.
        target = output_root / "depth" / f"episode_{new:06d}"
        shutil.copytree(source, target)
        episode_path = target / "episode.json"
        info = json.loads(episode_path.read_text())
        info["episode_index"] = new
        _write_json(episode_path, info)
        manifest = target / "frames.jsonl"
        temporary = manifest.with_suffix(".jsonl.tmp")
        with manifest.open() as reader, temporary.open("w") as writer:
            for line in reader:
                row = json.loads(line)
                row["episode_index"] = new
                writer.write(json.dumps(row) + "\n")
        temporary.replace(manifest)
    meta_path = source_root / "meta" / "depth.json"
    if meta_path.exists():
        info = json.loads(meta_path.read_text())
        first = info["first_depth_episode"]
        info["first_depth_episode"] = sum(old < first for old in mapping)
        _write_json(output_root / "meta" / "depth.json", info)

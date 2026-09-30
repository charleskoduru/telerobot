# Record RGB and RealSense depth episodes

This update keeps your existing RGB LeRobot videos, joint observations, actions,
leader mode, VR controls and recording buttons. For each enabled RealSense camera,
it also records the corresponding 16-bit depth array and sensor timestamps.
Depth comes from the same SDK frameset as the RGB observation, rather than a
second independent camera read. Multiple cameras have separate clocks; this
update does not enable hardware synchronization between different cameras.

## Setup in your existing working environment/container

1. Extract this updated project and use its `src/telerobot` files in your working
   Telerobot installation. Do not apply the old `desktop.patch` or `lerobot.patch`
   again; those files came from the original upload and are not this update.
2. In the environment/container where you run Telerobot:

   ```bash
   python -m pip install -r requirements-realsense.txt
   python -m pip install -e .
   python tools/list_realsense.py
   ```

   This uses your existing LeRobot environment; it does not upgrade LeRobot.
   The optional SDK dependency is separate so webcam-only installations and the
   original Poetry lockfile remain usable.
3. Edit `examples/config/realsense.yaml` with your real camera serial numbers,
   keeping them in quotes. The example switches `gripperCam` and `bevCam` to
   RealSense and keeps `baseCam` as an OpenCV RGB camera. Keep only the cameras
   you actually have, and keep the arm's `cameras` list consistent. Copy your
   current robot ports, camera paths and workspace settings if they differ.
4. Use a new dataset name such as `charlieK123/forceps_RGBD`, then run:

   ```bash
   telerobot run --config examples/config/realsense.yaml
   ```

5. Start, stop and save episodes with your usual desktop/VR recording controls.
   Saving a dataset uploads the depth companion files too when `push_to_hub`
   is enabled. Deleting episodes preserves and reindexes the remaining depth.

Your original `config.yaml` remains a working RGB-only example. To enable a
RealSense camera there instead, replace its OpenCV camera entry with:

```yaml
bevCam:
  type: realsense
  serial_number: "YOUR_CAMERA_SERIAL"
  width: 640
  height: 480
  fps: 30
  use_depth: true
  depth_width: 640
  depth_height: 480
  align_depth: true
  vr_gamma: 1.0
  vr_gain: 1.0
  vr_brightness: 15
```

Replace `YOUR_CAMERA_SERIAL` with the numeric serial printed by the listing tool.
Remove `index` and `fourcc` for this camera. Each RealSense must have a different
serial and belong to exactly one arm. OpenCV cameras can remain in the same
configuration. Both leader and VR modes use the same recorder.

`align_depth: true` projects depth onto RGB pixels. Set it to `false` to store
native, unaligned depth instead. The stored depth remains uint16 in either mode;
alignment changes the depth image geometry. The episode includes native and
stored depth intrinsics, RGB intrinsics, depth-to-color extrinsics, and the
camera's actual depth scale. Choose resolutions/FPS supported by your camera;
unsupported combinations fail at camera startup with a diagnostic.

For Docker, the SDK needs USB access in the SAME container as Telerobot. Existing
`privileged: true` containers normally have that access; otherwise expose USB
with a mapping such as `/dev/bus/usb:/dev/bus/usb` and appropriate device
permissions. Passing only `/dev/video*` is insufficient for SDK depth capture.
Close RealSense Viewer or other programs using these cameras before starting.

## Files saved in the dataset root

| Path | Contents |
| --- | --- |
| `videos/` | Existing RGB videos and normal LeRobot metadata |
| `data/` | Existing robot observations and actions |
| `meta/depth.json` | Depth camera settings, format, and first depth episode |
| `depth/episode_000000/bevCam/frame_000000.npy` | Exact uint16 depth array for frame 0 |
| `depth/episode_000000/frames.jsonl` | Per-frame filenames, episode/frame indices, device frame numbers and timestamps |
| `depth/episode_000000/calibration.json` | Depth scale and camera calibration for each recorded depth camera |
| `depth/episode_000000/episode.json` | Saved frame count and completion marker |

Match depth to the LeRobot RGB observation using **episode_index + frame_index**.
The JSONL `timestamp` is the LeRobot frame time, `frame_index / robot.fps`.
Sensor timestamps are stored separately in milliseconds, with their timestamp
clock domains. Host timestamps identify when the SDK frameset reached the
capture thread; they are not robot joint measurement timestamps.

Load a depth frame and convert it to meters:

```python
import json
from pathlib import Path
import numpy as np

episode = Path("/path/to/dataset/depth/episode_000000")
calibration = json.loads((episode / "calibration.json").read_text())["bevCam"]
depth_raw = np.load(episode / "bevCam/frame_000000.npy", allow_pickle=False)
depth_m = depth_raw.astype(np.float32) * calibration["depth_scale_m"]
valid = depth_raw != 0  # Zero means no valid depth measurement.
```

Depth is stored as `.npy`, not colorized RGB images or lossy video, to preserve
its measured values exactly. At 640x480@30, uncompressed depth uses about
18.4 MB/s (1.1 GB/minute) **per camera**, plus RGB videos. Disk writes run in a
bounded background queue; slow storage or write errors stop recording with an
explicit error rather than silently dropping depth frames.

An episode is marked `.incomplete` until both RGB and depth save successfully.
If a process crashes or storage fails, these files are kept, and restarting
refuses to overwrite them. Inspect/recover them and the matching RGB episode,
or move the incomplete directory outside the dataset root before resuming.
Do not merely rename it to a complete episode: RGB/depth counts must agree.
A dataset that has depth enabled cannot subsequently be resumed in RGB-only
mode. If you deliberately append depth to an existing RGB dataset, earlier
episodes stay RGB-only, and `first_depth_episode` documents that boundary.
Use a new repo/root if camera serials, geometry or FPS change.

These are **depth companion files**, not extra LeRobot policy input features.
Your existing ACT/SmolVLA training and rollout commands continue to use RGB.
Training a policy on depth requires a separate dataset-loader/model change.
For shiny surgical instruments, retain the RGB views: depth may contain missing
or unreliable measurements on reflective surfaces; this recorder keeps zeros
and all other sensor values without filling or filtering them.

## Validation performed

Hardware-free tests cover configuration, exact uint16 preservation, RGB/depth
pairing during concurrent preview, stale frames, episode completion, resume,
multiple cameras, disk failures, bounded-queue overload, and deletion/reindexing.
Run them with:

```bash
python -m unittest discover -s tests -v
```

RealSense hardware and your installed LeRobot/robot combination are not available
in the editing environment, so actual USB capture, robot motion and Hub uploads
still need a short test on your setup. First check serial discovery, then record
one short episode and confirm the depth frame count and shape before collecting
a full dataset.

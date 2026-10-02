# Native TIFF depth for ACT (LeRobot 0.6.2)

This project contains two separate changes:

1. `tools/convert_depth_to_tiff.py` creates a new LeRobot dataset with
   lossless, one-channel `uint16` depth TIFFs in millimetres.
2. `patches/lerobot-0.6.2-act-native-depth.patch` teaches ACT to process a
   one-channel visual through a separate depth backbone and fixes `uint16`
   normalization during live rollout.

The original `charlieK123/depth_dataSet` repository is never modified.

## 1. Apply the ACT patch

From the LeRobot 0.6.2 checkout mounted at `/workspace/lerobot`:

```bash
cd /workspace/lerobot
git apply --check /workspace/telerobot/patches/lerobot-0.6.2-act-native-depth.patch
git apply /workspace/telerobot/patches/lerobot-0.6.2-act-native-depth.patch
```

Confirm the environment:

```bash
/workspace/.venv/bin/python - <<'PY'
from importlib.metadata import version
print("LeRobot:", version("lerobot"))
assert version("lerobot").startswith("0.6.")
PY
```

## 2. Convert the raw RealSense sidecars

The converter reads each episode's `calibration.json`. It applies:

```text
millimetres = round(raw_uint16 * depth_scale_m * 1000)
```

This is important for the D405: a common scale is 0.0001 m per raw unit, so
the raw NPY values must not be relabelled as millimetres without conversion.

```bash
cd /workspace/telerobot

/workspace/.venv/bin/python tools/convert_depth_to_tiff.py \
  --source-repo charlieK123/depth_dataSet \
  --source-root /workspace/.cache/lerobot/charlieK123/depth_dataSet \
  --target-repo charlieK123/depth_dataSet_native_tiff_rgbd \
  --target-root /workspace/.cache/lerobot/charlieK123/depth_dataSet_native_tiff_rgbd \
  --camera gripperCam \
  --image-writer-threads 2 \
  --push-to-hub
```

Do not point `--target-root` at an existing dataset. LeRobot deliberately
refuses to overwrite it.

The expected depth feature is:

```json
{
  "observation.images.gripperCam_depth": {
    "dtype": "image",
    "shape": [480, 640, 1],
    "names": ["height", "width", "channels"],
    "info": {
      "is_depth_map": true,
      "depth_unit": "mm"
    }
  }
}
```

With 20,755 frames, raw 640x480 uint16 TIFF depth requires roughly 12.8 GB
before filesystem and repository overhead. Keep at least 16 GB free.

## 3. Validate the new dataset

```bash
/workspace/.venv/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo = "charlieK123/depth_dataSet_native_tiff_rgbd"
root = "/workspace/.cache/lerobot/charlieK123/depth_dataSet_native_tiff_rgbd"
key = "observation.images.gripperCam_depth"
ds = LeRobotDataset(repo, root=root, depth_output_unit="mm")

feature = ds.features[key]
print("Episodes:", ds.num_episodes)
print("Frames:", len(ds))
print("Feature:", feature)
assert ds.num_episodes == 50
assert len(ds) == 20755
assert feature["dtype"] == "image"
assert tuple(feature["shape"]) == (480, 640, 1)
assert feature["info"]["is_depth_map"] is True
assert feature["info"]["depth_unit"] == "mm"

sample = ds[len(ds) // 2][key]
print("Reader:", sample.dtype, tuple(sample.shape), sample.min().item(), sample.max().item())
assert tuple(sample.shape) == (1, 480, 640)
print("Native TIFF RGB-D dataset passed validation")
PY
```

LeRobot writes each depth frame through a lossless `uint16` TIFF and embeds
those TIFF bytes in the dataset's Parquet image column. It then removes the
temporary standalone `.tiff` staging files. Therefore, not seeing an
`images/.../*.tiff` directory after conversion is expected; `dtype: image`,
`is_depth_map: true`, and the exact decoded values are the relevant checks.

## 4. Train a new ACT policy

The old model was trained on an 8-bit, three-channel imitation of depth and
must not be resumed. Train a new policy after applying the ACT patch:

```bash
lerobot-train \
  --dataset.repo_id=charlieK123/depth_dataSet_native_tiff_rgbd \
  --dataset.depth_output_unit=mm \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.repo_id=charlieK123/gripper_native_tiff_depth_ACT \
  --policy.push_to_hub=true \
  --output_dir=outputs/train/act_native_tiff_rgbd_b8 \
  --job_name="forceps_act_native_tiff_depth" \
  --batch_size=8 \
  --num_workers=2 \
  --steps=50000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --env_eval_freq=0 \
  --wandb.enable=false
```

Your RTX 4080 Laptop GPU has 12 GB VRAM. Batch size 16 previously ran out of
memory with four visual streams, so begin with batch size 8 and reduce to 4 if
necessary.

## 5. Rollout requirement

Live rollout must provide the exact key
`observation.images.gripperCam_depth`, shaped `(1, 480, 640)` after batching,
in millimetres. The included LeRobot patch converts incoming `uint16` data to
`float32` before applying the saved training statistics, avoiding the earlier
`uint16 without overflow` failure.

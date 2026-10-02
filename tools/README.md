# Workspace Bounds GUI

## Native depth TIFF conversion

`convert_depth_to_tiff.py` converts the lossless RealSense `.npy` sidecars in
`charlieK123/depth_dataSet` into a new LeRobot 0.6.x dataset whose gripper depth
feature is `(480, 640, 1)` `uint16` millimetres stored as TIFF images. See
`NATIVE_DEPTH_ACT.md` for conversion, validation, ACT patching, and training.

A small browser-based tool for recording safe end-effector workspace bounds for the SO-101 robot.

This tool reads the robot's live joint angles, computes the end-effector XYZ pose using forward kinematics, and lets you capture corner points of the workspace. After capturing points, it outputs the `min: [...]` and `max: [...]` values needed for `config.yaml`.

---

## What it does

The GUI helps generate this part of `config.yaml`:
``` bash
python workspace_bounds_gui.py 
```
```yaml
end_effector_bounds:
  min: [0.20, -0.20, 0.05]
  max: [0.40,  0.20, 0.30]

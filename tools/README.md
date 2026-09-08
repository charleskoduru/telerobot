# Workspace Bounds GUI

A small browser-based tool for recording safe end-effector workspace bounds for the SO-101 robot.

This tool reads the robot's live joint angles, computes the end-effector XYZ pose using forward kinematics, and lets you capture corner points of the workspace. After capturing points, it outputs the `min: [...]` and `max: [...]` values needed for `config.yaml`.

---

## What it does

The GUI helps generate this part of `config.yaml`:

```yaml
end_effector_bounds:
  min: [0.20, -0.20, 0.05]
  max: [0.40,  0.20, 0.30]
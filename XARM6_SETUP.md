# xArm6 + Robotiq 2F-85 simulation with a GELLO leader

This update is based on `charleskoduru/telerobot` branch `teleopSim`, commit
`faf4d7877578ff98b65cfd995e7d1b42f5ef14c4`. It controls the simulated robot only.
No physical xArm connection or robot IP is used.

## Choose the robot

In the supplied `config.yaml`, change the first setting:

```yaml
profile: xarm6  # or so101
```

Each entry under `profiles` contains its robot, leader, cameras, and workspace.
The SO101 profile retains your previous configuration, wrist offset, and dataset
settings. The xArm6 profile has no dataset enabled by default, so it cannot append
seven-channel xArm data to your existing six-channel SO101 dataset.
Legacy YAML files without profiles still work. Separate standalone examples:

- `examples/config/so101-simulation.yaml`
- `examples/config/xarm6-gello.yaml`

## Install

Use the separate Docker bundle alongside this code. The Docker Compose host mount
must contain this updated `telerobot` folder. The Dockerfile's upstream clone is
only a fallback; it does not contain this update until you publish it yourself.

Existing environment alternative, from inside the active Python environment:

```bash
cd ~/leRobot_sim/telerobot
python -m pip install -e .
python -m pip install -r requirements-gello.txt
```

The SAPIEN 3.0.3, CUDA, LeRobot fork, and rendering dependencies come from your
existing Docker environment. GELLO input adds only Dynamixel SDK and pyserial.
It uses fresh synchronous reads, with the joint mapping from UFACTORY's GELLO
example, rather than starting the UFACTORY physical-robot script.

## Preview without the leader

Inside the container:

```bash
cd /workspace/telerobot
python -m telerobot.simulation.xarm6_preview --config config.yaml --animate
```

The selected profile must be xarm6. This previews slow base rotation, gripper
opening/closing, and exercises all configured camera captures. Ctrl+C stops it.
No USB leader or physical RealSense camera is opened. Rendering still requires
NVIDIA/Vulkan access, including with `viewer: false` if cameras are enabled.

## Connect your GELLO

On the host, locate its serial device:

```bash
ls -l /dev/serial/by-id/
```

Edit `profiles.xarm6.teleoperation.leader.port` in `config.yaml`. A stable
`/dev/serial/by-id/...` path is preferable to the default `/dev/ttyUSB0`.
The supplied Compose file exposes `/dev`, including those symlinks.

The example follows UFACTORY's xArm6 mapping:

| Setting | Value |
| --- | --- |
| Joint IDs in xArm joint1..joint6 order | `[1, 2, 4, 5, 6, 7]` |
| Joint signs | `[1, 1, -1, 1, 1, 1]` |
| Reference angles, radians | `[0, 0, -1.5708, 0, 1.5708, 0]` |
| Gripper motor | `8` (`-1` disables input and holds the simulated gripper open) |
| Unused physical joint held by torque | `[3]` |
| Bus | Dynamixel Protocol 2, 57600 baud, X-series position registers |

This assumes the same leader motor family, IDs, and assembly as that example.
`torque_joint_ids` explicitly enables holding torque at the unused motor's current
position while running, and releases it on normal shutdown. Set it to `[]` if your
leader has no such unused motor. Mapped arm/gripper motors are made backdrivable.
This is not a driver for the SO101's Feetech motors.

## Start teleoperation

```bash
cd /workspace/telerobot
telerobot run --config config.yaml
```

At the startup prompt, place GELLO at the configured reference pose and leave its
gripper open. Press Enter to calibrate. The simulation initializes to that pose,
then follows GELLO joint motion. Check direction one joint at a time.

Arm action values are degrees. `gripper.pos` is percent OPEN: 0 closed, 100 open.
The model converts the command into outer-knuckle joint targets; passive joints
follow closed-loop PhysX constraints. The public observation/action schema has
six arm channels and one gripper channel, despite the model's 12 physical DOFs.

If gripper travel is reversed/incomplete, measure the GELLO motor angle at open
and closed and set both `gripper_open_deg` and `gripper_closed_deg` in the leader
config. When omitted, the startup-open / 42-degree travel estimate follows the
UF ACTORY example. Restart to recalibrate the leader reference; the browser's VR
recalibration control does not recalibrate GELLO.

The desktop browser controls and camera preview remain available. No Quest
connection is needed for leader motion. Use the existing browser controls for
recording/reset. Reset returns the simulation to its captured initial pose;
subsequent leader following resumes with the configured target slew limit.

## Recording

Add a NEW dataset entry inside `profiles.xarm6` (alongside arms and cameras):

```yaml
dataset:
  repo_id: charlieK123/xarm6_robotiq85_sim
  single_task: picking up forceps and sorting the forceps
  push_to_hub: false
```

The existing recorder derives the seven-channel schema from the selected robot.
The D405 simulated depth packet uses the existing native RGB-D recording path.
Dataset recording/upload has not been exercised end-to-end for this new robot.

## Scope and checks

Implemented: model switching, bundled robot meshes, GELLO serial input and startup
calibration, six joint targets, coupled gripper, camera mounts, FK workspace guard,
arm target speed limits, invalid command rejection, and serial timeout hold.
The recorded action is the returned clamped/slew-limited target.

xArm6 is currently leader-only. Existing SO101 leader and VR modes remain supported.
The cameras, workspace bounds, servo gains, gripper force, and mount transform are
starting settings, not measurements of your hardware. There are no task objects
or table assets added to the existing ground-plane scene.

21 focused tests pass, including loading the real model into CPU PhysX, comparing
URDF FK with the simulated tool pose, and closing both gripper fingers through
physical linkage constraints. Fake SDK transport tests cover calibration, joint
signs, encoder wraparound, fresh reads, timeout, and cleanup. SO101 profile/schema
compatibility is covered. Python compilation and Git whitespace checks pass.

Not validated here: GPU rendering/viewer, actual GELLO USB hardware, a Docker build,
full existing LeRobot/WebXR/dataset test suite, or contact-driven adaptive grasping.
The gripper is an initial parallel-opening simulation, not a calibrated reproduction
of the hardware's adaptive grasp forces. Test pick-and-place contacts before
relying on generated grasp demonstrations.

To rerun the focused tests:

```bash
python -m pip install pytest
PYTHONPATH=src python -m pytest tests/test_xarm6.py -q
```

Asset sources and licenses are in `src/telerobot/simulation/XARM6/ASSET_SOURCE.md`.

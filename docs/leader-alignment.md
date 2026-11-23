# Match the simulated xArm6 to GELLO

Keep the normal startup reference-pose calibration, including the terminal Enter
prompt and open gripper. After startup, the desktop web dashboard displays
**Match sim to leader** for the GELLO + simulated xArm6 configuration only.

Click the button to read a fresh calibrated leader sample and immediately place
the simulated arm and gripper at that pose. Existing calibration offsets and the
captured reset pose are preserved. Joint velocities are cleared; drive targets
and the simulator's last-target cache are updated by `set_joint_state`, so normal
leader following resumes on the next cycle.

This explicit pose match is an instantaneous simulation reposition, not a smooth
servo move. Targets outside simulated joint limits or rejected by the configured
workspace guard are refused with a message below the button. Stop recording first;
the button is disabled during recording/countdown and the backend also refuses
requests while recording. No dataset configuration is needed.

The command uses a dedicated one-shot queue so later VR frames cannot overwrite
the click. All joint reads and simulation updates run on the existing control
thread. This feature does not add physical xArm control.

## Applying the patch

From the `telerobot` checkout on branch `teleopSim`:

```bash
git apply --check ~/Downloads/xarm6_match_leader_ui.patch
git apply ~/Downloads/xarm6_match_leader_ui.patch
```

Restart `telerobot`, open the desktop dashboard, and hard-refresh the page
(Ctrl+Shift+R). If the package is copied into a Docker image, rebuild that image
first so it contains the updated Python and web UI files.

Patch base: `af124d6a6f51c1c26ea551af1d0a522568ce7bdf` on `teleopSim`.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_leader_alignment.py -v
node tests/test_leader_alignment_ui.cjs
```

The focused tests use injected leader samples and mocked simulation objects, and
test WebSocket command handling and dashboard controls. A live SAPIEN/GELLO run
still needs to be checked on a machine with the simulator and leader hardware.

# xArm6 + Robotiq 2F-85 assets

Source: https://github.com/haosulab/ManiSkill-XArm6
Tag: v0.1.1
Commit: 438e7e4ecbdc0b21e2c2e24d7b19a731d98599bf

Bundled xarm6_robotiq.urdf, visual/, collision/, meshes_robotiq/, and LICENSE.
The model contains Robotiq ARG2F-85 meshes and the eef tool frame.
The original mesh and robot asset license is included as LICENSE.

Finger loop anchors and collision exclusions follow:
https://github.com/haosulab/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/agents/robots/xarm6/xarm6_robotiq.py

This adapter drives the two outer knuckles together, leaves inner joints passive,
and closes the four-bar loops with PhysX constraints. Adaptive contact behavior,
force calibration, and actual hardware equivalence are not validated.
Camera mounting transforms and workspace bounds are configurable starting points.

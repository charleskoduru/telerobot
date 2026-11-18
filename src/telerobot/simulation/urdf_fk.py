"""Small URDF forward-kinematics reader for the xArm leader workspace guard.

Uses the exact simulated model and degree-valued logical arm coordinates.
No IK/Placo dependency is needed for joint-space GELLO teleoperation.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin_transform(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        transform[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    return transform


class URDFForwardKinematics:
    def __init__(self, urdf_path: Path, target_frame="eef", joint_names=None):
        root = ET.parse(urdf_path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.chain = []
        link = target_frame
        if root.find(f"link[@name='{link}']") is None:
            raise ValueError(f"Missing tool frame {target_frame}")
        while link in by_child:
            joint = by_child[link]
            self.chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain.reverse()
        self.joint_names = list(joint_names or [f"joint{i}" for i in range(1, 7)])

    def forward_kinematics(self, joint_values):
        if len(joint_values) != len(self.joint_names) or not np.isfinite(joint_values).all():
            raise ValueError("FK expects one finite degree value per configured joint.")
        values = dict(zip(self.joint_names, joint_values))
        result = np.eye(4)
        for joint in self.chain:
            result = result @ origin_transform(joint.find("origin"))
            if joint.get("type") == "fixed":
                continue
            q = values[joint.get("name")]
            motion = np.eye(4)
            axis_element = joint.find("axis")
            axis = np.fromstring(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0", sep=" ")
            if joint.get("type") in ("revolute", "continuous"):
                motion[:3, :3] = rotation(axis, math.radians(q))
            elif joint.get("type") == "prismatic":
                motion[:3, 3] = axis * q
            result = result @ motion
        return result

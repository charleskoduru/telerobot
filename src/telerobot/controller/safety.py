"""Hard workspace safety checks applied to final joint commands.

The upstream EEBoundsAndSafety processor clamps the Cartesian *target* before IK.
That is useful, but it cannot protect leader-arm mode (which bypasses the VR IK
pipeline) and it cannot catch an IK result whose FK still lands outside the box.
This module adds a final, mode-independent FK check immediately before commands
are sent to the follower.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class WorkspaceSafetyGuard:
    """Reject final joint commands whose predicted EE pose leaves the safe box.

    The guard remembers the last *valid commanded joint pose*. If a later
    command would place the end effector outside the safe box, the arm is sent
    back to that latched pose instead of repeatedly copying the latest measured
    position. This creates a sticky boundary: the follower stays at the last
    valid pose until a new requested pose is inside the bounds.
    """

    def __init__(
        self,
        kinematics,
        motor_names: list[str],
        end_effector_bounds: dict[str, list[float]],
        enabled: bool = True,
        margin_m: float = 0.005,
    ) -> None:
        self.kinematics = kinematics
        self.motor_names = list(motor_names)
        self.enabled = bool(enabled)
        self.margin_m = max(float(margin_m), 0.0)
        self._last_warning_time = 0.0
        # Non-gripper joint targets from the most recent command whose FK was
        # inside the safe box. Keeping a commanded pose (rather than refreshing
        # from measured feedback every blocked cycle) prevents the boundary from
        # "ratcheting" along with gravity or mechanical sag.
        self._last_valid_arm_action: dict[str, float] | None = None

        raw_min = np.asarray(end_effector_bounds["min"], dtype=float)
        raw_max = np.asarray(end_effector_bounds["max"], dtype=float)
        if raw_min.shape != (3,) or raw_max.shape != (3,):
            raise ValueError("end_effector_bounds min/max must each contain exactly 3 values")
        if np.any(raw_min >= raw_max):
            raise ValueError("end_effector_bounds min must be strictly smaller than max")

        self.safe_min = raw_min + self.margin_m
        self.safe_max = raw_max - self.margin_m
        if np.any(self.safe_min >= self.safe_max):
            raise ValueError(
                f"workspace_guard_margin_m={self.margin_m} is too large for end_effector_bounds"
            )

    def _joint_vector(self, joints: dict) -> np.ndarray:
        values = []
        for name in self.motor_names:
            key = f"{name}.pos"
            if key not in joints:
                raise KeyError(f"Missing joint '{key}' required for workspace FK safety check")
            values.append(float(joints[key]))
        return np.asarray(values, dtype=float)

    def _position(self, joints: dict) -> np.ndarray:
        transform = self.kinematics.forward_kinematics(self._joint_vector(joints))
        return np.asarray(transform[:3, 3], dtype=float)

    def _violation(self, pos: np.ndarray) -> np.ndarray:
        below = np.maximum(self.safe_min - pos, 0.0)
        above = np.maximum(pos - self.safe_max, 0.0)
        return below + above

    def _arm_joint_action(self, joints: dict) -> dict[str, float]:
        """Extract non-gripper joint targets used to hold the arm rigidly."""
        arm_action: dict[str, float] = {}
        for name in self.motor_names:
            if name == "gripper":
                continue
            key = f"{name}.pos"
            if key in joints:
                arm_action[key] = float(joints[key])
        return arm_action

    def _remember_valid_action(self, requested_action: dict) -> None:
        arm_action = self._arm_joint_action(requested_action)
        if arm_action:
            self._last_valid_arm_action = arm_action

    def _hold_action(self, observation: dict, requested_action: dict) -> dict:
        """Hold the last valid arm pose while still allowing gripper commands.

        On the first blocked command there may not yet be a latched valid target.
        In that case capture the measured arm pose once, then keep using that same
        snapshot on subsequent blocked cycles. Do not refresh it from observation
        every cycle, otherwise gravity/sag can slowly walk the hold point outward.
        """
        if self._last_valid_arm_action is None:
            initial_hold = self._arm_joint_action(observation)
            if initial_hold:
                self._last_valid_arm_action = initial_hold

        safe_action = dict(requested_action)
        if self._last_valid_arm_action is not None:
            safe_action.update(self._last_valid_arm_action)
        return safe_action

    def _warn_blocked(self, current_pos: np.ndarray | None, target_pos: np.ndarray | None, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_time < 1.0:
            return
        self._last_warning_time = now

        logger.warning(
            "Workspace safety blocked command (%s). current=%s target=%s safe_min=%s safe_max=%s",
            reason,
            None if current_pos is None else np.round(current_pos, 4).tolist(),
            None if target_pos is None else np.round(target_pos, 4).tolist(),
            np.round(self.safe_min, 4).tolist(),
            np.round(self.safe_max, 4).tolist(),
        )

    def filter_action(self, observation: dict, requested_action: dict) -> dict:
        if not self.enabled:
            return requested_action

        try:
            current_pos = self._position(observation)
            target_pos = self._position(requested_action)
        except Exception as exc:
            # Safety checks should fail closed: if FK cannot be evaluated, hold the
            # arm rather than sending an unchecked command toward a wall/table.
            self._warn_blocked(None, None, f"FK check failed: {exc}")
            return self._hold_action(observation, requested_action)

        target_violation = float(np.linalg.norm(self._violation(target_pos)))

        # A valid target becomes the new sticky hold point. Once accepted, this
        # exact arm command is what we keep sending if later requests cross the
        # boundary.
        if target_violation <= 1e-6:
            self._remember_valid_action(requested_action)
            return requested_action

        # Never follow an out-of-bounds request, even if feedback has drifted.
        # Stay at the last valid commanded pose until the requested FK is back
        # inside the safe box.
        self._warn_blocked(current_pos, target_pos, "predicted EE pose outside bounds; holding last valid pose")
        return self._hold_action(observation, requested_action)

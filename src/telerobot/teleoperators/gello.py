"""GELLO xArm6 mapping using fresh, synchronous Dynamixel Protocol 2 reads.

Mapping follows xArm-Developer/ufactory_teleop. Unlike the upstream cached
background reader, each get_action call requires a successful bus transaction.
Arm outputs are degrees; gripper.pos is percent OPEN (0 closed, 100 open).
"""
from __future__ import annotations

import math
import numpy as np

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))


def validate_gello_config(cfg):
    ids, signs, start = cfg.joint_ids, cfg.joint_signs, cfg.start_joints_rad
    if len(ids) != 6 or len(signs) != 6 or len(start) != 6:
        raise ValueError("GELLO xarm6 needs six joint_ids, joint_signs, and start_joints_rad.")
    if any(s not in (-1, 1) for s in signs) or not np.isfinite(start).all():
        raise ValueError("GELLO signs must be +/-1 and start angles must be finite radians.")
    all_ids = list(ids) + ([] if cfg.gripper_id == -1 else [cfg.gripper_id])
    if len(set(all_ids)) != len(all_ids) or any(type(i) is not int or not 0 <= i <= 252 for i in all_ids):
        raise ValueError("GELLO motor IDs must be unique integers in 0..252; gripper_id can be -1.")
    if any(type(i) is not int or not 0 <= i <= 252 for i in cfg.torque_joint_ids):
        raise ValueError("torque_joint_ids must contain valid Dynamixel IDs.")
    if set(all_ids) & set(cfg.torque_joint_ids):
        raise ValueError("Locked/torqued joints must not overlap mapped joints or gripper.")
    if cfg.baudrate <= 0 or not cfg.use_degrees:
        raise ValueError("GELLO requires a positive baudrate and use_degrees: true.")
    endpoints = (cfg.gripper_open_deg, cfg.gripper_closed_deg)
    if (endpoints[0] is None) != (endpoints[1] is None):
        raise ValueError("Set both gripper_open_deg and gripper_closed_deg, or omit both.")
    if endpoints[0] is not None and (not np.isfinite(endpoints).all() or endpoints[0] == endpoints[1]):
        raise ValueError("GELLO gripper endpoints must be finite and different.")


class GelloLeader:
    def __init__(self, cfg):
        validate_gello_config(cfg)
        self.cfg = cfg
        self._port = self._packet = self._reader = None
        self._connected = False
        self._locked = []
        self._raw_previous = self._unwrapped = self._offsets = None

    @property
    def is_connected(self):
        return self._connected

    def _torque(self, motor_id, enabled):
        result, error = self._packet.write1ByteTxRx(self._port, motor_id, 64, int(enabled))
        if result != 0 or error:
            raise RuntimeError(f"Dynamixel {motor_id}: torque write failed ({result}, {error}).")

    def _read_raw(self):
        # GroupSyncRead performs a new bounded SDK transaction. Never reuse
        # previously cached values after missing packets or a device unplug.
        result = self._reader.txRxPacket()
        if result != 0:
            raise TimeoutError(f"GELLO serial read failed: {self._packet.getTxRxResult(result)}")
        values = []
        for motor_id in self._ids:
            if not self._reader.isAvailable(motor_id, 132, 4):
                raise TimeoutError(f"GELLO missing position for Dynamixel ID {motor_id}")
            raw = self._reader.getData(motor_id, 132, 4)
            if raw >= 2**31:
                raw -= 2**32
            values.append(raw * (2 * math.pi / 4096))
        return np.asarray(values, dtype=float)

    def connect(self, calibrate=True):
        del calibrate
        if self._connected:
            return
        from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead
        print("GELLO: place leader at start_joints_rad reference pose with gripper OPEN.")
        input("Press Enter to calibrate the leader and start simulation: ")
        try:
            self._port = PortHandler(self.cfg.port)
            self._packet = PacketHandler(2.0)
            if not self._port.openPort() or not self._port.setBaudRate(self.cfg.baudrate):
                raise RuntimeError(f"Cannot open GELLO port {self.cfg.port} at {self.cfg.baudrate} baud.")
            self._ids = list(self.cfg.joint_ids)
            if self.cfg.gripper_id >= 0:
                self._ids.append(self.cfg.gripper_id)
            for motor_id in self._ids:
                self._torque(motor_id, False)
            # Lock only explicitly selected unused motors at their present angle.
            for motor_id in self.cfg.torque_joint_ids:
                self._torque(motor_id, False)
                position, result, error = self._packet.read4ByteTxRx(self._port, motor_id, 132)
                if result != 0 or error:
                    raise RuntimeError(f"Cannot read unused GELLO joint {motor_id} before locking.")
                result, error = self._packet.write4ByteTxRx(self._port, motor_id, 116, position)
                if result != 0 or error:
                    raise RuntimeError(f"Cannot latch unused GELLO joint {motor_id}.")
                self._locked.append(motor_id)
                self._torque(motor_id, True)
            self._reader = GroupSyncRead(self._port, self._packet, 132, 4)
            for motor_id in self._ids:
                if not self._reader.addParam(motor_id):
                    raise RuntimeError(f"Cannot add GELLO joint {motor_id} to sync read.")
            self.calibrate()
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def calibrate(self):
        raw = self._read_raw()
        self._raw_previous = raw.copy()
        self._unwrapped = raw.copy()
        self._offsets = raw[:6] - np.asarray(self.cfg.start_joints_rad) / np.asarray(self.cfg.joint_signs)
        if self.cfg.gripper_id >= 0:
            # Upstream assumes startup=open and about 42 degrees of lever travel.
            # Explicit measured endpoints are preferred and override that estimate.
            self._open_rad = math.radians(self.cfg.gripper_open_deg) if self.cfg.gripper_open_deg is not None else raw[-1] - math.radians(0.2)
            self._closed_rad = math.radians(self.cfg.gripper_closed_deg) if self.cfg.gripper_closed_deg is not None else raw[-1] - math.radians(42)

    def get_action(self):
        if not self.is_connected:
            raise RuntimeError("GELLO is not connected.")
        return self.action_from_raw(self._read_raw())

    def action_from_raw(self, raw):
        """Map a fresh sample; unwrap encoder rollover without 360-degree jumps."""
        raw = np.asarray(raw, dtype=float)
        if raw.shape != self._raw_previous.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid GELLO joint sample.")
        delta = (raw - self._raw_previous + math.pi) % (2 * math.pi) - math.pi
        self._unwrapped += delta
        self._raw_previous = raw.copy()
        arm = (self._unwrapped[:6] - self._offsets) * self.cfg.joint_signs
        action = {f"{name}.pos": float(math.degrees(q)) for name, q in zip(ARM_JOINTS, arm)}
        if self.cfg.gripper_id >= 0:
            closed = np.clip((self._unwrapped[-1] - self._open_rad) / (self._closed_rad - self._open_rad), 0, 1)
            action["gripper.pos"] = float(100 * (1 - closed))
        else:
            action["gripper.pos"] = 100.0
        return action

    def disconnect(self):
        try:
            for motor_id in self._locked:
                try:
                    self._torque(motor_id, False)
                except Exception:
                    pass
        finally:
            self._locked.clear()
            if self._reader is not None:
                self._reader.clearParam()
            if self._port is not None:
                self._port.closePort()
            self._reader = self._packet = self._port = None
            self._connected = False

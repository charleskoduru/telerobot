#!/usr/bin/env python3
"""List RealSense serial numbers without connecting the robot."""
import pyrealsense2 as rs

cameras = rs.context().query_devices()
if len(cameras) == 0:
    raise SystemExit("No RealSense devices found. Check USB/device access inside your container.")
for device in cameras:
    print(f"{device.get_info(rs.camera_info.name)}  serial_number: '{device.get_info(rs.camera_info.serial_number)}'")

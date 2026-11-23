"""Run with: PYTHONPATH=src python -m unittest discover -s tests -p test_leader_alignment.py"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from telerobot.simulation.leader_alignment import align_simulation_to_leader
# The serial driver is irrelevant to these tests: encoder samples are injected.
serial_stub = ModuleType('dynamixel_sdk')
serial_stub.PortHandler = serial_stub.PacketHandler = serial_stub.GroupSyncRead = Mock()
with patch.dict(sys.modules, {'dynamixel_sdk': serial_stub}):
    from telerobot.teleoperators.gello import GelloLeader

ROOT = Path(__file__).resolve().parents[1]


def make_scene():
    action = {f'joint{i}.pos': float(i) for i in range(1, 7)}
    action['gripper.pos'] = 75.0
    robot = Mock()
    robot._clamp_action.side_effect = dict
    robot.articulation.dof = 12
    controller = Mock()
    controller.filter_joint_action.side_effect = lambda obs, requested: dict(requested)
    leader = Mock()
    leader.get_action.return_value = action
    return robot, leader, controller, action


class AlignmentTests(unittest.TestCase):
    def test_fresh_read_exact_pose_and_zero_velocity_without_recalibration(self):
        robot, leader, controller, action = make_scene()
        self.assertEqual(align_simulation_to_leader(robot, leader, controller), action)
        leader.get_action.assert_called_once_with()
        leader.calibrate.assert_not_called()
        robot.set_joint_state.assert_called_once_with(action)
        robot.send_action.assert_not_called()
        robot.articulation.set_qvel.assert_called_once_with([0.0] * 12)
        self.assertFalse(controller.has_initial_position)

    def test_recording_rejects_before_read_or_mutation(self):
        robot, leader, controller, _ = make_scene()
        with self.assertRaisesRegex(ValueError, 'Stop recording'):
            align_simulation_to_leader(robot, leader, controller, recording=True)
        leader.get_action.assert_not_called()
        robot.set_joint_state.assert_not_called()

    def test_workspace_and_joint_limits_reject_instead_of_partial_match(self):
        for block in ('joint', 'workspace'):
            with self.subTest(block=block):
                robot, leader, controller, action = make_scene()
                changed = dict(action, **{'joint1.pos': 0.0})
                if block == 'joint':
                    robot._clamp_action.side_effect = None
                    robot._clamp_action.return_value = changed
                else:
                    controller.filter_joint_action.side_effect = None
                    controller.filter_joint_action.return_value = changed
                with self.assertRaises(ValueError):
                    align_simulation_to_leader(robot, leader, controller)
                robot.set_joint_state.assert_not_called()
                robot.articulation.set_qvel.assert_not_called()

    def test_timeout_and_invalid_action_leave_pose_unchanged(self):
        for error in (TimeoutError('missing leader packet'), ValueError('invalid sample')):
            robot, leader, controller, _ = make_scene()
            if isinstance(error, TimeoutError):
                leader.get_action.side_effect = error
            else:
                robot._validate_action.side_effect = error
            with self.assertRaises(type(error)):
                align_simulation_to_leader(robot, leader, controller)
            robot.set_joint_state.assert_not_called()

    def test_real_gello_mapping_preserves_startup_offsets(self):
        robot, _, controller, _ = make_scene()
        cfg = SimpleNamespace(
            joint_ids=[1,2,4,5,6,7], joint_signs=[1,1,-1,1,1,1],
            start_joints_rad=[0,0,-np.pi/2,0,np.pi/2,0], gripper_id=-1,
            torque_joint_ids=[3], baudrate=57600, use_degrees=True,
            gripper_open_deg=None, gripper_closed_deg=None,
        )
        leader = GelloLeader(cfg)
        leader._read_raw = Mock(return_value=np.ones(6))
        leader.calibrate()
        offsets = leader._offsets.copy()
        leader._connected = True
        leader._read_raw.return_value = np.ones(6) + 0.1
        result = align_simulation_to_leader(robot, leader, controller)
        np.testing.assert_array_equal(leader._offsets, offsets)
        self.assertAlmostEqual(result['joint1.pos'], np.degrees(0.1))
        self.assertAlmostEqual(result['joint3.pos'], -90 - np.degrees(0.1))
        self.assertEqual(leader._read_raw.call_count, 2)  # startup plus click


class WebSocketCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_click_survives_later_vr_frames_and_is_consumed_once(self):
        # Load this module directly: server/__init__ imports optional WebRTC libs.
        spec = importlib.util.spec_from_file_location(
            'alignment_test_websocket', ROOT/'src/telerobot/server/websocket_server.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        server = module.WebSocketServer(use_ssl=False)

        class Client:
            def __aiter__(self):
                async def messages():
                    yield json.dumps({'command': 'align_simulation_to_leader'})
                    yield json.dumps({'command': 'align_simulation_to_leader'})
                    yield json.dumps({'action': 'none', 'right': None})
                return messages()

        await server.on_observation_received(Client())
        self.assertEqual(server.last_observation['action'], 'none')
        self.assertTrue(server.consume_alignment_request())
        self.assertFalse(server.consume_alignment_request())
        server._schedule_broadcast = Mock()
        server.send_alignment_result(False, 'outside workspace')
        server._schedule_broadcast.assert_called_once_with({
            'type': 'leader_alignment_result', 'ok': False, 'message': 'outside workspace'})


if __name__ == '__main__':
    unittest.main()

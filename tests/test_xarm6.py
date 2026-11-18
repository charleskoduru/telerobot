"""Config, hardware protocol, and real CPU PhysX checks; no GPU/leader needed."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from telerobot.config import load_config, LeaderConfig, build_leader_teleoperator
from telerobot.simulation.xarm6_robot import SimulatedXArm6, ARM_JOINTS, OUTER_JOINTS
from telerobot.simulation.so101_robot import SimulatedSO101, JOINT_NAMES
from telerobot.simulation.urdf_fk import URDFForwardKinematics
from telerobot.teleoperators.gello import GelloLeader, validate_gello_config

PROJECT = Path(__file__).resolve().parents[1]


def config():
    return load_config(PROJECT / 'examples/config/xarm6-gello.yaml')


def test_profile_switch_preserves_so101():
    raw = yaml.safe_load((PROJECT / 'config.yaml').read_text())
    assert load_config(PROJECT / 'config.yaml').arms['right'].type == 'xarm6'
    raw['profile'] = 'so101'
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'config.yaml'
        path.write_text(yaml.safe_dump(raw))
        cfg = load_config(path)
    assert cfg.arms['right'].type == 'so100'
    robot = SimulatedSO101(cfg, 'right')
    assert set(robot.action_features) == {f'{n}.pos' for n in JOINT_NAMES}
    assert cfg.teleoperation.simulation.joint_offsets_deg == {'wrist_roll': -90.0}


@pytest.mark.parametrize('changes', [
    {'joint_ids':[1,2]}, {'joint_signs':[0]*6}, {'joint_ids':[1]*6},
    {'start_joints_rad':[math.nan]*6}, {'gripper_id':1}, {'torque_joint_ids':[1]},
    {'gripper_open_deg':2}, {'gripper_open_deg':1,'gripper_closed_deg':1}, {'baudrate':0},
])
def test_invalid_leader_config(changes):
    with pytest.raises(ValueError):
        validate_gello_config(replace(config().teleoperation.leader, **changes))


class Port:
    def __init__(self, port): self.closed = False
    def openPort(self): return True
    def setBaudRate(self, baud): return True
    def closePort(self): self.closed = True


class Packet:
    def __init__(self, protocol): pass
    def write1ByteTxRx(self, *args): return 0, 0
    def read4ByteTxRx(self, *args): return 2048, 0, 0
    def write4ByteTxRx(self, *args): return 0, 0
    def getTxRxResult(self, result): return 'test disconnected'


class Reader:
    def __init__(self, *args):
        self.positions = {i:2048 for i in [1,2,4,5,6,7,8]}
        self.failed = False
        self.read_count = 0
    def addParam(self, motor_id): return True
    def txRxPacket(self): self.read_count += 1; return -1 if self.failed else 0
    def isAvailable(self, *args): return True
    def getData(self, motor_id, *args): return self.positions[motor_id]
    def clearParam(self): pass


def test_real_adapter_calibration_sign_gripper_timeout_and_cleanup():
    module = ModuleType('dynamixel_sdk')
    module.PortHandler, module.PacketHandler, module.GroupSyncRead = Port, Packet, Reader
    cfg = config()
    leader = build_leader_teleoperator(cfg)
    with patch.dict(sys.modules, {'dynamixel_sdk':module}), patch('builtins.input',return_value=''):
        leader.connect()
    initial = leader.get_action()
    assert np.allclose([initial[f'{n}.pos'] for n in ARM_JOINTS], np.degrees(cfg.teleoperation.leader.start_joints_rad))
    assert initial['gripper.pos'] > 99
    leader._reader.positions[4] += 1024  # Third mapped arm joint has reversed sign.
    leader._reader.positions[8] -= 478  # Approximately 42 degrees of lever travel.
    action = leader.get_action()
    assert action['joint3.pos'] == pytest.approx(initial['joint3.pos'] - 90)
    assert action['gripper.pos'] < 1
    assert leader._reader.read_count == 3
    leader._reader.failed = True
    with pytest.raises(TimeoutError): leader.get_action()
    port = leader._port
    leader.disconnect()
    assert port.closed and not leader.is_connected
    leader.disconnect()


def physics_robot():
    import sapien
    cfg = config()
    cfg.cameras = {}
    cfg.arms['right'].cameras = []
    robot = SimulatedXArm6(cfg,'right')
    source = robot._urdf_path()
    tree = ET.parse(source)
    # Omit rendering only: keep all real collision meshes, inertials and joints.
    for link in tree.findall('link'):
        for visual in link.findall('visual'): link.remove(visual)
    for mesh in tree.findall('.//mesh'):
        mesh.set('filename',str(source.parent / mesh.get('filename')))
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'physics.urdf'
        tree.write(path)
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(1/240)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        articulation = loader.load(str(path))
    robot.scene, robot.articulation = scene, articulation
    robot._joints = {j.name:j for j in articulation.get_active_joints()}
    for joint in robot._joints.values():
        joint.set_drive_property(stiffness=1000,damping=100,force_limit=100)
    robot._configure_articulation(sapien)
    robot._connected = True
    return robot


def initial_action():
    return dict(zip([f'{n}.pos' for n in ARM_JOINTS], [0,0,-90,0,90,0]), **{'gripper.pos':100})


def test_model_mesh_references_and_camera_links():
    robot=SimulatedXArm6(config(),'right')
    tree=ET.parse(robot._urdf_path())
    for mesh in tree.findall('.//mesh'):
        assert (robot._urdf_path().parent / mesh.get('filename')).is_file()
    links={link.get('name') for link in tree.findall('link')}
    for cam in robot.cfg.cameras.values():
        assert cam.simulation['attach_to'] == 'world' or cam.simulation['attach_to'] in links


def test_cpu_physx_fk_joint_mapping_finger_loops_and_rate_limit():
    robot=physics_robot()
    action=initial_action()
    # CPU-only scene has no renderer. Skip the render call at pose initialization.
    with patch.object(robot.scene,'update_render'):
        robot.set_joint_state(action)
    assert robot.articulation.dof == 12
    assert len(robot.action_features)==7
    fk=URDFForwardKinematics(robot._urdf_path(),joint_names=ARM_JOINTS)
    expected=fk.forward_kinematics([action[f'{n}.pos'] for n in ARM_JOINTS])
    eef=next(link for link in robot.articulation.get_links() if link.name=='eef')
    np.testing.assert_allclose(eef.pose.to_transformation_matrix(),expected,atol=1e-6)
    requested=dict(action, **{'joint1.pos':120,'gripper.pos':0})
    applied=robot.send_action(requested)
    assert applied['joint1.pos']==2  # 60 degrees/sec at 30 Hz
    assert applied['gripper.pos']==0
    # Hold the arm and close fingers for two seconds of real PhysX.
    for _ in range(480):
        robot.articulation.set_qf(robot.articulation.compute_passive_force())
        robot.scene.step()
    q=dict(zip([j.name for j in robot.articulation.get_active_joints()],robot.articulation.get_qpos()))
    assert np.isfinite(list(q.values())).all()
    assert q[OUTER_JOINTS[0]] > 0.5 and q[OUTER_JOINTS[1]] > 0.5
    assert abs(q[OUTER_JOINTS[0]] - q[OUTER_JOINTS[1]]) < 0.02
    assert set(robot.get_observation())==set(robot.action_features)
    for invalid in ({},dict(action, **{'joint2.pos':math.nan})):
        with pytest.raises(ValueError): robot.send_action(invalid)


def test_simulation_factory_without_physical_follower_imports():
    from telerobot.config import load_robot
    robot,cfg=load_robot(PROJECT/'examples/config/xarm6-gello.yaml')
    assert isinstance(robot,SimulatedXArm6)
    assert not robot.is_connected
    robot,cfg=load_robot(PROJECT/'examples/config/so101-simulation.yaml')
    assert type(robot) is SimulatedSO101


def test_rollover_without_joint_jump():
    cfg=replace(config().teleoperation.leader,gripper_id=-1)
    leader=GelloLeader(cfg)
    raw=np.full(6,2*math.pi-0.01)
    leader._raw_previous=raw.copy();leader._unwrapped=raw.copy()
    leader._offsets=raw.copy()
    action=leader.action_from_raw(np.full(6,0.01))
    assert action['joint1.pos']==pytest.approx(math.degrees(0.02))
    assert action['joint3.pos']==pytest.approx(-math.degrees(0.02))
    assert action['gripper.pos']==100


def test_guard_holds_last_valid_xarm_pose():
    import importlib.util
    spec=importlib.util.spec_from_file_location('xarm_test_safety',PROJECT/'src/telerobot/controller/safety.py')
    safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)
    robot=SimulatedXArm6(config(),'right')
    fk=URDFForwardKinematics(robot._urdf_path(),joint_names=ARM_JOINTS)
    action=initial_action()
    point=fk.forward_kinematics([action[f'{n}.pos'] for n in ARM_JOINTS])[:3,3]
    guard=safety.WorkspaceSafetyGuard(fk,list(ARM_JOINTS),{'min':point-0.01,'max':point+0.01},margin_m=0)
    assert guard.filter_action(action,action)==action
    outside=dict(action, **{'joint1.pos':90,'gripper.pos':0})
    held=guard.filter_action(action,outside)
    assert held['joint1.pos']==action['joint1.pos']
    assert held['gripper.pos']==0


@pytest.mark.parametrize('mutate', [
    lambda raw: raw['teleoperation']['simulation'].update(control_source='vr'),
    lambda raw: raw['teleoperation']['simulation'].update(joint_offsets_deg={'wrist_roll':-90}),
    lambda raw: raw['robot'].update(fps=0),
    lambda raw: raw['teleoperation']['simulation'].update(max_joint_speed_deg_s=math.nan),
    lambda raw: raw['teleoperation'].update(mode='leader'),
])
def test_reject_incompatible_xarm_configs(mutate):
    raw=yaml.safe_load((PROJECT/'examples/config/xarm6-gello.yaml').read_text());mutate(raw)
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'bad.yaml';path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError): load_config(path)

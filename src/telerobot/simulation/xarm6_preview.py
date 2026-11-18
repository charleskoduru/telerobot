"""Open xArm6 + Robotiq cameras/viewer without a connected leader arm."""
import argparse
import math
import time
from telerobot.config import load_config
from telerobot.simulation.xarm6_robot import SimulatedXArm6, ARM_JOINTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples/config/xarm6-gello.yaml')
    parser.add_argument('--animate', action='store_true', help='Slowly move joint1 and open/close fingers')
    args = parser.parse_args()
    cfg = load_config(args.config)
    arm_name = next(iter(cfg.arms))
    if cfg.arms[arm_name].type != 'xarm6':
        parser.error('Preview requires type: xarm6')
    robot = SimulatedXArm6(cfg, arm_name)
    start = {f'{n}.pos': math.degrees(q) for n, q in zip(ARM_JOINTS, cfg.teleoperation.leader.start_joints_rad)}
    start['gripper.pos'] = 100.0
    try:
        robot.connect()
        robot.set_joint_state(start)
        t0 = time.monotonic()
        while True:
            action = dict(start)
            if args.animate:
                t = time.monotonic() - t0
                action['joint1.pos'] += 15 * math.sin(t / 3)
                action['gripper.pos'] = 50 + 50 * math.cos(t / 2)
            robot.send_action(action)
            robot.step()
            robot.get_observation()  # Exercise configured RGB-D camera captures.
            time.sleep(1 / cfg.fps)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()

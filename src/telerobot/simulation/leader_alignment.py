"""Explicit simulation pose matching; never recalibrates or commands hardware."""
import math


def align_simulation_to_leader(robot, leader, controller, *, recording=False):
    """Use a fresh calibrated sample and reject targets that cannot match exactly."""
    if recording:
        raise ValueError("Stop recording before matching the simulation to the leader.")
    action = leader.get_action()
    robot._validate_action(action)
    bounded = robot._clamp_action(action)
    if any(not math.isclose(action[key], bounded[key], abs_tol=1e-6, rel_tol=0)
           for key in action):
        raise ValueError("Leader pose exceeds simulated joint limits; move it inside the limits.")
    observation = robot.get_observation()
    checked = controller.filter_joint_action(observation, action)
    if any(not math.isclose(action[key], checked[key], abs_tol=1e-6, rel_tol=0)
           for key in action):
        raise ValueError("Leader pose was rejected by the workspace guard.")
    # This is a deliberate simulation teleport, not a rate-limited servo move.
    robot.set_joint_state(action)
    robot.articulation.set_qvel([0.0] * robot.articulation.dof)
    controller.has_initial_position = False
    return dict(action)

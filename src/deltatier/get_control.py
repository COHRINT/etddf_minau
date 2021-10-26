from normalize_angle import normalize_angle
import numpy as np
def get_control(x_gt, waypoint, agent, STATES):
    TARGET_V = 0.2
    MAX_THETA_DOT = 0.1
    DELTA_CHANGE = 1.0

    state = x_gt[STATES*agent:STATES*(agent+1), 0]
    theta = state[2]
    theta_dot = state[5]
    speed = state[3]

    delta = waypoint - state[:2]

    target_angle = np.arctan2(delta[1], delta[0])
    delta_angle = normalize_angle( target_angle -  theta)

    if abs(delta_angle) > MAX_THETA_DOT:
        target_theta_dot = MAX_THETA_DOT * np.sign(delta_angle)
    else:
        target_theta_dot = delta_angle

    accel_theta_dot = target_theta_dot - theta_dot
    accel_v = TARGET_V - speed

    return DELTA_CHANGE * np.array([accel_v, accel_theta_dot])
from normalize_angle import normalize_angle
import numpy as np
def get_control(x_gt, waypoint, agent, STATES):

    MAX_VEL = 0.2

    state = x_gt[STATES*agent:STATES*(agent+1), 0]
    pos = state[:2]
    delta = waypoint - pos
    target_vel = MAX_VEL * ( delta / np.linalg.norm(delta) )

    return target_vel
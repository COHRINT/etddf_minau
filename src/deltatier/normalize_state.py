import numpy as np
from normalize_angle import normalize_angle

def normalize_state(x_gt, NUM_AGENTS, STATES):
    for a in range(NUM_AGENTS):
        x_gt[STATES*a+2, 0] = normalize_angle(x_gt[STATES*a+2, 0])
    return x_gt
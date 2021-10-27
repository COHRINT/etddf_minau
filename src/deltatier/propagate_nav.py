import numpy as np
from normalize_angle import normalize_angle
import sys

def propagate_nav(body, body_Q):

    F = np.eye(body.shape[0])
    F[0,3] = 1
    F[1,4] = 1
    F[2,5] = 1
    F[3:,3:] = 0
    body = np.reshape(body, (body.shape[0],1))

    body_new = F.dot(body)

    body_new_Q = np.dot( F.dot(body_Q), F.T)
    return [body_new, body_new_Q]
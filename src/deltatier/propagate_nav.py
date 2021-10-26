import numpy as np
from normalize_angle import normalize_angle

def propagate_nav(body, body_Q):

    """ TODO add runge-kutta integration"""

    x = body[0]
    y = body[1]
    theta = body[2]
    v = body[3]
    vp = body[4]
    theta_dot = body[5]

    x_new = x + v * np.cos(theta)
    y_new = y + v * np.sin(theta)
    theta_new = normalize_angle(theta + theta_dot)

    body_new = np.array([[x_new, y_new, theta_new, v, vp, theta_dot]]).T

    dxdtheta = -v * np.sin(theta)
    dxdv = np.cos(theta)
    dydtheta = v * np.cos(theta)
    dydv = np.sin(theta)

    F = np.eye(body.shape[0])
    F[0,2] = dxdtheta
    F[0,3] = dxdv
    F[1,2] = dydtheta
    F[1,3] = dydv

    body_new_Q = np.dot( np.dot(F, body_Q), F.T )

    return [body_new, body_new_Q]

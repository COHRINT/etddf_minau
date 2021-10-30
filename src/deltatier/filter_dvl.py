import numpy as np
from numpy.linalg import inv
from normalize_state import normalize_state

def filter_dvl(x_hat, P, x_gt, w, w_perceived, NUM_AGENTS, STATES, agent):

    start_row = agent*STATES + 3;
    end_row = start_row + 1;
    dvl = x_gt[start_row:end_row, 0] + np.random.normal(0, w, (2,1))

    H = np.zeros((2, STATES));
    H[0,3] = 1;
    H[1,4] = 1;

    K = P @ H.T @ inv( H @ P @ H.T + w_perceived * np.eye(2) )
    x_hat = x_hat + K @ (dvl - H @ x_hat)
    x_hat = normalize_state(x_hat, 1, STATES)
    P = P - K @ H @ P

    return x_hat, P

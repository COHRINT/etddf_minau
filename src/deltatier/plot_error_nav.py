import matplotlib.pyplot as plt
import numpy as np

def plot_error_nav(x_navs_history, P_navs_history, x_gt_history, STATES, agent):

    NUM_LOOPS = np.shape(x_gt_history)[1]

    start_row = STATES*agent
    end_row = start_row + STATES
    x_gt_history_agent = x_gt_history[start_row:end_row, :]

    x_nav_history = x_navs_history[start_row:end_row, :]
    P_nav_history = P_navs_history[start_row:end_row, :]

    print(np.shape(x_nav_history))
    print(np.shape(x_gt_history_agent))
    error = x_gt_history_agent - x_nav_history

    fig = plt.figure()
    LABELS = ["x", "y", "z", "theta", "x_dot", "y_dot", "z", "theta_dot"]

    for s in range(STATES):
        label = LABELS[s]
        plt.subplot(1, STATES, s+1)

        state_error = error[s, :]
        indices = np.arange(s, NUM_LOOPS*STATES, STATES)
        sigma = 2*np.sqrt( P_nav_history[s, indices] )
        sigma_neg = -sigma
        plt.plot(state_error, color="r")
        plt.plot(sigma, color="g")
        plt.plot(sigma_neg, color="g")
        plt.title( label + " Error" )

    plt.show()


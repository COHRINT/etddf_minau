import matplotlib.pyplot as plt
import numpy as np

def plot_error_track(x_gt_history, x_hat_history_lst, P_history_lst, STATES, BLUE_NUM, RED_NUM):
    # x_gt_history is numpy array
    # x_hat_history and P_history are python lists of numpy arrays

    # Get error
    # Loop through states & plot with uncertainty

    NUM_AGENTS = BLUE_NUM + RED_NUM
    fig, axs = plt.subplots(NUM_AGENTS, 6) #x,y,z,vels

    num_loops = x_gt_history.shape[1]

    for a in range(NUM_AGENTS):
        agent_type = "Blue"
        if a >= BLUE_NUM:
            agent_type = "Red"

        state_list = ["x", "y", "z", "x_vel", "y_vel", "z_vel"]
        for s in range(6):
            state = state_list[s]

            state_error = []
            uncertainty = []
            for i in range(num_loops):
                if s < 3:
                    truth_state = x_gt_history[STATES*a + s, i]
                else: # Skip the orientation state in the truth
                    truth_state = x_gt_history[STATES*a + s + 1, i]
                est_state = x_hat_history_lst[i][6*a + s, 0]
                unc_state = P_history_lst[i][6*a + s, 6*a + s]
                two_sigma = 2*np.sqrt(unc_state)

                state_error.append(truth_state - est_state)
                uncertainty.append(two_sigma)

            axs[a, s].plot(state_error, c="r")
            axs[a,s].plot(uncertainty, c="g")
            axs[a,s].plot([-x for x in uncertainty], c="g")
            axs[a,s].set_title(agent_type + " " + str(a) + ": " + state)

    plt.show()
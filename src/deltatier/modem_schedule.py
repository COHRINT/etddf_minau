import numpy as np
from numpy.linalg import norm

"""
This file manages the modem scheduler

Start with just taking a global modem measurement of all agents on every time cycle
"""

def modem_schedule(loop_num, kfs, x_gt, agent, STATES, BLUE_NUM, modem_location, w, w_perceived_modem_range, w_perceived_modem_azimuth, delta_range):

    ping_delay = 3
    broadcast_delay = 4

    num_agents = x_gt.shape[0] / STATES

    ping_time = ping_delay*BLUE_NUM + broadcast_delay
    agent_share_times = []
    for b in range(BLUE_NUM):
        agent_share_times.append( ping_time + broadcast_delay*(b+1) )
    total_time = agent_share_times[-1] + 1

    current_iter = np.mod(loop_num, total_time)

    # TODO change
    # current_iter = ping_time
    kf = kfs[agent]

    # Surface broadcasts positions
    if current_iter == ping_time:

        

        for b in range(BLUE_NUM):

            # Generate measurement
            agent_states_gt = x_gt[STATES*b : STATES*(b+1),0]
            agent_position_gt = agent_states_gt[:3]

            modem_location = np.array(modem_location)
            delta = agent_position_gt - modem_location
            truth_range = norm(delta)
            range_meas = truth_range + np.random.normal(0.0, w)
            truth_azimuth = np.arctan2(delta[1], delta[0])
            azimuth_meas = truth_azimuth + np.random.normal(0.0, w)

            kf.filter_range_from_untracked(range_meas, w_perceived_modem_range, modem_location, b)
            kf.filter_azimuth_from_untracked(azimuth_meas, w_perceived_modem_azimuth, modem_location, b)
        
    else: # Check if an agent is sharing
        pass
        # This agent is sharing
        # if agent_share_times[agent] == current_iter:
        #     buffer, delta = kf.pull_buffer(delta_range)

            # TODO rx_buffer

    pass
import numpy as np
from numpy.linalg import norm

MISSASSOCATION_COUNT = 0
PROTOTRACK_MISSASSOCIATION = 0

def take_error_sonar_meas(kf, associator, x_gt, x_nav, agent, w, w_perceived_range, w_perceieved_azimuth, sonar_range, prob_det, STATES, loop_num, landmark_pos=[]):
    """
    Randomly generate sonar measurements around the agent and make sure proto is always returned
    """
    global MISSASSOCATION_COUNT, PROTOTRACK_MISSASSOCIATION
    num_agents = int( x_gt.shape[0] / STATES )
    false_detection_rate = 0.1
    if np.random.binomial(1, false_detection_rate):
        rel_range_meas = np.random.uniform(0, 20)
        rel_azimuth_meas = np.random.uniform(-np.pi, np.pi)
        ### Associator Node ###

        # Create agent dict
        x_hat = np.copy(kf.x_hat)
        P = np.copy(kf.P)
        agent_dict = {}
        for b in range(num_agents):
            if b == agent: # Don't need our own position
                continue
            mean = np.reshape( x_hat[6*b:6*b+2,0], (-1,1) )
            cov = P[6*b:6*b+2, 6*b:6*b+2]
            agent_dict[b] = [mean, cov]
            
        meas = np.array([
            [rel_range_meas * np.cos(rel_azimuth_meas)],
            [rel_range_meas * np.sin(rel_azimuth_meas)]
        ])
        taker_position = np.reshape( x_hat[6*agent:6*agent+2,0], (-1,1) )
        global_meas = meas + taker_position
        R = np.eye(2) # Not used rn...
        associated_agent, proto = associator.associate(agent_dict, global_meas, R, loop_num)

        if associated_agent != "proto":
            MISSASSOCATION_COUNT += 1
            if proto:
                PROTOTRACK_MISSASSOCIATION += 1
            print("{} {}".format(MISSASSOCATION_COUNT, PROTOTRACK_MISSASSOCIATION))

        if associated_agent != "proto":
            kf.filter_range_tracked(rel_range_meas, w_perceived_range, agent, associated_agent)
            kf.filter_azimuth_tracked(rel_azimuth_meas, w_perceieved_azimuth, agent, associated_agent)

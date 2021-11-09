import numpy as np
from numpy.linalg import norm

def take_sonar_meas(kf, associator, x_gt, x_nav, agent, w, w_perceived_range, w_perceieved_azimuth, sonar_range, prob_det, STATES, loop_num, landmark_pos=[]):
    """
    landmark_pos : list of list of landmark positions

    Generates and fuses sonar measurements to the kalman filter

    First assumes 360 detection model
    Then take in scan angle
    """

    num_agents = int( x_gt.shape[0] / STATES )
    agent_states = x_gt[STATES*agent : STATES*(agent+1),0]
    agent_position = agent_states[:3]
    agent_theta_gt = agent_states[3]
    agent_theta_est = x_nav[3,0]

    for a in range(num_agents):
        if a == agent:
            continue

        a_states = x_gt[STATES*a : STATES*(a+1),0]
        a_pos = a_states[:3]

        delta = a_pos - agent_position

        # We've got a detection from detection algorithm
        if norm(delta) < sonar_range and np.random.binomial(1, prob_det):
            
            rel_range_meas = norm(delta) + np.random.normal(0.0, w)
            x_delta = delta[0]
            y_delta = delta[1]
            rel_azimuth_meas = np.arctan2(y_delta, x_delta) - agent_theta_gt + np.random.normal(0.0, w)

            rel_azimuth_meas = rel_azimuth_meas + agent_theta_est

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
            associated_agent = associator.associate(agent_dict, global_meas, R, loop_num)

            if associated_agent != "proto":
                assert associated_agent == a

            if associated_agent != "proto":
                kf.filter_range_tracked(rel_range_meas, w_perceived_range, agent, associated_agent)
                kf.filter_azimuth_tracked(rel_azimuth_meas, w_perceieved_azimuth, agent, associated_agent)

            ### SONAR CONTROL NODE ###


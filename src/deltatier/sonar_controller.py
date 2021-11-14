import numpy as np
from deltatier.normalize_angle import normalize_angle
from copy import deepcopy

def scan_agent(mean, my_pos, scan_size):
    delta = mean[:2] - np.reshape(my_pos[:2], (-1,1))
    x_delta = delta[0]
    y_delta = delta[1]
    world_angle = np.arctan2(y_delta, x_delta)
    return normalize_angle( world_angle - scan_size / 2.0 )

def scan_control(scan_angle, my_pos, agent_dict, prototrack, scan_size, ping_thresh, lost_thresh):
    """
    scan_angle in inertial frame
    Returns:
    - float: the start scan region in inertial frame
    - bool: whether scanning 360
    """
    # Determine if an agent needs pinging
    lost_agent = False
    for agent in agent_dict:
        mean = agent_dict[agent][0]
        cov = agent_dict[agent][1]
        if np.trace(cov) > lost_thresh:
            lost_agent = True
        elif np.trace(cov) > ping_thresh: # Agent is lost and attempt to scan
            print("Pinging: {}".format(agent))
            return scan_agent(mean, my_pos, scan_size), False

    # Check if we have a lost agent, scan 360 deg
    if lost_agent:

        if prototrack is not None: # If we have any prototracks scan there first
            print("Prototrack started, rescanning")
            mean = prototrack[0]
            return scan_agent(mean, my_pos, scan_size), False
        else:
            print("Agent lost: scanning")
            return normalize_angle( scan_angle - scan_size ), True # Subtract b/c ping360 scans down
    else:
        # ping agent with highest uncertainty
        agents, unc = [], []
        for agent in agent_dict:
            cov = agent_dict[agent][1]
            agents.append(agent)
            unc.append(np.trace(cov))

        max_index = np.argmax(unc)
        agent = agents[max_index]
        print("Pinging: {}".format(agent))
        mean = agent_dict[agent][0]
        return scan_agent(mean, my_pos, scan_size), False
import numpy as np
from normalize_angle import normalize_angle
from copy import deepcopy

def scan_agent(mean, my_pos, scan_size):
    delta = mean - np.reshape(my_pos[:2], (-1,1))
    x_delta = delta[0]
    y_delta = delta[1]
    world_angle = np.arctan2(y_delta, x_delta)
    return normalize_angle( world_angle - scan_size / 2.0 )

def scan_control(scan_angle, my_pos, agent_dict, prototrack, scan_size, ping_thresh, lost_thresh):
    """
    Returns the start scan region
    """
    # Determine if an agent needs pinging
    lost_agent = False
    for agent in agent_dict:
        mean = agent_dict[agent][0]
        cov = agent_dict[agent][1]
        if np.trace(cov) > lost_thresh:
            lost_agent = True
        elif np.trace(cov) > ping_thresh: # Agent is lost and attempt to scan
            return scan_agent(mean, my_pos, scan_size)

    # Check if we have a lost agent, scan 360 deg
    if lost_agent:

        if prototrack is not None: # If we have any prototracks scan there first
            print("Prototrack started, rescanning")
            mean = prototrack[0]
            return scan_agent(mean, my_pos, scan_size)
        else:
            print("Agent lost: scanning")
            return normalize_angle( scan_angle + scan_size )
    else:
        # ping agent with highest uncertainty
        agents, unc = [], []
        for agent in agent_dict:
            cov = agent_dict[agent][1]
            agents.append(agent)
            unc.append(np.trace(cov))

        max_index = np.argmax(unc)
        agent = agents[max_index]
        mean = agent_dict[agent][0]
        return scan_agent(mean, my_pos, scan_size)
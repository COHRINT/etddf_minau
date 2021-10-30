def get_estimate_nav(x_navs, P_navs, agent, STATES):
    x_nav = x_navs[:, agent]
    P_nav = P_navs[:, agent*STATES:(agent+1)*STATES]
    return x_nav.reshape(-1,1), P_nav
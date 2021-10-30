def set_estimate_nav(x_nav, P_nav, x_navs, P_navs, agent, STATES):
    x_navs[:, agent] = x_nav.reshape(-1)
    P_navs[:, agent*STATES:(agent+1)*STATES] = P_nav
    return x_navs, P_navs
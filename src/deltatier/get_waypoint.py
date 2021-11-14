import numpy as np

def get_waypoint(MAP_DIM):
    new_x = MAP_DIM*np.random.uniform() - MAP_DIM / 2.0
    new_y = MAP_DIM*np.random.uniform() - MAP_DIM / 2.0
    return np.array([new_x, new_y])

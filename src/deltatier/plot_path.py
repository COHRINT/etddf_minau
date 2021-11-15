import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 4,3
from matplotlib.animation import FuncAnimation

def plot_path(x_gt_history, MAP_DIM, NUM_LOOPS, waypoints):
    
    waypoint = waypoints[:,0]

    # create a figure with an axes
    fig, ax = plt.subplots()
    # set the axes limits
    ax.axis([-MAP_DIM/2.0,MAP_DIM/2.0,-MAP_DIM/2.0,MAP_DIM/2.0])
    # set equal aspect such that the circle is not shown as ellipse
    ax.set_aspect("equal")
    # create a point in the axes
    print(waypoint)
    point, = ax.plot(0,1, marker="o")

    def update(index):
        # Create the transformation
        x = x_gt_history[0,index]
        y = x_gt_history[1,index]
        point.set_data([x],[y])
        return point, 

    # create animation with 10ms interval, which is repeated,
    # provide the full circle (0,2pi) as parameters
    ani = FuncAnimation(fig, update, interval=10, blit=True, repeat=True,
                        frames=range(NUM_LOOPS))

    plt.show()
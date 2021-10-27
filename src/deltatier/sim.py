#!/usr/bin/env python
from plot_path import plot_path
from propagate_nav import propagate_nav
from normalize_state import normalize_state
from get_control import get_control
from get_waypoint import get_waypoint
from normalize_angle import normalize_angle
import numpy as np
import sys

"""
My goal for this simulator is to
- Verify in a simple setting correct implementation of DT
"""

# np.random.seed(0)

# Simulation

BLUE_NUM = 2;
RED_NUM = 0;
NUM_AGENTS = BLUE_NUM + RED_NUM;
STATES = 6; # Each agent has x,y,theta, x_vel,y_vel, theta_vel
TRACK_STATES = 4 * NUM_AGENTS; # x,y,x_dot, y_dot for each agent
TOTAL_STATES = STATES * NUM_AGENTS; 
TOTAL_TRACK_STATES = TRACK_STATES * BLUE_NUM;
NUM_LOOPS = 200;
MAP_DIM = 20; # Square with side length
PROB_DETECTION = 0.8;
SONAR_RANGE = 10.0;
MODEM_LOCATION = np.array([[11,11]]).T;

# Noise Params
q = 0.05; # std
w = 0.1; # std
w_gps = 1.0; # std
q_perceived = q*q;
w_perceived = w*w;
w_gps_perceived = w_gps*w_gps;

q_perceived_tracking = 0.05;
w_perceived_nonlinear = 0.2;
w_perceived_modem_range = 0.1;
w_perceived_modem_azimuth = w_perceived_nonlinear;

w_perceived_sonar_range = 0.1;
w_perceived_sonar_azimuth = w_perceived_nonlinear;

# Initialize x_gt
x_gt = np.zeros((TOTAL_STATES,1))
for i in range(NUM_AGENTS):
    x_gt[STATES*i,0] = MAP_DIM*np.random.uniform() - MAP_DIM / 2.0
    x_gt[STATES*i+1,0] = MAP_DIM*np.random.uniform() - MAP_DIM / 2.0
    x_gt[STATES*i+2,0] = normalize_angle( 2*np.pi*np.random.uniform() )

Q = np.eye(TOTAL_STATES)
for a in range(NUM_AGENTS):
    Q[STATES*a+3, STATES*a+3] = 0.1
    Q[STATES*a+4, STATES*a+4] = 0.1
    Q[STATES*a+5, STATES*a+5] = 0.1

# vel_cmd = np.zeros((2*NUM_AGENTS,1))
# U = np.zeros((TOTAL_STATES, 2*NUM_AGENTS))
# for a in range(NUM_AGENTS):
#     U[STATES*a+3, 2*a] = 1; # x vel
#     U[STATES*a+4, 2*a+1] = 1; # y vel

waypoints = np.zeros((2, NUM_AGENTS))
for a in range(NUM_AGENTS):
    waypoints[:, a] = get_waypoint(MAP_DIM)
print(waypoints)

print(x_gt)

# HISTORY
x_gt_history = np.zeros((TOTAL_STATES, NUM_LOOPS))

for loop_num in range(NUM_LOOPS):
    # print(loop_num)

    # Cheeck reached waypoint
    for a in range(NUM_AGENTS):
        x_pos = x_gt[STATES*a:STATES*a+2, 0]
        delta = np.linalg.norm( waypoints[:,a] - x_pos )
        if delta < 1.0:
            print("Waypoint reached by agent {}!".format(a))
            waypoints[:,a] = get_waypoint(MAP_DIM)
    
    # Get control input
    vel_cmd = np.zeros((2*NUM_AGENTS, 1))
    for a in range(NUM_AGENTS):
        waypoint = waypoints[:,a]
        vel_cmd[2*a:2*a+2,0] = get_control(x_gt, waypoint, a, STATES)

    # Update truth
    for a in range(NUM_AGENTS):
        x_gt_agent = np.reshape( x_gt[STATES*a : STATES*(a+1), 0], (STATES,1) )
        x_gt_agent[3:5, 0] = vel_cmd[2*a:2*a+2,0]
        F = np.eye(6)
        F[0,3] = 1
        F[1,4] = 1
        F[2,5] = 1
        F[3:,3:] = 0
        x_gt_agent = np.dot(F, x_gt_agent)
        x_gt[STATES*a : STATES*(a+1), 0] = x_gt_agent[:,0]

    x_gt = normalize_state(x_gt, NUM_AGENTS, STATES)
    x_gt_history[:, loop_num] = x_gt[:,0]

    # Just check if it's reaching waypoints

print(x_gt)
# plot_path(x_gt_history, MAP_DIM, NUM_LOOPS, waypoints)
#!/usr/bin/env python
from associator import Associator
from kf_filter import LANDMARK_UNCERTAINTY, KalmanFilter
from modem_schedule import modem_schedule
from plot_error_track import plot_error_track
from take_sonar_meas import take_sonar_meas
from take_error_sonar_meas import take_error_sonar_meas
from filter_dvl import filter_dvl
from filter_baro import filter_baro
from filter_compass import filter_compass
from get_estimate_nav import get_estimate_nav
from plot_path import plot_path
from propagate_nav import propagate_nav
from normalize_state import normalize_state
from get_control import get_control
from get_waypoint import get_waypoint
from normalize_angle import normalize_angle
from plot_error_nav import plot_error_nav
import numpy as np
import numpy.matlib
import sys
from copy import deepcopy

from set_estimate_nav import set_estimate_nav

np.random.seed(4)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

"""
My goal for this simulator is to
- Verify in a simple setting correct implementation of DT
"""

AGENT_TO_PLOT = 0

# Simulation

BLUE_NUM = 2
RED_NUM = 1 # OR 1
NUM_AGENTS = BLUE_NUM + RED_NUM
STATES = 8 # Each agent has x,y,z, theta, x_vel,y_vel, z_vel, theta_vel
TRACK_STATES = 6 * NUM_AGENTS # x,y,z, x_dot, y_dot, z_dot for each agent
TOTAL_STATES = STATES * NUM_AGENTS
TOTAL_TRACK_STATES = TRACK_STATES * BLUE_NUM
NUM_LOOPS = 300
MAP_DIM = 20 # Square with side length
PROB_DETECTION = 1.0
SONAR_RANGE = 30.0
MODEM_LOCATION = [11,11,0]
DELTA_RANGE = list(range(1,256))
DELTA_DICT = {"sonar_range" : 0.02, "sonar_azimuth" : 0.01}
BUFFER_SIZE = 32
LOST_AGENT_STD = 10 # Standard deviation for the associator to consider this agent "lost"
SCAN_ANGLE_SIZE = 40 * (np.pi / 180.0)
scan_start_angles = [0] * BLUE_NUM
PING_THRESH = 10.0
LOST_THRESH = 20.0
LANDMARK_LOC = None #[5,5,0]

# Noise Params
q_pos = 0.01 # std
q_theta = 0.01
# q_pos = 0.0 # std
# q_theta = 0.0

w = 0.1 # std
w_gps = 1.0 # std
q_nav_pos = 0.005
q_nav_vel = 0.1
w_perceived = w*w
w_gps_perceived = w_gps*w_gps

q_perceived_tracking_pos = 0.02
q_perceived_tracking_vel = 0.0005

w_perceived_modem_range = 0.3
w_perceived_modem_azimuth = 0.2
w_perceived_sonar_range = 0.3
w_perceived_sonar_azimuth = 0.2

# Initialize x_gt
x_gt = np.zeros((TOTAL_STATES,1))
for i in range(NUM_AGENTS):
    x_gt[STATES*i,0] = MAP_DIM*np.random.uniform() - MAP_DIM / 2.0
    x_gt[STATES*i+1,0] = MAP_DIM*np.random.uniform() - MAP_DIM / 2.0
    # zero z
    x_gt[STATES*i+3,0] = normalize_angle( 2*np.pi*np.random.uniform() )
    # x_gt[STATES*i+4,0] = np.random.uniform()
    # x_gt[STATES*i+5,0] = np.random.uniform()
    

# Initialize Nav Filter Estimate
P = 0.1 * np.eye(STATES)
x_navs = deepcopy( np.reshape( x_gt, (STATES, NUM_AGENTS), "F"))
P_navs = np.matlib.repmat(P, 1, NUM_AGENTS)

# Trim red agent states
x_navs = x_navs[:, :BLUE_NUM]
P_navs = P_navs[:, :STATES*BLUE_NUM]

x_navs_history = np.zeros((STATES*BLUE_NUM, NUM_LOOPS))
P_navs_history = np.zeros((STATES*BLUE_NUM, NUM_LOOPS*STATES))

U = np.zeros((STATES,2))
U[4,0] = 1
U[5,1] = 1
U *= 0.05

waypoints = np.zeros((2, NUM_AGENTS))
for a in range(NUM_AGENTS):
    waypoints[:, a] = get_waypoint(MAP_DIM)

blue_positions = []
for a in range(BLUE_NUM):
    agent_state = x_gt[STATES*a : STATES*(a+1),0]
    pos = agent_state[:3].tolist()
    blue_positions.append(pos)

if LANDMARK_LOC is not None:
    landmark_positions = [LANDMARK_LOC]
else:
    landmark_positions = []

blue_filters = []
for b in range(BLUE_NUM):
    blue_position = deepcopy(blue_positions)
    # save_value = blue_position[b]
    # blue_position = [[]] * len(blue_position)
    # blue_position[b] = save_value

    kf = KalmanFilter(blue_position, landmark_positions, RED_NUM, is_deltatier=True)
    blue_filters.append( kf )

blue_associators = []
for b in range(BLUE_NUM):
    associator = Associator(10, LOST_AGENT_STD, 3, q_perceived_tracking_pos*np.eye(2))
    blue_associators.append( associator )

# HISTORY
x_gt_history = np.zeros((TOTAL_STATES, NUM_LOOPS))

for loop_num in range(NUM_LOOPS):
    # print(loop_num)

    # Cheeck reached waypoint
    for a in range(NUM_AGENTS):
        x_pos = x_gt[STATES*a:STATES*a+2, 0]
        delta = np.linalg.norm( waypoints[:,a] - x_pos )
        if delta < 1.0:
            #print("Waypoint reached by agent {}!".format(a))
            waypoints[:,a] = get_waypoint(MAP_DIM)
    
    # Get control input
    vel_cmd = np.zeros((2*NUM_AGENTS, 1))
    for a in range(NUM_AGENTS):
        waypoint = waypoints[:,a]
        vel_cmd[2*a:2*a+2,0] = get_control(x_gt, waypoint, a, STATES)

    # Update truth
    for a in range(NUM_AGENTS):
        x_gt_agent = np.reshape( x_gt[STATES*a : STATES*(a+1), 0], (STATES,1) )
        F = np.eye(STATES)
        F[0,4] = 1
        F[1,5] = 1
        F[2,6] = 1
        F[3,7] = 1
        vel = x_gt_agent[4:6,0]
        target_vel = vel_cmd[2*a:2*a+2,0]
        deltav = target_vel - vel
        control = np.reshape( U @ deltav, (STATES,1) )

        Q = np.random.normal(0.0, [q_pos]*2 +[0] + [q_theta] + [0]*4)
        x_gt_agent = np.dot(F, x_gt_agent) + control
        x_gt_agent += Q.reshape(-1,1)
        x_gt[STATES*a : STATES*(a+1), 0] = x_gt_agent[:,0]

    x_gt = normalize_state(x_gt, NUM_AGENTS, STATES)
    x_gt_history[:, loop_num] = x_gt[:,0]

    for a in range(BLUE_NUM):
        
        # Navigation Filter Prediction
        x_nav, P_nav = get_estimate_nav(x_navs, P_navs, a, STATES)
        F = np.eye(STATES)
        F[0,4] = 1
        F[1,5] = 1
        F[2,6] = 1
        F[3,7] = 1
        x_nav = F @ x_nav
        Q_nav = [q_nav_pos]*4 + [q_nav_vel]*4
        P_nav = F @ P_nav @ F.T + np.diag(Q_nav)
        x_nav = normalize_state(x_nav, 1, STATES)

        # Nav Filter Correction
        x_nav, P_nav = filter_dvl(x_nav, P_nav, x_gt, w, w_perceived, NUM_AGENTS, STATES, a)
        x_nav, P_nav = filter_baro(x_nav, P_nav, x_gt, w, w_perceived, NUM_AGENTS, STATES, a)
        x_nav, P_nav = filter_compass(x_nav, P_nav, x_gt, w, w_perceived, NUM_AGENTS, STATES, a)

        x_navs, P_navs = set_estimate_nav(x_nav, P_nav, x_navs, P_navs, a, STATES)


        # Update Tracking Filter
        kf = blue_filters[a]
        kf.propogate(q_perceived_tracking_pos, q_perceived_tracking_vel)

        # TODO add a scan region
        associator = blue_associators[a]
        scan_start_angle = scan_start_angles[a]
        scan_start_angles[a] = take_sonar_meas(kf, associator, x_gt, x_nav, a, w, w_perceived_sonar_range, w_perceived_sonar_azimuth, SONAR_RANGE, PROB_DETECTION, STATES, loop_num, scan_start_angle, SCAN_ANGLE_SIZE, PING_THRESH, LOST_THRESH, LANDMARK_LOC)
        # take_error_sonar_meas(kf, associator, x_gt, x_nav, a, w, w_perceived_sonar_range, w_perceived_sonar_azimuth, SONAR_RANGE, PROB_DETECTION, STATES, loop_num, scan_start_angle, SCAN_ANGLE_SIZE)

    for a in range(BLUE_NUM):
        modem_schedule(loop_num, blue_filters, x_gt, a, STATES, BLUE_NUM, MODEM_LOCATION, w, \
            w_perceived_modem_range, w_perceived_modem_azimuth, q_perceived_tracking_pos, \
            q_perceived_tracking_vel, BUFFER_SIZE, DELTA_RANGE, DELTA_DICT, x_gt_history)

    # Intersect estimates
    for a in range(BLUE_NUM):
        x_nav, P_nav = get_estimate_nav(x_navs, P_navs, a, STATES)
        kf = blue_filters[a]
        x_nav, P_nav = kf.intersect_strapdown(x_nav, P_nav, a, share_depth=True)
        x_navs, P_navs = set_estimate_nav(x_nav, P_nav, x_navs, P_navs, a, STATES)

    # Record x_navs_history, P_navs_history
    x_navs_history[:, loop_num] = np.reshape(x_navs, -1, "F")
    for a in range(BLUE_NUM):
        _, P_nav = get_estimate_nav(x_navs, P_navs, a, STATES)
        P_navs_history[STATES*a:STATES*(a+1), STATES*loop_num: STATES*(loop_num+1)] = P_nav

# print(P_navs)
plot_error_nav(x_navs_history, P_navs_history, x_gt_history, STATES, 0)

agent_to_plot_kf = blue_filters[AGENT_TO_PLOT]
x_hat_history_lst = agent_to_plot_kf.x_hat_history_prior
P_history_lst = agent_to_plot_kf.P_history_prior
plot_error_track(x_gt_history, x_hat_history_lst, P_history_lst, STATES, BLUE_NUM, RED_NUM)
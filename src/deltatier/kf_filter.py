from copy import deepcopy
import numpy as np
from numpy.linalg import inv, norm
from deltatier.normalize_angle import normalize_angle
# from normalize_angle import normalize_angle
import scipy
import scipy.optimize
from scipy.stats import norm as normaldist

"""
Ledgers
- x_nav_history_prior
- P_nav_history_prior
- measurement
- x_hat_history_prior
- P_history_prior

States
- x_common
- P_common
- last_shared
- x_hat
- P

What if I add the useful feature for when we don't know the starting location of a blue agent? Just start a track

Unknown starting locations allow for flexibility of the initialization on the first measurement
They are unuseful for association and require a global position to hone in the uncertainty


THIS MAY COMPLICATE THINGS
we track blue agent positions and uncertainty
we do not track velocity of landmarks

KEEP landmarks in the estimate, so that we can talk about them more naturally..
"""

STATES = 6 # x,y,z, x_vel, y_vel, z_vel
LANDMARK_UNCERTAINTY = 1.0 # Or else we won't be able to associate
KNOWN_POSITION_UNCERTAINTY = 1.0
KNOWN_VELOCITY_UNCERTAINTY = 1e-2
UNKNOWN_AGENT_UNCERTAINTY = 1e6 # Red and blue agents with unknown starting locations

# DELTATIER
MEAS_COLUMNS = ["type", "index", "startx1", "startx2", "data", "R"]
MEAS_TYPES_INDICES = ["modem_range", "modem_azimuth", "sonar_range", "sonar_azimuth", "sonar_range_implicit", "sonar_azimuth_implicit"]

IMPLICIT_BYTE_COST = 0.0
EXPLICIT_BYTE_COST = 1.5

EIGHT_STATE_NAV_FILTER = False # else 6 states

"""
All the todos here
- list creation: maybe can pull straight from the launch file -- maybe unknown blue agents, 
- no landmark_positions
- approximate modem measurements as being taken currently
- should add a parameter for regular kalman filter and deltatier
- 

Separate test for unknown blue agents, so we don't necessarily need a good starting position, acoustic modems will help out
Should repeat the strategy with just launching the if statement to debug


"""

class KalmanFilter:

    def __init__(self, blue_positions, landmark_positions, red_agent=False, is_deltatier=True, \
            known_posititon_unc=KNOWN_POSITION_UNCERTAINTY, \
            known_velocity_unc=KNOWN_VELOCITY_UNCERTAINTY, \
            unknown_agent_unc=UNKNOWN_AGENT_UNCERTAINTY):
        """
        first positions are blue
        next are landmark
        last are red
        
        blue_positions : [[x,y,z],[x,y,z],[],...]
        landmark_positions: [[x,y,z],[x,y,z],...]
        red_agent : bool # Whether there is a red agent in the environment
        is_deltatier : bool
        """
        self.BLUE_NUM = len(blue_positions)
        self.LANDMARK_NUM = len(landmark_positions)
        self.RED_NUM = 1 if red_agent else 0
        self.NUM_AGENTS = self.BLUE_NUM + self.RED_NUM + self.LANDMARK_NUM

        self.index = -1

        # Initialize estimates using provided positions
        self.P = []
        self.x_hat = []
        for p in blue_positions:
            
            if not p: # unknown starting blue position
                self.x_hat.extend([0]*3) # Position
                self.x_hat.extend([0]*3) # Velocity
                self.P.extend([unknown_agent_unc]*3 + [known_velocity_unc]*3)
            else:
                self.x_hat.extend(p) # Position
                self.x_hat.extend([0]*3) # Velocity
                self.P.extend([known_posititon_unc]*3 + [known_velocity_unc]*3)

        if red_agent:
            self.x_hat.extend([0]*3) # Position
            self.x_hat.extend([0]*3) # Velocity
            self.P.extend([unknown_agent_unc]*3 + [known_velocity_unc]*3)

        for p in landmark_positions:
            self.x_hat.extend(p) # Position (no velocity)
            self.P.extend([LANDMARK_UNCERTAINTY, LANDMARK_UNCERTAINTY, known_posititon_unc]) # great x,y knowledge, low z
        
        self.NUM_STATES = len(self.x_hat)
        self.x_hat = np.reshape(self.x_hat, (-1,1))
        self.P = np.diag(self.P)

        # Used for OOSM and DT
        self.x_hat_history_prior = [] # x_hat at index is estimate prior to index filtering step
        self.P_history_prior = []
        self.ledger = []
        self.x_nav_history_prior = []
        self.P_nav_history_prior = []

        self.is_deltatier = is_deltatier
        if self.is_deltatier:
            self.x_common = self.x_hat
            self.P_common = self.P
            self.common_ledger = [] # Just for modem updates
            """
            Reason for common ledger: We don't want to share modem update since they have already been delivered
            However the rx_buffer process updates the common estimate with the shared_buffer
            Since the buffer does not contain the modem meas, we must introduce some other way
            """
            self.last_share_index = 0

    def get_agent_ids(self):
        """
        Returns the ids (startx's for agents)
        """
        ids = []
        for a in range(self.BLUE_NUM + self.RED_NUM + self.LANDMARK_NUM):
            ids.append( self._get_agent_state_index( a ) )
        return ids


    def propogate(self, position_process_noise, velocity_process_noise):
        """
        Only 1Hz update is supported for DeltaTier
        """
        self.index += 1

        # Save the last estimate
        self.x_hat_history_prior.append( np.copy(self.x_hat))
        self.P_history_prior.append( np.copy(self.P))

        self.x_hat, self.P = KalmanFilter._propogate( \
            self.x_hat, self.P, position_process_noise, \
            velocity_process_noise, self.BLUE_NUM, self.RED_NUM, \
            self.LANDMARK_NUM)

        return self.index

    # Either to an agent or landmark
    def filter_range_tracked(self, meas_value, R, collecting_agent, collected_agent): 
        startx1 = self._get_agent_state_index(collecting_agent)
        startx2 = self._get_agent_state_index(collected_agent)
        self.x_hat, self.P = KalmanFilter._fuse_range_tracked(self.x_hat, self.P, startx1, startx2, meas_value, R)

        # Add to ledger
        type_ind = MEAS_TYPES_INDICES.index("sonar_range")
        meas_row = [type_ind, self.index, startx1, startx2, meas_value, R]
        self.ledger.append(meas_row)

    # Either to an agent or landmark
    def filter_azimuth_tracked(self, meas_value, R, collecting_agent, collected_agent): 
        """
        Meas value must be relative to x-axis because of linear filter
        """
        startx1 = self._get_agent_state_index(collecting_agent)
        startx2 = self._get_agent_state_index(collected_agent)
        self.x_hat, self.P = KalmanFilter._fuse_azimuth_tracked(self.x_hat, self.P, startx1, startx2, meas_value, R)

        # Add to ledger
        type_ind = MEAS_TYPES_INDICES.index("sonar_azimuth")
        meas_row = [type_ind, self.index, startx1, startx2, meas_value, R]
        self.ledger.append(meas_row)

    # Used in minau project for psuedo-gps measurements from surface beacon
    def filter_range_from_untracked(self, meas_value, R, position, collected_agent, index=None):
        """
        For OOSM problem, add index to be fused at
        """
        startx1 = self._get_agent_state_index(collected_agent)

        # Fuse measurement now if not OOSM
        if index is None or index == self.index:
            self.x_hat, self.P = KalmanFilter._fuse_range_from_untracked(self.x_hat, self.P, startx1, position, meas_value, R)

        # Add to ledger
        type_ind = MEAS_TYPES_INDICES.index("modem_range")
        ind = index if index is not None else self.index
        meas_row = [type_ind, ind, startx1, -1, meas_value, R]
        self.ledger.append(meas_row)
        if self.is_deltatier:
            self.common_ledger.append(meas_row)
        
    # Used in minau project for psuedo-gps measurements from surface beacon
    def filter_azimuth_from_untracked(self, meas_value, R, position, collected_agent, index=None):
        """
        Meas value must be relative to x-axis because of linear filter
        For OOSM problem, add index to be fused at
        """
        startx1 = self._get_agent_state_index(collected_agent)

        # Fuse measurement now if not OOSM
        if index is None or index == self.index:
            self.x_hat, self.P = KalmanFilter._fuse_azimuth_from_untracked(self.x_hat, self.P, startx1, position, meas_value, R)

        # Add to ledger
        type_ind = MEAS_TYPES_INDICES.index("modem_azimuth")
        ind = index if index is not None else self.index
        meas_row = [type_ind, ind, startx1, -1, meas_value, R]
        self.ledger.append(meas_row)
        if self.is_deltatier:            
            self.common_ledger.append(meas_row)
    
    def pull_buffer(self, mults, delta_dict, position_process_noise, velocity_process_noise, modem_loc, buffer_size):
        """
        mults : list of floats/ints
        delta_dict : dict{"meas_type" -> base_delta }
        """
        middle_index = int( len(mults) / 2 )
        mult = mults[ middle_index ]

        x_common_start = self.x_common
        P_common_start = self.P_common

        ledger_mat = KalmanFilter._get_ledger_mat(self.ledger)

        while True:

            x_common = x_common_start
            P_common = P_common_start

            share_buffer = []

            # Loop through all time indices
            for index in range(self.last_share_index, self.index+1):
                measurements = self._get_measurements_at_time(ledger_mat, index)

                x_common, P_common = KalmanFilter._propogate(x_common, P_common, \
                    position_process_noise, velocity_process_noise, self.BLUE_NUM, self.RED_NUM,\
                    self.LANDMARK_NUM)

                x_common_bar = x_common
                P_common_bar = P_common

                for mi in range(measurements.shape[0]):
                    meas = measurements[mi,:]
                    meas_type = MEAS_TYPES_INDICES[ int(meas[MEAS_COLUMNS.index("type")]) ]
                    startx1 = int(meas[MEAS_COLUMNS.index("startx1")])
                    startx2 = int(meas[MEAS_COLUMNS.index("startx2")])
                    data = meas[MEAS_COLUMNS.index("data")]
                    R = meas[MEAS_COLUMNS.index("R")]

                    # ["modem_range", "modem_azimuth", "sonar_range", "sonar_azimuth", "sonar_range_implicit", "sonar_azimuth_implicit"]
                    if meas_type == "modem_range":
                        x_common, P_common = KalmanFilter._fuse_range_from_untracked( 
                            x_common, P_common, startx1, modem_loc, data, R)
                        # Don't add to shared_buffer
                    elif meas_type == "modem_azimuth":
                        x_common, P_common = KalmanFilter._fuse_azimuth_from_untracked(
                            x_common, P_common, startx1, modem_loc, data, R)
                        # Don't add to shared_buffer
                    elif meas_type == "sonar_range":
                        pred, H = KalmanFilter._predict_range(x_common, startx1, startx2)
                        innovation = data - pred
                        delta = delta_dict["sonar_range"] * mult

                        # Fuse explicitly
                        if abs(innovation) > delta:
                            x_common, P_common = KalmanFilter._fuse_range_tracked(
                                x_common, P_common, startx1, startx2, data, R)

                            type_ind = MEAS_TYPES_INDICES.index("sonar_range")
                            meas_row = [type_ind, index, startx1, startx2, data, R]
                            share_buffer.append( meas_row )

                        else: # Fuse implicitly
                            x_ref = x_common
                            h_x_hat = pred
                            h_x_bar, H_ = KalmanFilter._predict_range(x_common_bar, startx1, startx2)
                            h_x_ref, H_ = KalmanFilter._predict_range(x_ref, startx1, startx2)
                            x_common, P_common = KalmanFilter._implicit_fuse(
                                x_common_bar, P_common_bar, x_common, P_common, 
                                x_ref, H, R, delta, 
                                h_x_hat, h_x_bar, h_x_ref, False)
                            np.linalg.matrix_rank(P_common) # Just check matrix isn't singular

                            type_ind = MEAS_TYPES_INDICES.index("sonar_range_implicit")
                            meas_row = [type_ind, index, startx1, startx2, 0.0, R]
                            share_buffer.append( meas_row )
                        
                    elif meas_type == "sonar_azimuth":
                        pred, H = KalmanFilter._predict_azimuth(x_common, startx1, startx2)
                        innovation = normalize_angle( data - pred )
                        delta = delta_dict["sonar_azimuth"] * mult

                        # Fuse explicitly
                        if abs(innovation) > delta:
                            x_common, P_common = KalmanFilter._fuse_azimuth_tracked(
                                x_common, P_common, startx1, startx2, data, R)

                            type_ind = MEAS_TYPES_INDICES.index("sonar_azimuth")
                            meas_row = [type_ind, index, startx1, startx2, data, R]
                            share_buffer.append( meas_row )

                        else: # Fuse implicitly
                            x_ref = x_common
                            h_x_hat = pred
                            h_x_bar, H_ = KalmanFilter._predict_azimuth(x_common_bar, startx1, startx2)
                            h_x_ref, H_ = KalmanFilter._predict_azimuth(x_ref, startx1, startx2)
                            x_common, P_common = KalmanFilter._implicit_fuse(
                                x_common_bar, P_common_bar, x_common, P_common, 
                                x_ref, H, R, delta, 
                                h_x_hat, h_x_bar, h_x_ref, True)
                            np.linalg.matrix_rank(P_common) # Just check matrix isn't singular

                            type_ind = MEAS_TYPES_INDICES.index("sonar_azimuth_implicit")
                            meas_row = [type_ind, index, startx1, startx2, 0.0, R]
                            share_buffer.append( meas_row )
                    else:
                        raise ValueError("Unrecognized measurement type: " + str(meas_type))

            if not share_buffer: # We didn't take any shareable measurements!
                return mults[0], [], 0, 0
            # Check if share_buffer overflowed
            share_buffer_mat = KalmanFilter._get_ledger_mat(share_buffer)
            cost, explicit_cnt, implicit_cnt = self._get_buffer_size( share_buffer_mat )
            if cost > buffer_size:
                if len(mults) == 1: # There were no matches!
                    raise Exception("There were no matching deltatiers!")
                mults = mults[middle_index+1:]
            else:
                if len(mults) == 1: # We've found our multiplier!
                    break
                mults = mults[:middle_index+1]

            # Pick the middle delta
            middle_index = int( len(mults) / 2 ) - 1
            mult = mults[ middle_index ]

        self.x_common = x_common
        self.P_common = P_common
        self.last_share_index = self.index + 1 # Don't share this index again

        return mult, share_buffer, explicit_cnt, implicit_cnt

    def rx_buffer(self, mult, buffer, delta_dict, modem_loc, position_process_noise, velocity_process_noise, agent, fast_ci=False):
        """
        Ignore artificially inserting depth in the rewind...
        """
        # Merge x_common with buffer so modem information shared
        buffer.extend(self.common_ledger)

        ledger_mat = KalmanFilter._get_ledger_mat(self.ledger)
        buffer_mat = KalmanFilter._get_ledger_mat(buffer)

        index_col = MEAS_COLUMNS.index("index")
        last_share_index = int(buffer_mat.max(axis=0)[index_col])

        # THEN add the fusion. This is actually a simpler function than pull_buffer()

        x_common = self.x_common
        P_common = self.P_common
        x_hat = self.x_hat_history_prior[self.last_share_index]
        P = self.P_history_prior[self.last_share_index]

        # Loop through all time indices
        for index in range(self.last_share_index, self.index):

            x_common, P_common = KalmanFilter._propogate(x_common, P_common, \
                position_process_noise, velocity_process_noise, self.BLUE_NUM, self.RED_NUM,\
                self.LANDMARK_NUM)
            x_hat, P = KalmanFilter._propogate(x_hat, P, \
                position_process_noise, velocity_process_noise, self.BLUE_NUM, self.RED_NUM,\
                self.LANDMARK_NUM)

            x_common_bar = deepcopy(x_common)
            P_common_bar = deepcopy(P_common)
            x_hat_bar = deepcopy(x_hat)
            P_bar = deepcopy(P)

            # Shared Buffer First!
            measurements = self._get_measurements_at_time(buffer_mat, index)
            for mi in range(measurements.shape[0]):
                meas = measurements[mi,:]
                meas_type = MEAS_TYPES_INDICES[ int(meas[MEAS_COLUMNS.index("type")]) ]
                startx1 = int(meas[MEAS_COLUMNS.index("startx1")])
                startx2 = int(meas[MEAS_COLUMNS.index("startx2")])
                data = meas[MEAS_COLUMNS.index("data")]
                R = meas[MEAS_COLUMNS.index("R")]

                if meas_type == "modem_range":
                    x_common, P_common = KalmanFilter._fuse_range_from_untracked( 
                        x_common, P_common, startx1, modem_loc, data, R)
                elif meas_type == "modem_azimuth":
                    x_common, P_common = KalmanFilter._fuse_azimuth_from_untracked(
                        x_common, P_common, startx1, modem_loc, data, R)
                elif meas_type == "sonar_range":

                    x_common, P_common = KalmanFilter._fuse_range_tracked(
                        x_common, P_common, startx1, startx2, data, R)
                    x_hat, P = KalmanFilter._fuse_range_tracked(
                        x_hat, P, startx1, startx2, data, R)
                elif meas_type == "sonar_range_implicit":
                    delta = delta_dict["sonar_range"] * mult

                    # Main Estimate
                    pred, H = KalmanFilter._predict_range(x_common, startx1, startx2)
                    x_ref = x_common
                    h_x_hat = pred
                    h_x_bar, H_ = KalmanFilter._predict_range(x_common_bar, startx1, startx2)
                    h_x_ref, H_ = KalmanFilter._predict_range(x_ref, startx1, startx2)
                    x_common, P_common = KalmanFilter._implicit_fuse(
                        x_common_bar, P_common_bar, x_common, P_common, 
                        x_ref, H, R, delta, 
                        h_x_hat, h_x_bar, h_x_ref, False)
                    np.linalg.matrix_rank(P_common) # Just check matrix isn't singular

                    # Main Estimate
                    pred, H = KalmanFilter._predict_range(x_hat, startx1, startx2)
                    x_ref = x_common
                    h_x_hat = pred
                    h_x_bar, H_ = KalmanFilter._predict_range(x_hat_bar, startx1, startx2)
                    h_x_ref, H_ = KalmanFilter._predict_range(x_ref, startx1, startx2)
                    x_hat, P = KalmanFilter._implicit_fuse(
                        x_hat_bar, P_bar, x_hat, P, 
                        x_ref, H, R, delta, 
                        h_x_hat, h_x_bar, h_x_ref, False)
                    np.linalg.matrix_rank(P) # Just check matrix isn't singular
                    
                elif meas_type == "sonar_azimuth":
                    x_common, P_common = KalmanFilter._fuse_azimuth_tracked(
                        x_common, P_common, startx1, startx2, data, R)
                    x_hat, P = KalmanFilter._fuse_azimuth_tracked(
                        x_hat, P, startx1, startx2, data, R)

                elif meas_type == "sonar_azimuth_implicit":
                    delta = delta_dict["sonar_azimuth"] * mult

                    # Main Estimate
                    pred, H = KalmanFilter._predict_azimuth(x_common, startx1, startx2)
                    x_ref = x_common
                    h_x_hat = pred
                    h_x_bar, H_ = KalmanFilter._predict_azimuth(x_common_bar, startx1, startx2)
                    h_x_ref, H_ = KalmanFilter._predict_azimuth(x_ref, startx1, startx2)
                    x_common, P_common = KalmanFilter._implicit_fuse(
                        x_common_bar, P_common_bar, x_common, P_common, 
                        x_ref, H, R, delta, 
                        h_x_hat, h_x_bar, h_x_ref, True)
                    np.linalg.matrix_rank(P_common) # Just check matrix isn't singular

                    # Main Estimate
                    pred, H = KalmanFilter._predict_azimuth(x_hat, startx1, startx2)
                    x_ref = x_common
                    h_x_hat = pred
                    h_x_bar, H_ = KalmanFilter._predict_azimuth(x_hat_bar, startx1, startx2)
                    h_x_ref, H_ = KalmanFilter._predict_azimuth(x_ref, startx1, startx2)
                    x_hat, P = KalmanFilter._implicit_fuse(
                        x_hat_bar, P_bar, x_hat, P, 
                        x_ref, H, R, delta, 
                        h_x_hat, h_x_bar, h_x_ref, True)
                    np.linalg.matrix_rank(P) # Just check matrix isn't singular

                else:
                    raise ValueError("Unrecognized measurement type: " + str(meas_type))

            # end for measurements
            # Main Filter measurements
            measurements = self._get_measurements_at_time(ledger_mat, index)
            for mi in range(measurements.shape[0]):
                meas = measurements[mi,:]
                meas_type = MEAS_TYPES_INDICES[ int(meas[MEAS_COLUMNS.index("type")]) ]
                startx1 = int(meas[MEAS_COLUMNS.index("startx1")])
                startx2 = int(meas[MEAS_COLUMNS.index("startx2")])
                data = meas[MEAS_COLUMNS.index("data")]
                R = meas[MEAS_COLUMNS.index("R")]

                if meas_type == "modem_range":
                    x_hat, P = KalmanFilter._fuse_range_from_untracked( 
                        x_hat, P, startx1, modem_loc, data, R)
                elif meas_type == "modem_azimuth":
                    x_hat, P = KalmanFilter._fuse_azimuth_from_untracked(
                        x_hat, P, startx1, modem_loc, data, R)
                elif meas_type == "sonar_range":
                    x_hat, P = KalmanFilter._fuse_range_tracked(
                        x_hat, P, startx1, startx2, data, R)
                elif meas_type == "sonar_azimuth":
                    x_hat, P = KalmanFilter._fuse_azimuth_tracked(
                        x_hat, P, startx1, startx2, data, R)
                else:
                    raise ValueError("Unrecognized measurement type: " + str(meas_type))
            
            # Intersect with navigation filter
            if len(self.x_nav_history_prior) > index:
                x_nav = self.x_nav_history_prior[index]
                P_nav = self.P_nav_history_prior[index]
                x_hat, P = KalmanFilter._filter_artificial_depth_static(x_hat, P, x_nav[2], self.BLUE_NUM, self.RED_NUM, self.LANDMARK_NUM, R=1e-2)
                
                if EIGHT_STATE_NAV_FILTER:
                    # Go 8 -> 6
                    rot_mat_nav = np.array([
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]]
                    )
                    nav_est_x = np.dot( rot_mat_nav, x_nav)
                    nav_est_P = np.dot( np.dot(rot_mat_nav, P_nav), rot_mat_nav.T)
                else:
                    nav_est_x = x_nav
                    nav_est_P = P_nav

                rot_mat = np.zeros((6, x_hat.shape[0]))
                rot_mat[:,6*agent:6*(agent+1)] = np.eye(6)
                x_hat_agent = np.dot(rot_mat, x_hat)
                P_agent = np.dot( np.dot( rot_mat, P ), rot_mat.T)

                if fast_ci:
                    mean_result, cov_result = KalmanFilter._fast_covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)
                else:
                    mean_result, cov_result = KalmanFilter._covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)

                # PSCI for tracking filter
                D_inv = inv(cov_result) - inv(P_agent)
                D_inv_d = np.dot( inv(cov_result), mean_result) - np.dot( inv(P_agent), x_hat_agent)
                D_inv_zeros = np.dot( np.dot( rot_mat.T, D_inv), rot_mat)
                D_inv_d_zeros = np.dot( rot_mat.T, D_inv_d)
                P_new = inv( inv(P) + D_inv_zeros)
                x_hat = np.dot(P_new, np.dot( inv(P), x_hat) + D_inv_d_zeros )
                P = P_new

        # end for index
        self.x_hat = x_hat
        self.P = P
        self.x_common = x_common
        self.P_common = P_common
        self.last_share_index = last_share_index
    
    def catch_up(self, start_index, modem_loc, position_process_noise, velocity_process_noise, agent, fast_ci=False):
        """
        In the case of OOSM catch the current filter up to the current timestep
        """
        x_hat = deepcopy(self.x_hat_history_prior[start_index])
        P = deepcopy(self.P_history_prior[start_index])

        ledger_mat = KalmanFilter._get_ledger_mat(self.ledger)
        for index in range(start_index, self.index + 1):
            x_hat, P = KalmanFilter._propogate(x_hat, P, \
                position_process_noise, velocity_process_noise, self.BLUE_NUM, self.RED_NUM,\
                self.LANDMARK_NUM)
            measurements = self._get_measurements_at_time(ledger_mat, index)
            for mi in range(measurements.shape[0]):
                meas = measurements[mi,:]
                meas_type = MEAS_TYPES_INDICES[ int(meas[MEAS_COLUMNS.index("type")]) ]
                startx1 = int(meas[MEAS_COLUMNS.index("startx1")])
                startx2 = int(meas[MEAS_COLUMNS.index("startx2")])
                data = meas[MEAS_COLUMNS.index("data")]
                R = meas[MEAS_COLUMNS.index("R")]

                if meas_type == "modem_range":
                    x_hat, P = KalmanFilter._fuse_range_from_untracked( 
                        x_hat, P, startx1, modem_loc, data, R)
                elif meas_type == "modem_azimuth":
                    x_hat, P = KalmanFilter._fuse_azimuth_from_untracked(
                        x_hat, P, startx1, modem_loc, data, R)
                elif meas_type == "sonar_range":
                    x_hat, P = KalmanFilter._fuse_range_tracked(
                        x_hat, P, startx1, startx2, data, R)
                elif meas_type == "sonar_azimuth":
                    x_hat, P = KalmanFilter._fuse_azimuth_tracked(
                        x_hat, P, startx1, startx2, data, R)
                else:
                    raise ValueError("Unrecognized measurement type: " + str(meas_type))
            
            # Intersect with navigation filter
            if len(self.x_nav_history_prior) > index:
                x_nav = self.x_nav_history_prior[index]
                P_nav = self.P_nav_history_prior[index]
                x_hat, P = KalmanFilter._filter_artificial_depth_static(x_hat, P, x_nav[2], self.BLUE_NUM, self.RED_NUM, self.LANDMARK_NUM, R=1e-2)
                
                if EIGHT_STATE_NAV_FILTER:
                    # Go 8 -> 6
                    rot_mat_nav = np.array([
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]]
                    )
                    nav_est_x = np.dot( rot_mat_nav, x_nav)
                    nav_est_P = np.dot( np.dot(rot_mat_nav, P_nav), rot_mat_nav.T)
                else:
                    nav_est_x = x_nav
                    nav_est_P = P_nav

                rot_mat = np.zeros((6, x_hat.shape[0]))
                rot_mat[:,6*agent:6*(agent+1)] = np.eye(6)
                x_hat_agent = np.dot(rot_mat, x_hat)
                P_agent = np.dot( np.dot( rot_mat, P ), rot_mat.T)

                if fast_ci:
                    mean_result, cov_result = KalmanFilter._fast_covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)
                else:
                    mean_result, cov_result = KalmanFilter._covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)

                # PSCI for tracking filter
                D_inv = inv(cov_result) - inv(P_agent)
                D_inv_d = np.dot( inv(cov_result), mean_result) - np.dot( inv(P_agent), x_hat_agent)
                D_inv_zeros = np.dot( np.dot( rot_mat.T, D_inv), rot_mat)
                D_inv_d_zeros = np.dot( rot_mat.T, D_inv_d)
                P_new = inv( inv(P) + D_inv_zeros)
                x_hat = np.dot(P_new, np.dot(inv(P), x_hat) + D_inv_d_zeros )
                P = P_new

        self.x_hat = x_hat
        self.P = P

    def intersect_strapdown(self, x_nav, P_nav, agent, fast_ci=False, share_depth=True):
        """
        share_depth: take the depth estimate from x_nav and filter it as part of every agent
        """
        if share_depth:
            self._filter_artificial_depth(x_nav[2])

        if self.is_deltatier:
            self.x_nav_history_prior.append(deepcopy(x_nav))
            self.P_nav_history_prior.append(deepcopy(P_nav))

        if EIGHT_STATE_NAV_FILTER:
            # Go 8 -> 6            
            rot_mat_nav = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0]]
            )
            nav_est_x = np.dot( rot_mat_nav, x_nav)
            nav_est_P = np.dot( np.dot(rot_mat_nav, P_nav), rot_mat_nav.T)
        else:
            nav_est_x = x_nav
            nav_est_P = P_nav

        x_hat_agent, P_agent, rot_mat = self.get_agent_states(agent)

        if fast_ci:
            mean_result, cov_result = KalmanFilter._fast_covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)
        else:
            mean_result, cov_result = KalmanFilter._covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)

        # PSCI for tracking filter        self.x_hat = P_new @ (inv(self.P) @ self.x_hat + D_inv_d_zeros )
        D_inv = inv(cov_result) - inv(P_agent)
        D_inv_d = np.dot( inv(cov_result), mean_result) - np.dot( inv(P_agent), x_hat_agent)
        D_inv_zeros = np.dot( np.dot( rot_mat.T, D_inv), rot_mat)
        D_inv_d_zeros = np.dot( rot_mat.T, D_inv_d)
        P_new = inv( inv(self.P) + D_inv_zeros)
        self.x_hat = np.dot(P_new, np.dot(inv(self.P), self.x_hat) + D_inv_d_zeros )
        self.P = P_new

        # PSCI for navigation Filter (just X,Y)
        rot_mat = np.zeros((2, 8)) if EIGHT_STATE_NAV_FILTER else np.zeros((2, 6))
        rot_mat[:2,:2] = np.eye(2)
        mean_result = mean_result[:2, 0].reshape(-1,1)
        cov_result = cov_result[:2, :2]
        x_nav_position = np.dot( rot_mat, x_nav)
        P_nav_position = np.dot( np.dot( rot_mat, P_nav), rot_mat.T)

        # PSCI for Nav filter
        D_inv = inv(cov_result) - inv(P_nav_position)
        D_inv_d = np.dot( inv(cov_result), mean_result) - np.dot( inv(P_nav_position), x_nav_position)
        D_inv_zeros = np.dot( np.dot( rot_mat.T, D_inv), rot_mat)
        D_inv_d_zeros = np.dot( rot_mat.T, D_inv_d)
        P_nav_new = inv( inv(P_nav) + D_inv_zeros)
        x_nav = np.dot(P_nav_new, np.dot(inv(P_nav), x_nav) + D_inv_d_zeros )
        P_nav = P_nav_new
        
        return x_nav, P_nav

    def get_agent_states(self, agent):
        rot_mat = np.zeros((6, self.x_hat.shape[0]))
        rot_mat[:,6*agent:6*(agent+1)] = np.eye(6)
        x_hat_agent = np.dot( rot_mat, self.x_hat)
        P_agent = np.dot( np.dot( rot_mat, self.P), rot_mat.T)

        return x_hat_agent, P_agent, rot_mat

    """ 
    ################################################################################################# 
    ################################# BEGIN PRIVATE MEMBER METHODS ##################################
    #################################################################################################
    """

    @staticmethod
    def _propogate(x_hat, P, position_process_noise, velocity_process_noise, BLUE_NUM, RED_NUM, LANDMARK_NUM):
        num_states = x_hat.shape[0]

        # Create F and Q matrices
        F = np.eye(num_states)
        Q = []
        for b in range(BLUE_NUM + RED_NUM):
            F[6*b, 6*b + 3] = 1
            F[6*b + 1, 6*b + 4] = 1
            F[6*b + 2, 6*b + 5] = 1
            Q.extend([position_process_noise]*3 + [velocity_process_noise]*3)
        for l in range( LANDMARK_NUM ):
            Q.extend([0]*3)
        Q = np.diag(Q)

        x_hat = np.dot( F, x_hat)
        P = np.dot(F, np.dot(P, F.T)) + Q

        return x_hat, P

    @staticmethod
    def _fuse(x_hat, P, H, R, innovation):
        K = np.dot( np.dot( P, H.T), inv(np.dot(H, np.dot(P, H.T)) + R))
        x_hat = x_hat + np.dot(K, innovation )
        P = P - np.dot(K, np.dot(H, P))
        return x_hat, P

    @staticmethod
    def _filter_artificial_depth_static(x_hat, P, depth, BLUE_NUM, RED_NUM, LANDMARK_NUM, R=1e-2):

        num_states = x_hat.shape[0]

        for b in range(BLUE_NUM + RED_NUM):
            H = np.zeros((1, num_states))
            H[0, 6*b + 2] = 1

            innovation = depth - np.dot(H, x_hat)
            x_hat, P = KalmanFilter._fuse(x_hat, P, H, R, innovation)

        num_dynamic_states = 6*(BLUE_NUM + RED_NUM)
        for l in range(LANDMARK_NUM):
            H = np.zeros((1, num_states))
            H[0, num_dynamic_states + 3*l + 2] = 1

            innovation = depth - np.dot(H, x_hat)
            x_hat, P = KalmanFilter._fuse(x_hat, P, H, R, innovation)

        return x_hat, P
        
    def _filter_artificial_depth(self, depth, R=1e-2):
        """
        Approximate the environment as 2D and fuse a certain depth for all agents/landmarks
        """
        self.x_hat, self.P = KalmanFilter._filter_artificial_depth_static(
            self.x_hat, self.P, depth, self.BLUE_NUM, self.RED_NUM, self.LANDMARK_NUM, R)

    @staticmethod
    def _fuse_range_tracked(x_hat, P, startx1, startx2, meas_value, R):
    
        pred, H = KalmanFilter._predict_range(x_hat, startx1, startx2)

        innovation = meas_value - pred
        x_hat, P = KalmanFilter._fuse(x_hat, P, H, R, innovation)

        return x_hat, P

    @staticmethod
    def _fuse_azimuth_tracked(x_hat, P, startx1, startx2, meas_value, R):
        pred, H = KalmanFilter._predict_azimuth(x_hat, startx1, startx2)

        innovation = normalize_angle( meas_value - pred )
        x_hat, P = KalmanFilter._fuse(x_hat, P, H, R, innovation)

        return x_hat, P

    @staticmethod
    def _fuse_range_from_untracked(x_hat, P, startx1, position, meas_value, R):
        NUM_STATES = x_hat.shape[0]
        x1 = x_hat[startx1, 0]
        y1 = x_hat[startx1 + 1, 0]
        z1 = x_hat[startx1 + 2, 0]
        x2 = position[0]
        y2 = position[1]
        z2 = position[2]

        delta_pred = np.array([x1 - x2,y1 - y2, z1 - z2])
        if norm(delta_pred) < 1e-3:
            delta_pred[0,0] = 1e-3
            delta_pred[1,0] = 1e-3
            delta_pred[2,0] = 1e-3
        pred = norm(delta_pred)

        drdx1 = delta_pred[0] / norm(delta_pred)
        drdy1 = delta_pred[1] / norm(delta_pred)
        drdz1 = delta_pred[2] / norm(delta_pred)
        
        H = np.zeros((1, NUM_STATES))
        H[0, startx1] = drdx1
        H[0, startx1+1] = drdy1
        H[0, startx1+2] = drdz1

        innovation = meas_value - pred
        x_hat, P = KalmanFilter._fuse(x_hat, P, H, R, innovation)

        return x_hat, P

    @staticmethod
    def _fuse_azimuth_from_untracked(x_hat, P, startx1, position, meas_value, R):
        NUM_STATES = x_hat.shape[0]

        x1 = x_hat[startx1, 0]
        y1 = x_hat[startx1 + 1, 0]
        x2 = position[0]
        y2 = position[1]

        delta_pred = np.array([[x1 - x2],[y1 - y2]])
        if norm(delta_pred) < 1e-3:
            delta_pred[0,0] = 1e-3
            delta_pred[1,0] = 1e-3
        pred = np.arctan2(delta_pred[1,0], delta_pred[0,0])

        dadx = -delta_pred[1,0] / norm(delta_pred)**2
        dady = delta_pred[0,0] / norm(delta_pred)**2
        H = np.zeros((1, NUM_STATES))
        H[0, startx1] = dadx
        H[0, startx1 + 1] = dady
        
        innovation = normalize_angle( meas_value - pred )
        if abs(innovation) < np.radians(90):
            x_hat, P = KalmanFilter._fuse(x_hat, P, H, R, innovation)
        else:
            print("Modem azimuth innovation too large, rejecting: {}".format(innovation))

        return x_hat, P

    @staticmethod
    def _predict_range(x_hat, startx1, startx2):
        NUM_STATES = x_hat.shape[0]

        x1 = x_hat[startx1, 0]
        y1 = x_hat[startx1 + 1, 0]
        z1 = x_hat[startx1 + 2, 0]
        x2 = x_hat[startx2, 0]
        y2 = x_hat[startx2 + 1, 0]
        z2 = x_hat[startx2 + 2, 0]

        delta_pred = np.array([[x2 - x1], [y2 - y1], [z2 - z1]])
        if norm(delta_pred) < 1e-3:
            delta_pred[0,0] = 1e-3
            delta_pred[1,0] = 1e-3
            delta_pred[2,0] = 1e-3
        pred = norm(delta_pred)

        drdx1 = (x1 - x2) / norm(delta_pred)
        drdx2 = (x2 - x1) / norm(delta_pred)
        drdy1 = (y1 - y2) / norm(delta_pred)
        drdy2 = (y2 - y1) / norm(delta_pred)
        drdz1 = (z1 - z2) / norm(delta_pred)
        drdz2 = (z2 - z1) / norm(delta_pred)

        H = np.zeros((1, NUM_STATES))
        H[0, startx1] = drdx1
        H[0, startx2] = drdx2
        H[0, startx1+1] = drdy1
        H[0, startx2+1] = drdy2
        H[0, startx1+2] = drdz1
        H[0, startx2+2] = drdz2

        return pred, H
    
    @staticmethod
    def _predict_azimuth(x_hat, startx1, startx2):
        NUM_STATES = x_hat.shape[0]
        x1 = x_hat[startx1, 0]
        y1 = x_hat[startx1 + 1, 0]
        x2 = x_hat[startx2, 0]
        y2 = x_hat[startx2 + 1, 0]

        delta_pred = np.array([[x2 - x1], [y2 - y1]])
        if norm(delta_pred) < 1e-3:
            delta_pred[0,0] = 1e-3
            delta_pred[1,0] = 1e-3
        pred = np.arctan2(delta_pred[1,0], delta_pred[0,0])

        dadx1 = (y2 - y1) / norm(delta_pred)**2 # TODO check these are correct partials
        dadx2 = -(y2 - y1) / norm(delta_pred)**2
        dady1 = -(x2 - x1) / norm(delta_pred)**2
        dady2 = (x2 - x1) / norm(delta_pred)**2
        H = np.zeros((1, NUM_STATES))
        H[0, startx1] = dadx1
        H[0, startx2] = dadx2
        H[0, startx1+1] = dady1
        H[0, startx2+1] = dady2

        return pred, H
    
    @staticmethod
    def _implicit_fuse(x_bar, P_bar, x_hat, P, x_ref, C, R, delta, h_x_hat, h_x_bar, h_x_ref, angle_meas):
        mu = h_x_hat - h_x_bar
        Qe = np.dot(C, np.dot(P_bar, C.T)) + R
        alpha = h_x_ref - h_x_bar

        Qf = lambda x : 1 - normaldist.cdf(x)

        if angle_meas:
            nu_minus = normalize_angle(-delta + alpha - mu) / np.sqrt(Qe)
            nu_plus = normalize_angle(delta + alpha - mu) / np.sqrt(Qe)
        else:
            nu_minus = (-delta + alpha - mu) / np.sqrt(Qe)
            nu_plus = (delta + alpha - mu) / np.sqrt(Qe)

        tmp = (normaldist.pdf(nu_minus) - normaldist.pdf(nu_plus)) / (Qf(nu_minus) - Qf(nu_plus))
        z_bar = tmp * np.sqrt(Qe)
        tmp2 = (nu_minus * normaldist.pdf(nu_minus) - nu_plus*normaldist.pdf(nu_plus)) / (Qf(nu_minus) - Qf(nu_plus))
        curly_theta = tmp**2 - tmp2;
        K = np.dot(P, np.dot(C.T, inv( np.dot(C, np.dot(P, C.T)) + R)))
        x_hat = x_hat + K * z_bar
        P = P - curly_theta * np.dot(K, np.dot(C, P))

        return x_hat, P

    def _get_buffer_size(self, buffer_mat):
        """
        Calculates the number of explicit and implict measurements
        """

        index_col = MEAS_COLUMNS.index("type")
        num_explicit, num_implicit = 0, 0

        # Explicit: sonar range
        meas_type = MEAS_TYPES_INDICES.index("sonar_range")
        meas = buffer_mat[ np.where(buffer_mat[:,index_col] == meas_type)]
        num_explicit += meas.shape[0]

        # Explicit: sonar azimuth
        meas_type = MEAS_TYPES_INDICES.index("sonar_azimuth")
        meas = buffer_mat[ np.where(buffer_mat[:,index_col] == meas_type)]
        num_explicit += meas.shape[0]

        # Implicit: sonar range
        meas_type = MEAS_TYPES_INDICES.index("sonar_range_implicit")
        meas = buffer_mat[ np.where(buffer_mat[:,index_col] == meas_type)]
        num_implicit += meas.shape[0]

        # Implicit: sonar azimuth
        meas_type = MEAS_TYPES_INDICES.index("sonar_azimuth_implicit")
        meas = buffer_mat[ np.where(buffer_mat[:,index_col] == meas_type)]
        num_implicit += meas.shape[0]

        cost = num_explicit*EXPLICIT_BYTE_COST + num_implicit*IMPLICIT_BYTE_COST
        return cost, num_explicit, num_implicit

    @staticmethod
    def _get_ledger_mat(ledger):
        """
        Assume we can share any measurements in the ledger since they were taken by us OR are modem measurements...
        """
        return np.array(ledger)

    def _get_measurements_at_time(self, ledger_mat, index):
        """
        Returns all rows (measurements) in ledger_mat at the specified index
        ledger_mat produced by first calling self._get_ledger_mat()
        """
        index_col = MEAS_COLUMNS.index("index")
        return ledger_mat[ np.where(ledger_mat[:,index_col] == index)]

    @staticmethod
    def _covariance_intersection(xa, Pa, xb, Pb):
        """Runs covariance intersection on the two estimates A and B
        Arguments:
            xa {np.ndarray} -- mean of A
            Pa {np.ndarray} -- covariance of A
            xb {np.ndarray} -- mean of B
            Pb {np.ndarray} -- covariance of B
        
        Returns:
            c_bar {np.ndarray} -- intersected estimate
            Pcc {np.ndarray} -- intersected covariance
        """
        Pa_inv = np.linalg.inv(Pa)
        Pb_inv = np.linalg.inv(Pb)

        fxn = lambda omega: np.trace(np.linalg.inv(omega*Pa_inv + (1-omega)*Pb_inv))
        omega_optimal = scipy.optimize.minimize_scalar(fxn, bounds=(0,1), method="bounded").x

        Pcc = np.linalg.inv(omega_optimal*Pa_inv + (1-omega_optimal)*Pb_inv)
        c_bar = Pcc.dot( omega_optimal*Pa_inv.dot(xa) + (1-omega_optimal)*Pb_inv.dot(xb))
        return c_bar.reshape(-1,1), Pcc

    @staticmethod
    def _fast_covariance_intersection(xa, Pa, xb, Pb):
        Pa_inv = np.linalg.inv(Pa)
        Pb_inv = np.linalg.inv(Pb)

        omega_optimal = np.trace(Pb) / (np.trace(Pa) + np.trace(Pb))

        Pcc = np.linalg.inv(omega_optimal*Pa_inv + (1-omega_optimal)*Pb_inv)
        c_bar = Pcc.dot( omega_optimal*Pa_inv.dot(xa) + (1-omega_optimal)*Pb_inv.dot(xb))
        return c_bar.reshape(-1,1), Pcc

    def _get_agent_state_index(self, agent_num):
        if agent_num < self.BLUE_NUM + self.RED_NUM:
            return 6*agent_num
        else: # landmark
            remainder = agent_num - self.BLUE_NUM - self.RED_NUM
            return 6*(self.BLUE_NUM + self.RED_NUM) + 3*remainder
    
    def get_agent_index_from_state(self, state):
        return int( state / 6 )
    

if __name__ == "__main__":
    # Test scripts
    blue_positions = [[1,2,3],[4,5,6],[]]
    landmark_positions = [[1,2,3],[6,7,8]]
    
    kf = KalmanFilter(blue_positions, landmark_positions, red_agent=True, is_deltatier=True)
    kf.propogate(0.2, 0.1)
    kf.filter_artificial_depth(0.1)
    print("Success")
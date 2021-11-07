import numpy as np
from numpy.linalg import inv, norm
from normalize_angle import normalize_angle
import itertools
import scipy
import scipy.optimize

"""
Ledgers
- x_nav_history
- P_nav_history
- measurement
- x_history
- P_history

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
LANDMARK_UNCERTAINTY = 1e-3
KNOWN_POSITION_UNCERTAINTY = 1.0
KNOWN_VELOCITY_UNCERTAINTY = 1e-2
UNKNOWN_AGENT_UNCERTAINTY = 1e6 # Red and blue agents with unknown starting locations

# DELTATIER
MEAS_COLUMNS = ["type", "index", "start_x1", "start_x2", "data", "R"]
MEAS_TYPES_INDICES = ["modem_range", "modem_azimuth", "sonar_range", "sonar_azimuth", "sonar_range_implicit", "sonar_azimuth_implicit"]

class KalmanFilter:

    def __init__(self, blue_positions, landmark_positions, red_agent=False, is_deltatier=True):
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
                self.P.extend([UNKNOWN_AGENT_UNCERTAINTY]*3 + [KNOWN_VELOCITY_UNCERTAINTY]*3)
            else:
                self.x_hat.extend(p) # Position
                self.x_hat.extend([0]*3) # Velocity
                self.P.extend([KNOWN_POSITION_UNCERTAINTY]*3 + [KNOWN_VELOCITY_UNCERTAINTY]*3)

        if red_agent:
            self.x_hat.extend([0]*3) # Position
            self.x_hat.extend([0]*3) # Velocity
            self.P.extend([KNOWN_POSITION_UNCERTAINTY]*3 + [KNOWN_VELOCITY_UNCERTAINTY]*3)

        for p in landmark_positions:
            self.x_hat.extend(p) # Position (no velocity)
            self.P.extend([LANDMARK_UNCERTAINTY, LANDMARK_UNCERTAINTY, KNOWN_POSITION_UNCERTAINTY]) # great x,y knowledge, low z
        
        self.NUM_STATES = len(self.x_hat)
        self.x_hat = np.reshape(self.x_hat, (-1,1))
        self.P = np.diag(self.P)

        self.is_deltatier = is_deltatier
        if self.is_deltatier:
            self.x_hat_history = []
            self.P_history = []

            self.ledger = []

            self.x_common = self.x_hat
            self.P_common = self.P
            self.last_share_index = 0
            
            self.x_nav_history = []
            self.P_nav_history = []

    def propogate(self, position_process_noise, velocity_process_noise, delta_time=1.0):

        # Save the last estimate
        if self.is_deltatier:
            self.x_hat_history.append( np.copy(self.x_hat))
            self.P_history.append( np.copy(self.P))

        # Create F and Q matrices
        F = np.eye(self.NUM_STATES)
        Q = []
        for b in range(self.BLUE_NUM + self.RED_NUM):
            F[6*b, 6*b + 3] = 1
            F[6*b + 1, 6*b + 4] = 1
            F[6*b + 2, 6*b + 5] = 1
            Q.extend([position_process_noise]*3 + [velocity_process_noise]*3)
        for l in range(self.LANDMARK_NUM):
            Q.extend([0]*3)
        Q = np.diag(Q)

        F = F * delta_time
        Q = Q * delta_time

        self.x_hat = F @ self.x_hat
        self.P = F @ self.P @ F.T + Q

        self.index += 1
        return self.index

    def filter_artificial_depth(self, depth, R=1e-2):
        """
        Approximate the environment as 2D and fuse a certain depth for all agents/landmarks
        """
        for b in range(self.BLUE_NUM + self.RED_NUM):
            H = np.zeros((1, self.NUM_STATES))
            H[0, 6*b + 2] = 1

            x = self.x_hat
            P = self.P

            K = P @ H.T @ inv(H @ P @ H.T + R)
            x = x + K @ (depth - H @ x)
            P = P - K @ H @ P

            self.x_hat = x
            self.P = P

        num_dynamic_states = 6*(self.BLUE_NUM + self.RED_NUM)
        for l in range(self.LANDMARK_NUM):
            H = np.zeros((1, self.NUM_STATES))
            H[0, num_dynamic_states + 3*l + 2] = 1

            x = self.x_hat
            P = self.P

            K = P @ H.T @ inv(H @ P @ H.T + R)
            x = x + K @ (depth - H @ x)
            P = P - K @ H @ P

            self.x_hat = x
            self.P = P

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
    def filter_range_from_untracked(self, meas_value, R, position, collected_agent):
        startx1 = self._get_agent_state_index(collected_agent)
        self.x_hat, self.P = KalmanFilter._fuse_range_from_untracked(self.x_hat, self.P, startx1, position, meas_value, R)

        # Add to ledger
        type_ind = MEAS_TYPES_INDICES.index("modem_range")
        meas_row = [type_ind, self.index, startx1, -1, meas_value, R]
        self.ledger.append(meas_row)
        
    # Used in minau project for psuedo-gps measurements from surface beacon
    def filter_azimuth_from_untracked(self, meas_value, R, position, collected_agent):
        """
        Meas value must be relative to x-axis because of linear filter
        """
        startx1 = self._get_agent_state_index(collected_agent)
        self.x_hat, self.P = KalmanFilter._fuse_azimuth_from_untracked(self.x_hat, self.P, startx1, position, meas_value, R)

        # Add to ledger
        type_ind = MEAS_TYPES_INDICES.index("modem_azimuth")
        meas_row = [type_ind, self.index, startx1, -1, meas_value, R]
        self.ledger.append(meas_row)

    @staticmethod
    def _fuse_range_tracked(x_hat, P, startx1, startx2, meas_value, R):
        NUM_STATES = x_hat.shape[0]

        x1 = x_hat[startx1, 0]
        y1 = x_hat[startx1 + 1, 0]
        z1 = x_hat[startx1 + 2, 0]
        x2 = x_hat[startx2, 0]
        y2 = x_hat[startx2 + 1, 0]
        z2 = x_hat[startx2 + 2, 0]

        delta_pred = np.array([[x2 - x1], [y2 - y1], [z2 - z1]])
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

        K = P @ H.T @ inv(H @ P @ H.T + R)
        x_hat = x_hat + K * (meas_value - pred)
        P = P - K @ H @ P

        return x_hat, P

    @staticmethod
    def _fuse_azimuth_tracked(x_hat, P, startx1, startx2, meas_value, R):
        NUM_STATES = x_hat.shape[0]
        x1 = x_hat[startx1, 0]
        y1 = x_hat[startx1 + 1, 0]
        x2 = x_hat[startx2, 0]
        y2 = x_hat[startx2 + 1, 0]

        delta_pred = np.array([[x2 - x1], [y2 - y1]])
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

        K = P @ H.T @ inv(H @ P @ H.T + R)
        x_hat = x_hat + K * normalize_angle(meas_value - pred)
        P = P - K @ H @ P

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
        pred = norm(delta_pred)

        drdx1 = delta_pred[0] / norm(delta_pred)
        drdy1 = delta_pred[1] / norm(delta_pred)
        drdz1 = delta_pred[2] / norm(delta_pred)
        
        H = np.zeros((1, NUM_STATES))
        H[0, startx1] = drdx1
        H[0, startx1+1] = drdy1
        H[0, startx1+2] = drdz1

        K = P @ H.T @ inv(H @ P @ H.T + R)
        x_hat = x_hat + K * (meas_value - pred)
        P = P - K @ H @ P

        return x_hat, P

    @staticmethod
    def _fuse_azimuth_from_untracked(x_hat, P, startx1, position, meas_value, R):
        NUM_STATES = x_hat.shape[0]

        x1 = x_hat[startx1, 0]
        y1 = x_hat[startx1 + 1, 0]
        x2 = position[0]
        y2 = position[1]

        delta_pred = np.array([[x1 - x2],[y1 - y2]])
        pred = np.arctan2(delta_pred[1,0], delta_pred[0,0])

        dadx = -delta_pred[1,0] / norm(delta_pred)**2
        dady = delta_pred[0,0] / norm(delta_pred)**2
        H = np.zeros((1, NUM_STATES))
        H[0, startx1] = dadx
        H[0, startx1 + 1] = dady

        K = P @ H.T @ inv(H @ P @ H.T + R)
        x_hat = x_hat + K * normalize_angle(meas_value - pred)
        P = P - K @ H @ P

        return x_hat, P

    def pull_buffer(self, deltas, agent):

        # Pick the middle delta
        delta = deltas[ int( len(deltas) / 2 ) ]



        pass

    def rx_buffer(self, buffer, modem_loc):
        pass

    def intersect_strapdown(self, x_nav, P_nav, agent, fast_ci=False):

        self.x_nav_history.append(x_nav)
        self.P_nav_history.append(P_nav)

        # Go 8 -> 6
        rot_mat_nav = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0]]
        )
        nav_est_x = rot_mat_nav @ x_nav
        nav_est_P = rot_mat_nav @ P_nav @ rot_mat_nav.T

        rot_mat = np.zeros((6, self.x_hat.shape[0]))
        rot_mat[:,6*agent:6*(agent+1)] = np.eye(6)
        x_hat_agent = rot_mat @ self.x_hat
        P_agent = rot_mat @ self.P @ rot_mat.T

        if fast_ci:
            mean_result, cov_result = KalmanFilter._fast_covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)
        else:
            mean_result, cov_result = KalmanFilter._covariance_intersection(x_hat_agent, P_agent, nav_est_x, nav_est_P)

        # PSCI for tracking filter
        D_inv = inv(cov_result) - inv(P_agent)
        D_inv_d = inv(cov_result) @ mean_result - inv(P_agent) @ x_hat_agent
        D_inv_zeros = rot_mat.T @ D_inv @ rot_mat
        D_inv_d_zeros = rot_mat.T @ D_inv_d
        P_new = inv( inv(self.P) + D_inv_zeros)
        self.x_hat = P_new @ (inv(self.P) @ self.x_hat + D_inv_d_zeros )
        self.P = P_new

        # PSCI for navigation Filter (just X,Y)
        rot_mat = np.zeros((2, 8))
        rot_mat[:2,:2] = np.eye(2)
        mean_result = mean_result[:2, 0].reshape(-1,1)
        cov_result = cov_result[:2, :2]
        x_nav_position = rot_mat @ x_nav
        P_nav_position = rot_mat @ P_nav @ rot_mat.T
        
        D_inv = inv(cov_result) - inv(P_nav_position)
        D_inv_d = inv(cov_result) @ mean_result - inv(P_nav_position) @ x_nav_position
        D_inv_zeros = rot_mat.T @ D_inv @ rot_mat
        D_inv_d_zeros = rot_mat.T @ D_inv_d
        P_nav_new = inv( inv(P_nav) + D_inv_zeros )
        x_nav = P_nav_new @ ( inv(P_nav) @ x_nav + D_inv_d_zeros )
        P_nav = P_nav_new

        return x_nav, P_nav

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
        pass

    def _get_agent_state_index(self, agent_num):
        if agent_num < self.BLUE_NUM + self.RED_NUM:
            return 6*agent_num
        else: # landmark
            remainder = agent_num - self.BLUE_NUM + self.RED_NUM
            return 6*(self.BLUE_NUM + self.RED_NUM) + 3*remainder
    

if __name__ == "__main__":
    # Test scripts
    blue_positions = [[1,2,3],[4,5,6],[]]
    landmark_positions = [[1,2,3],[6,7,8]]
    
    kf = KalmanFilter(blue_positions, landmark_positions, red_agent=True, is_deltatier=True)
    kf.propogate(0.2, 0.1)
    kf.filter_artificial_depth(0.1)
    print("Success")
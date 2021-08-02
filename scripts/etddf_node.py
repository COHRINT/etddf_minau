#!/usr/bin/env python
from __future__ import division
"""@package etddf

ROS interface script for delta tiering filter

Filter operates in ENU

"""

from etddf.delta_tier import DeltaTier
from etddf.most_recent import MostRecent
import rospy
import threading
from minau.msg import ControlStatus
from etddf_minau.msg import Measurement, MeasurementPackage, NetworkEstimate, AssetEstimate
from etddf_minau.srv import GetMeasurementPackage
import numpy as np
import tf
np.set_printoptions(suppress=True)
from copy import deepcopy
from std_msgs.msg import Header, Float64, Int64
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion, Twist, Vector3, TwistWithCovariance, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from minau.msg import SonarTargetList, SonarTarget
from cuprint.cuprint import CUPrint

__author__ = "Luke Barbier"
__copyright__ = "Copyright 2020, COHRINT Lab"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"
__license__ = "MIT"
__maintainer__ = "Luke Barbier"
__version__ = "3.0"

NUM_OWNSHIP_STATES = 6

class ETDDF_Node:

    def __init__(self, 
                my_name, \
                update_rate, \
                delta_tiers, \
                asset2id, \
                delta_codebook_table, \
                buffer_size, \
                meas_space_table, \
                x0,\
                P0,\
                Q,\
                default_meas_variance,
                use_control_input):

        # self.br = tf.TransformBroadcaster()

        self.update_rate = update_rate
        self.asset2id = asset2id
        self.Q = Q
        self.use_control_input = use_control_input
        self.default_meas_variance = default_meas_variance
        self.my_name = my_name
        self.landmark_dict = rospy.get_param("~landmarks", {})

        self.cuprint = CUPrint(rospy.get_name())
        
        if rospy.get_param("~simple_sharing"):
            self.cuprint("Sharing most recent")
            self.filter = MostRecent(NUM_OWNSHIP_STATES, \
                                    x0,\
                                    P0,\
                                    buffer_size,\
                                    meas_space_table,\
                                    delta_codebook_table,\
                                    delta_tiers,\
                                    self.asset2id,\
                                    my_name,
                                    default_meas_variance
            )
        else:
            self.cuprint("Delta Tiering")
            self.filter = DeltaTier(NUM_OWNSHIP_STATES, \
                                    x0,\
                                    P0,\
                                    buffer_size,\
                                    meas_space_table,\
                                    delta_codebook_table,\
                                    delta_tiers,\
                                    self.asset2id,\
                                    my_name, \
                                    default_meas_variance)
        
        self.network_pub = rospy.Publisher("etddf/estimate/network", NetworkEstimate, queue_size=10)

        self.asset_pub_dict = {}
        for asset in self.asset2id.keys():
            if "topside" in asset:
                continue
            self.asset_pub_dict[asset] = rospy.Publisher("etddf/estimate/" + asset, Odometry, queue_size=10)        

        self.update_seq = 0
        self.last_depth_meas = None
        rospy.sleep(rospy.Duration(1 / self.update_rate))
        self.last_update_time = rospy.get_rostime() - rospy.Duration(1 / self.update_rate)
        self.meas_lock = threading.Lock()
        self.update_lock = threading.Lock()
        self.last_orientation = None
        self.red_asset_found = False
        self.red_asset_names = rospy.get_param("~red_team_names")

        # Modem & Measurement Packages
        rospy.Subscriber("etddf/packages_in", MeasurementPackage, self.meas_pkg_callback, queue_size=1)

        if self.use_control_input:
            raise NotImplementedError("Control input")
            self.control_input = None
            rospy.Subscriber("uuv_control/control_status", ControlStatus, self.control_status_callback, queue_size=1)

        if rospy.get_param("~strapdown_topic") != "None":
            self.cuprint("Intersecting with strapdown")
            rospy.Subscriber( rospy.get_param("~strapdown_topic"), Odometry, self.nav_filter_callback, queue_size=1)
            # Set up publisher for correcting the odom estimate
            self.intersection_pub = rospy.Publisher("set_pose", PoseWithCovarianceStamped, queue_size=1)
            rospy.wait_for_message( rospy.get_param("~strapdown_topic"), Odometry)
        else:
            self.cuprint("Not intersecting with strapdown filter")
            rospy.Timer(rospy.Duration(1 / self.update_rate), self.no_nav_filter_callback)
        
        # Initialize Buffer Service
        rospy.Service('etddf/get_measurement_package', GetMeasurementPackage, self.get_meas_pkg_callback)

        # Wait for our first strapdown msg
        self.cuprint("loaded, sleeping for RL to correct...")
        rospy.sleep(15) # Wait for RL to correct
        self.cuprint("Finally loaded")

        # Sonar Subscription
        if rospy.get_param("~measurement_topics/sonar") != "None":
            self.cuprint("Subscribing to sonar")
            rospy.Subscriber(rospy.get_param("~measurement_topics/sonar"), SonarTargetList, self.sonar_callback)

        

    def orientation_estimate_callback(self, odom):
        self.meas_lock.acquire()
        
        self.meas_lock.release()

    def sonar_callback(self, sonar_list):
        self.cuprint("Receiving sonar meas!!")
        self.update_lock.acquire()
        for target in sonar_list.targets:
            # self.cuprint("Receiving sonar measurements")
            if self.last_orientation is None: # No orientation, no linearization of the sonar measurement
                print("no ori")
                self.update_lock.release()
                return
            if target.id == "detection":
                continue

            # self.cuprint("Receiving sonar data")
            # Convert quaternions to Euler angles.
            self.meas_lock.acquire()
            (r, p, y) = tf.transformations.euler_from_quaternion([self.last_orientation.x, \
                self.last_orientation.y, self.last_orientation.z, self.last_orientation.w])
            self.meas_lock.release()
            # y = (np.pi/180.0) * 8
            bearing_world = y + target.bearing_rad

            z = target.range_m * np.sin(target.elevation_rad)
            xy_dist = target.range_m * np.cos(target.elevation_rad)
            x = xy_dist * np.cos(bearing_world)
            y = xy_dist * np.sin(bearing_world)

            now = rospy.get_rostime()
            sonar_x, sonar_y = None, None

            if "landmark_" in target.id:
                sonar_x = Measurement("sonar_x", now, self.my_name, "", x, self.default_meas_variance["sonar_x"], self.landmark_dict[target.id[len("landmark_"):]], -1.0)
                sonar_y = Measurement("sonar_y", now, self.my_name, "", y, self.default_meas_variance["sonar_x"], self.landmark_dict[target.id[len("landmark_"):]], -1.0)
            else:
                sonar_x = Measurement("sonar_x", now, self.my_name, target.id, x, self.default_meas_variance["sonar_x"], [], -1.0)
                sonar_y = Measurement("sonar_y", now, self.my_name, target.id, y, self.default_meas_variance["sonar_y"], [], -1.0)

                if target.id in self.red_asset_names and not self.red_asset_found:
                    self.cuprint("Red Asset detected!")
                    self.red_asset_found = True
            sonar_z = Measurement("sonar_z", now, self.my_name, target.id, 0, self.default_meas_variance["sonar_z"], [], -1.0)

            self.filter.add_meas(sonar_x)
            self.filter.add_meas(sonar_y)
            self.filter.add_meas(sonar_z)
        self.cuprint("sonar meas added")
        self.update_lock.release()

    def all_assets_same_plane(self):
        now = rospy.get_rostime()
        for a in self.asset2id:
            if a != self.my_name:                
                sonar_z = Measurement("sonar_z", now, self.my_name, a, 0, self.default_meas_variance["sonar_z"], [], -1.0)
                self.filter.add_meas(sonar_z)


    def no_nav_filter_callback(self, event):
        t_now = rospy.get_rostime()
        delta_t_ros =  t_now - self.last_update_time
        self.update_lock.acquire()

        u = np.zeros((3,1))
        Q = self.Q
        self.all_assets_same_plane()
        self.filter.update(t_now, u, Q, None, None)
        
        self.publish_estimates(t_now)
        self.last_update_time = t_now
        self.update_seq += 1
        self.update_lock.release()

    def nav_filter_callback(self, odom):
        # Update at specified rate
        t_now = rospy.get_rostime()
        delta_t_ros =  t_now - self.last_update_time
        if delta_t_ros < rospy.Duration(1/self.update_rate):
            return

        # Update orientation
        self.last_orientation = odom.pose.pose.orientation
        self.last_orientation_cov = np.array(odom.pose.covariance).reshape(6,6)
        self.last_orientation_dot = odom.twist.twist.angular
        self.last_orientation_dot_cov = np.array(odom.twist.covariance).reshape(6,6)

        self.update_lock.acquire()

        u = np.zeros((3,1))
        Q = self.Q

        # Add Bruce's position
        # gps_x = Measurement("gps_x", t_now, "bluerov2_7", "", 0.0, self.default_meas_variance["gps_x"], [], -1.0)
        # gps_y = Measurement("gps_y", t_now, "bluerov2_7", "", 0.0, self.default_meas_variance["gps_y"], [], -1.0)
        # gps_z = Measurement("depth", t_now, "bluerov2_7", "", 0.5, self.default_meas_variance["gps_x"], [], -1.0)
        # self.filter.add_meas(gps_x)
        # self.filter.add_meas(gps_y)
        # self.filter.add_meas(gps_z)

        # Turn odom estimate into numpy
        # Note the velocities are in the base_link frame --> Transform to odom frame # Assume zero pitch/roll
        v_baselink = np.array([[odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z]]).T
        (r, p, y) = tf.transformations.euler_from_quaternion([self.last_orientation.x, \
                self.last_orientation.y, self.last_orientation.z, self.last_orientation.w])
        rot_mat = np.array([ # base_link to odom frame
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y),  0],
            [0,         0,          1]
            ])
        v_odom = rot_mat.dot( v_baselink )

        mean = np.array([[odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z, \
                        v_odom[0,0], v_odom[1,0], v_odom[2,0]]]).T
        cov_pose = np.array(odom.pose.covariance).reshape(6,6)
        cov_twist = np.array(odom.twist.covariance).reshape(6,6)
        cov = np.zeros((6,6))
        cov[:3,:3] = cov_pose[:3,:3] #+ np.eye(3) * 4 #sim
        cov[3:,3:] = rot_mat.dot( cov_twist[:3,:3] ).dot( rot_mat.T ) #+ np.eye(3) * 0.03 #sim

        self.all_assets_same_plane()
        c_bar, Pcc = self.filter.update(t_now, u, Q, mean, cov)

        if c_bar is not None and Pcc is not None and self.update_seq % 10 == 0:
            # Correct the odom estimate
            # msg = PoseWithCovarianceStamped()
            # msg.header = odom.header
            # msg.header.frame_id = "odom"

            # # Transform
            # mean -= transform
            # msg.pose.pose.position.x = c_bar[0,0]
            # msg.pose.pose.position.y = c_bar[1,0]
            # msg.pose.pose.position.z = c_bar[2,0]
            # msg.pose.pose.orientation = self.last_orientation
            # new_cov = np.zeros((6,6))
            # new_cov[:3,:3] = Pcc[:3,:3] # TODO add full cross correlations
            # new_cov[3:,3:] = self.last_orientation_cov[3:,3:]

            # msg.pose.covariance = list(new_cov.flatten())
            # self.intersection_pub.publish( msg )
            pass

        self.publish_estimates(t_now)
        self.last_update_time = t_now
        self.update_seq += 1
        self.update_lock.release()
    
    def control_status_callback(self, msg):
        self.update_lock.acquire()
        if msg.is_setpoint_active and msg.is_heading_velocity_setpoint_active:
            self.control_input = np.array([[msg.setpoint_velocity.y, msg.setpoint_velocity.z, -msg.setpoint_velocity.z]]).T
        else:
            self.control_input = None
        # GRAB CONTROL INPUT
        self.update_lock.release()

    def depth_callback(self, msg):
        self.meas_lock.acquire()
        self.last_depth_meas = msg.data
        self.meas_lock.release()

    def publish_estimates(self, timestamp):
        ne = NetworkEstimate()
        for asset in self.asset2id.keys():
            if "topside" in asset:
                continue
            if "red" in asset and not self.red_asset_found:
                continue
            # else:
            #     print("publishing " + asset + "'s estimate")

            # Construct Odometry Msg for Asset

            mean, cov = self.filter.get_asset_estimate(asset)
            pose_cov = np.zeros((6,6))
            pose_cov[:3,:3] = cov[:3,:3]
            if asset == self.my_name:
                pose = Pose(Point(mean[0],mean[1],mean[2]), \
                            self.last_orientation)
                pose_cov[3:,3:] = self.last_orientation_cov[3:,3:]
            else:
                pose = Pose(Point(mean[0],mean[1],mean[2]), \
                            Quaternion(0,0,0,1))
                pose_cov[3:,3:] = np.eye(3) * 3
            pwc = PoseWithCovariance(pose, list(pose_cov.flatten()))

            twist_cov = np.zeros((6,6))
            twist_cov[:3,:3] = cov[3:6,3:6]
            if asset == self.my_name:
                tw = Twist(Vector3(mean[3],mean[4],mean[5]), self.last_orientation_dot)
                twist_cov[3:, 3:] = self.last_orientation_dot_cov[3:,3:]
            else:
                tw = Twist(Vector3(mean[3],mean[4],mean[5]), Vector3(0,0,0))
                twist_cov[3:, 3:] = np.eye(3) * -1
            twc = TwistWithCovariance(tw, list(twist_cov.flatten()))
            h = Header(self.update_seq, timestamp, "map")
            o = Odometry(h, "map", pwc, twc)

            ae = AssetEstimate(o, asset)
            ne.assets.append(ae)
            self.asset_pub_dict[asset].publish(o)

        self.network_pub.publish(ne)

        # Publish transform
        # self.br.sendTransform(
        #     self.odom2map_tf,
        #     (0,0,0,1),
        #     timestamp,
        #     "map",
        #     "odom")

    def meas_pkg_callback(self, msg):
        self.update_lock.acquire()
        # Modem Meas taken by topside
        
        if msg.src_asset == "topside":
            self.cuprint("Receiving Surface Modem Measurements")
            modem_indices = []
            for meas in msg.measurements:
                # Approximate the fuse on the next update, so we can get other asset's position immediately
                if meas.meas_type == "modem_elevation":
                    rospy.logerr("Ignoring Modem Elevation Measurement since we have depth measurements")
                    continue
                elif meas.meas_type == "modem_azimuth":
                    meas.global_pose = list(meas.global_pose)
                    # self.cuprint("azimuth: " + str(meas.data))
                    meas.data = (meas.data * np.pi) / 180
                    meas.variance = self.default_meas_variance["modem_azimuth"]
                elif meas.meas_type == "modem_range":
                    meas.global_pose = list(meas.global_pose)
                    # self.cuprint("range: " + str(meas.data))
                    meas.variance = self.default_meas_variance["modem_range"]
                ind = self.filter.add_meas(meas, common=True)
                modem_indices.append(ind)
            self.filter.catch_up(min(modem_indices))

        # Buffer
        else:
            self.cuprint("receiving buffer")
            # Loop through buffer and see if we've found the red agent
            for i in range(len(msg.measurements)):
                if msg.measurements[i].measured_asset in self.red_asset_names and not self.red_asset_found:
                    self.red_asset_found = True
                    self.cuprint("Red asset measurement received!")
            
            implicit_cnt, explicit_cnt = self.filter.receive_buffer(msg.measurements, msg.delta_multiplier, msg.src_asset)
            
            # implicit_cnt, explicit_cnt = self.filter.catch_up(msg.delta_multiplier, msg.measurements, self.Q, msg.all_measurements)

        self.update_lock.release()
        self.cuprint("Finished")

    def get_meas_pkg_callback(self, req):
        self.cuprint("pulling buffer")
        self.update_lock.acquire()
        delta, buffer = self.filter.pull_buffer()
        self.update_lock.release()
        mp = MeasurementPackage(buffer, buffer, self.my_name, delta)
        self.cuprint("returning buffer")
        return mp

################################
### Initialization Functions ###
################################

def get_indices_from_asset_names(blue_team):
    my_name = rospy.get_param("~my_name")
    red_team = rospy.get_param("~red_team_names")
    asset2id = {}
    asset2id[my_name] = 0

    next_index = 1
    for asset in blue_team:
        if asset == my_name:
            continue
        else:
            asset2id[asset] = next_index
            next_index += 1
    for asset in red_team:
        asset2id[asset] = next_index
        next_index += 1

    if my_name != "topside":
        asset2id["topside"] = -1 # arbitrary negative number

    return asset2id

def get_delta_codebook_table():
    delta_codebook = {}

    meas_info = rospy.get_param("~measurements")
    for meas in meas_info.keys():
        base_et_delta = meas_info[meas]["base_et_delta"]
        delta_codebook[meas] = base_et_delta
    return delta_codebook

def get_meas_space_table():
    meas_space_table = {}

    meas_info = rospy.get_param("~measurements")
    for meas in meas_info.keys():
        meas_space_table[meas] = meas_info[meas]["buffer_size"]

    meas_space_table["burst"] = rospy.get_param("~buffer_space/burst")

    return meas_space_table

def _dict2arr(d):
    return np.array([[d["x"]],\
                    [d["y"]],\
                    [d["z"]],\
                    [d["x_vel"]], \
                    [d["y_vel"]],\
                    [d["z_vel"]]])
def _list2arr(l):
    return np.array([l]).reshape(-1,1)

def _add_velocity_states(base_states):
    velocities = np.zeros((base_states.size,1))
    return np.concatenate((base_states, velocities), axis=0)

def get_initial_estimate(num_states, blue_team_names, blue_team_positions):
    default_starting_position = _dict2arr(rospy.get_param("~default_starting_position"))
    uncertainty_known_starting_position = _dict2arr( rospy.get_param("~initial_uncertainty/known_starting_position"))
    uncertainty_unknown_starting_position = _dict2arr( rospy.get_param("~initial_uncertainty/unknown_starting_position"))

    my_starting_position = rospy.get_param("~starting_position")
    if not my_starting_position:
        my_starting_position = deepcopy(default_starting_position)
    else:
        my_starting_position = _add_velocity_states( _list2arr(my_starting_position))
    ownship_uncertainty = _dict2arr( rospy.get_param("~initial_uncertainty/ownship") )

    uncertainty = np.zeros((num_states,num_states))
    uncertainty_vector = np.zeros((num_states,1))
    uncertainty_vector[:NUM_OWNSHIP_STATES] = ownship_uncertainty
    uncertainty += np.eye(num_states) * uncertainty_vector

    state_vector = my_starting_position
    my_name = rospy.get_param("~my_name")
    red_team_names = rospy.get_param("~red_team_names")

    next_index_unc = 1
    next_index_pos = 1
    for asset in blue_team_names:
        if asset == my_name:
            next_index_pos += 1
            continue
        if len(blue_team_positions) >= next_index_pos: # we were given the positione of this asset in roslaunch
            next_position = _add_velocity_states( _list2arr( blue_team_positions[next_index_pos-1]))
            uncertainty_vector = np.zeros((num_states,1))
            uncertainty_vector[next_index_unc*NUM_OWNSHIP_STATES:(next_index_unc+1)*NUM_OWNSHIP_STATES] = uncertainty_known_starting_position
            uncertainty += np.eye(num_states) * uncertainty_vector
        else:
            next_position = deepcopy(default_starting_position)
            uncertainty_vector = np.zeros((num_states,1))
            uncertainty_vector[next_index_unc*NUM_OWNSHIP_STATES:(next_index_unc+1)*NUM_OWNSHIP_STATES] = uncertainty_unknown_starting_position
            uncertainty += np.eye(num_states) * uncertainty_vector

        state_vector = np.concatenate((state_vector, next_position),axis=0)
        next_index_unc += 1
        next_index_pos += 1
    for asset in red_team_names:
        next_position = deepcopy(default_starting_position)
        state_vector = np.concatenate((state_vector, next_position),axis=0)

        uncertainty_vector = np.zeros((num_states,1))
        uncertainty_vector[next_index_unc*NUM_OWNSHIP_STATES:(next_index_unc+1)*NUM_OWNSHIP_STATES] = uncertainty_unknown_starting_position
        uncertainty += np.eye(num_states) * uncertainty_vector

        next_index_unc += 1
    
    return state_vector, uncertainty

def get_process_noise(num_states, blue_team_names):
    Q = np.zeros((num_states, num_states))
    ownship_Q = _dict2arr(rospy.get_param("~process_noise/ownship"))
    blueteam_Q = _dict2arr(rospy.get_param("~process_noise/blueteam"))
    redteam_Q = _dict2arr(rospy.get_param("~process_noise/redteam"))

    Q_vec = np.zeros((num_states,1))
    Q_vec[:NUM_OWNSHIP_STATES] = ownship_Q
    Q += np.eye(num_states) * Q_vec

    my_name = rospy.get_param("~my_name")
    red_team_names = rospy.get_param("~red_team_names")

    next_index = 1
    for asset in blue_team_names:
        if asset == my_name:
            continue
        Q_vec = np.zeros((num_states,1))
        Q_vec[next_index*NUM_OWNSHIP_STATES:(next_index+1)*NUM_OWNSHIP_STATES] = blueteam_Q
        Q += np.eye(num_states) * Q_vec
        next_index += 1
    for asset in red_team_names:
        Q_vec = np.zeros((num_states,1))
        Q_vec[next_index*NUM_OWNSHIP_STATES:(next_index+1)*NUM_OWNSHIP_STATES] = redteam_Q
        Q += np.eye(num_states) * Q_vec
        next_index += 1
    return Q

def get_default_meas_variance():
    meas_vars = {}
    meas_info = rospy.get_param("~measurements")
    for meas in meas_info.keys():
        sd = meas_info[meas]["default_sd"]
        meas_vars[meas] = sd ** 2
    return meas_vars

if __name__ == "__main__":
    rospy.init_node("etddf_node")
    my_name = rospy.get_param("~my_name")
    update_rate = rospy.get_param("~update_rate")
    delta_tiers = rospy.get_param("~delta_tiers")
    blue_team_names = rospy.get_param("~blue_team_names")
    blue_team_positions = rospy.get_param("~blue_team_positions")

    # Don't track topside if it isn't this agent
    if my_name != "topside" and "topside" in blue_team_names:
        ind = blue_team_names.index("topside")
        if ind >= 0:
            blue_team_names.pop(ind)
            blue_team_positions.pop(ind)


    asset2id = get_indices_from_asset_names(blue_team_names)
    delta_codebook_table = get_delta_codebook_table()
    buffer_size = rospy.get_param("~buffer_space/capacity")
    meas_space_table = get_meas_space_table()
    if my_name != "topside":
        num_assets = len(asset2id) - 1 # subtract topside
    else:
        num_assets = len(asset2id)
    x0, P0 = get_initial_estimate(num_assets * NUM_OWNSHIP_STATES, blue_team_names, blue_team_positions)
    Q = get_process_noise(num_assets * NUM_OWNSHIP_STATES, blue_team_names)
    # rospy.logwarn("{}, {}, {}, {}".format(my_name, x0.shape, P0.shape, Q.shape))
    default_meas_variance = get_default_meas_variance()
    use_control_input = rospy.get_param("~use_control_input")

    et_node = ETDDF_Node(my_name,
                        update_rate, \
                        delta_tiers, \
                        asset2id, \
                        delta_codebook_table, \
                        buffer_size, \
                        meas_space_table, \
                        x0,\
                        P0,\
                        Q,\
                        default_meas_variance,\
                        use_control_input)

    rospy.spin()
#!/usr/bin/env python
from __future__ import division

"""@package etddf

ROS interface script for delta tiering filter

Filter operates in ENU

steps: get this to at least launch by itself
verify it works in sim for static sonar (fast scan) & dynamic agent -> plot the error (associator, no sonar control)
check the controller works statically - may need a correction here

"""

import rospy
from etddf_minau.msg import MeasurementPackage, NetworkEstimate, AssetEstimate, Measurement
from etddf_minau.srv import GetMeasurementPackage
import numpy as np
import tf
np.set_printoptions(suppress=True)
from copy import deepcopy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion, Twist, Vector3, TwistWithCovariance, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from minau.msg import SonarTargetList, SonarTarget
from cuprint.cuprint import CUPrint
from deltatier.kf_filter import KalmanFilter, MEAS_TYPES_INDICES, MEAS_COLUMNS

class ETDDF_Node:

    def __init__(self):
        self.my_name = rospy.get_param("~my_name")
        self.cuprint = CUPrint("{}/etddf".format(self.my_name))
        self.blue_agent_names = rospy.get_param("~blue_team_names")
        blue_positions = rospy.get_param("~blue_team_positions")

        self.topside_name = rospy.get_param("~topside_name")
        assert self.topside_name not in self.blue_agent_names

        red_agent_name = rospy.get_param("~red_team_name")

        self.update_times = []
        
        self.red_agent_exists = red_agent_name != ""
        if self.red_agent_exists:
            self.red_agent_name = red_agent_name
            self.red_agent_id = len(self.blue_agent_names)

        self.use_strapdown = rospy.get_param("~use_strapdown")
        self.do_correct_strapdown = rospy.get_param("~correct_strapdown")
        self.correct_strapdown_next_seq = False
        self.position_process_noise = rospy.get_param("~position_process_noise")
        self.velocity_process_noise = rospy.get_param("~velocity_process_noise")
        self.fast_ci = rospy.get_param("~fast_ci")
        self.force_modem_pose = rospy.get_param("~force_modem_pose")
        self.meas_variances = {}
        self.meas_variances["sonar_range"] = rospy.get_param("~force_sonar_range_var")
        self.meas_variances["sonar_az"] = rospy.get_param("~force_sonar_az_var")
        self.meas_variances["modem_range"] = rospy.get_param("~force_modem_range_var")
        self.meas_variances["modem_az"] = rospy.get_param("~force_modem_az_var")

        known_position_uncertainty = rospy.get_param("~known_position_uncertainty")
        unknown_position_uncertainty = rospy.get_param("~unknown_position_uncertainty")
        
        self.is_deltatier = rospy.get_param("~is_deltatier")
        if self.is_deltatier:
            self.delta_multipliers = rospy.get_param("~delta_tiers")
            self.delta_codebook_table = {"sonar_range" : rospy.get_param("~sonar_range_start_et_delta"),
                                         "sonar_azimuth" : rospy.get_param("~sonar_az_start_et_delta")}
            self.buffer_size = rospy.get_param("~buffer_space")

        if self.is_deltatier:
            rospy.Service('etddf/get_measurement_package', GetMeasurementPackage, self.get_meas_pkg_callback)

        self.kf = KalmanFilter(blue_positions, [], self.red_agent_exists, self.is_deltatier, \
            known_posititon_unc=known_position_uncertainty,\
            unknown_agent_unc=unknown_position_uncertainty)
        
        # Initialize the kalman filter to the correct starting estimate
        x_hat = np.array([3.89131562419, -1.81523517305, -0.609213583867, 0.0590610492218, -0.0123524654153, 0.00180473321339, \
                                3.38198297047, 0.159090854054, -0.610862394329, 0.0366285762217, 0.00680334780657, -0.00342542529857])
        blue7_pos_cov = [4.066656067779953, -1.5404561786027737, -6.516080657320234e-07, 0.0, 0.0, 0.0, -1.5404561786027724, 1.3353038668609196, -7.152718624737255e-06, 0.0, 0.0, 0.0, -6.516080657320111e-07, -7.15271862473727e-06, 0.008369161321799212, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04041716405632286, -0.0005849874715729819, -5.589835383432124e-07, 0.0, 0.0, 0.0, -0.0005849874715729821, 0.0157834857305976, 2.1024579596438633e-08, 0.0, 0.0, 0.0, -5.589835383432123e-07, 2.1024579596438752e-08, 0.021678826527810384]
        blue7_vel_cov = [0.006970526495199368, -0.0010569644397999919, -8.688092559903357e-09, -0.0, -0.0, -0.0, -0.0010569644397999875, 0.004757777386387238, -2.2034527050738752e-08, -0.0, -0.0, -0.0, -8.68809255990197e-09, -2.203452705074336e-08, 0.0034241466326507017, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0]
        blue5_pos_cov = [3.026238619036268, -1.4324979041230894, -5.522027094957632e-06, 0.0, 0.0, 0.0, -1.43249790412309, 1.3819694062935084, -1.2752419419034878e-06, 0.0, 0.0, 0.0, -5.52202709495763e-06, -1.2752419419034922e-06, 0.009170215343487253, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -1.0]
        blue5_vel_cov = [0.0008331419165405064, -5.9810235723929415e-05, -2.108120588498449e-10, -0.0, -0.0, -0.0, -5.981023572392958e-05, 0.0006655106386352285, -4.353339389205928e-11, -0.0, -0.0, -0.0, -2.1081205884969358e-10, -4.3533393891775726e-11, 0.0005561217229920902, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0]

        blue7_pos_cov = np.reshape(blue7_pos_cov, (6,6))[:3,:3]
        blue7_vel_cov = np.reshape(blue7_vel_cov, (6,6))[:3,:3]
        blue5_pos_cov = np.reshape(blue5_pos_cov, (6,6))[:3,:3]
        blue5_vel_cov = np.reshape(blue5_vel_cov, (6,6))[:3,:3]
        cov = np.zeros((12,12))
        cov[:3,:3] = blue7_pos_cov
        cov[3:6,3:6] = blue7_vel_cov
        cov[6:9,6:9] = blue5_pos_cov
        cov[9:12,9:12] = blue5_vel_cov

        self.kf.x_hat = np.reshape(x_hat, (-1,1))
        self.kf.P = cov        

        self.network_pub = rospy.Publisher("etddf/estimate/network", NetworkEstimate, queue_size=10)
        self.asset_pub_dict = {}
        for asset in self.blue_agent_names:
            self.asset_pub_dict[asset] = rospy.Publisher("etddf/estimate/" + asset, Odometry, queue_size=10)
        if self.red_agent_exists:
            self.asset_pub_dict[self.red_agent_name] = rospy.Publisher("etddf/estimate/" + self.red_agent_name, Odometry, queue_size=10)

        self.last_update_time = rospy.get_rostime()

        # Modem & Measurement Packages
        rospy.Subscriber("etddf/packages_in", MeasurementPackage, self.meas_pkg_callback, queue_size=1)

        self.last_orientation_rad = None

        # Strapdown configuration
        self.update_seq = 0
        self.strapdown_correction_period = rospy.get_param("~strapdown_correction_period")
        strap_topic = "odometry/filtered/odom"
        rospy.Subscriber( strap_topic, Odometry, self.nav_filter_callback, queue_size=1)
        self.intersection_pub = rospy.Publisher("set_pose", PoseWithCovarianceStamped, queue_size=1)
        self.cuprint("Waiting for strapdown")
        rospy.wait_for_message( strap_topic, Odometry)
        self.cuprint("Strapdown found")

        # Sonar Subscription
        rospy.Subscriber("sonar_processing/target_list/associated", SonarTargetList, self.sonar_callback)
        self.cuprint("Loaded")

    def sonar_callback(self, msg):
        if self.last_orientation_rad == None:
            return
        # self.cuprint("Receiving sonar meas")
        collecting_agent_id = self.blue_agent_names.index(self.my_name)
        for st in msg.targets:
            collected_agent_id = self.blue_agent_names.index( st.id )
            range_meas = st.range_m
            azimuth_meas = st.bearing_rad + self.last_orientation_rad
            if self.meas_variances["sonar_range"] == -1:
                R_range = st.range_variance
            else:
                R_range = self.meas_variances["sonar_range"]
            if self.meas_variances["sonar_az"] == -1:
                R_az = st.bearing_variance
            else:
                R_az = self.meas_variances["sonar_az"]

            rounded_range_meas = round(range_meas, 1)
            rounded_azimuth_meas = round(np.degrees(azimuth_meas),1)
            # self.cuprint("{} r: {} az: {} (deg)".format(st.id, rounded_range_meas, rounded_azimuth_meas))

            self.kf.filter_azimuth_tracked(azimuth_meas, R_az, collecting_agent_id, collected_agent_id)
            self.kf.filter_range_tracked(range_meas, R_range, collecting_agent_id, collected_agent_id)

    def nav_filter_callback(self, odom):

        # Correct strapdown if first msg
        if self.last_orientation_rad is None:
            last_orientation_quat = odom.pose.pose.orientation
            (r, p, y) = tf.transformations.euler_from_quaternion([last_orientation_quat.x, \
                    last_orientation_quat.y, last_orientation_quat.z, last_orientation_quat.w])
            self.last_orientation_rad = y
            orientation_cov = np.array(odom.pose.covariance).reshape(6,6)
            self.correct_strapdown(odom.header, self.kf.x_hat, self.kf.P, last_orientation_quat, orientation_cov)
            return

        # Update at specified rate
        t_now = rospy.get_rostime()
        delta_t_ros =  t_now - self.last_update_time
        if delta_t_ros < rospy.Duration(1):
            return

        self.kf.propogate(self.position_process_noise, self.velocity_process_noise)
        self.update_times.append(t_now)

        # Update orientation
        last_orientation_quat = odom.pose.pose.orientation
        (r, p, y) = tf.transformations.euler_from_quaternion([last_orientation_quat.x, \
                last_orientation_quat.y, last_orientation_quat.z, last_orientation_quat.w])
        self.last_orientation_rad = y
        orientation_cov = np.array(odom.pose.covariance).reshape(6,6)

        if self.use_strapdown:
            # last_orientation_dot = odom.twist.twist.angular
            # last_orientation_dot_cov = np.array(odom.twist.covariance).reshape(6,6)

            # Turn odom estimate into numpy
            # Note the velocities are in the base_link frame --> Transform to odom frame # Assume zero pitch/roll
            v_baselink = np.array([[odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z]]).T
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

            my_id = self.blue_agent_names.index(self.my_name)
            x_nav, P_nav = self.kf.intersect_strapdown(mean, cov, my_id, fast_ci=False)

            if self.do_correct_strapdown and (self.update_seq % self.strapdown_correction_period == 0):
                if x_nav is not None and P_nav is not None:
                    self.correct_strapdown(odom.header, x_nav, P_nav, last_orientation_quat, orientation_cov)
            elif self.correct_strapdown_next_seq:
                self.correct_strapdown(odom.header, x_nav, P_nav, last_orientation_quat, orientation_cov)
                self.correct_strapdown_next_seq = False

        self.publish_estimates(t_now, last_orientation_quat, orientation_cov)
        self.last_update_time = t_now
        self.update_seq += 1

    def correct_strapdown(self, header, x_nav, P_nav, orientation, orientation_cov):
        msg = PoseWithCovarianceStamped()
        msg.header = header
        msg.header.frame_id = "odom"

        # Transform
        msg.pose.pose.position.x = x_nav[0,0]
        msg.pose.pose.position.y = x_nav[1,0]
        msg.pose.pose.position.z = x_nav[2,0]
        msg.pose.pose.orientation = orientation
        new_cov = np.zeros((6,6))
        new_cov[:3,:3] = P_nav[:3,:3] # TODO add full cross correlations
        new_cov[3:,3:] = orientation_cov[3:,3:]

        msg.pose.covariance = list(new_cov.flatten())
        self.intersection_pub.publish( msg )

    def publish_estimates(self, timestamp, last_orientation_quat, orientation_cov):
        ne = NetworkEstimate()
        for asset in self.blue_agent_names:
            
            ind = self.blue_agent_names.index(asset)
            x_hat_agent, P_agent, _ = self.kf.get_agent_states(ind)
            pose_cov = np.zeros((6,6))
            pose_cov[:3,:3] = P_agent[:3,:3]
            if asset == self.my_name:
                pose = Pose(Point(x_hat_agent[0],x_hat_agent[1],x_hat_agent[2]),last_orientation_quat)
                pose_cov[3:,3:] = orientation_cov[3:,3:]
            elif "red" in asset:
                pose_cov = 5*np.eye(6) # Just set single uncertainty
                red_agent_depth = -0.7
                pose = Pose(Point(x_hat_agent[0],x_hat_agent[1],red_agent_depth), Quaternion(0,0,0,1))
                pose_cov[3:,3:] = -np.eye(3)
            else:
                pose = Pose(Point(x_hat_agent[0],x_hat_agent[1],x_hat_agent[2]), Quaternion(0,0,0,1))
                pose_cov[3:,3:] = -np.eye(3)
            pwc = PoseWithCovariance(pose, list(pose_cov.flatten()))

            twist_cov = -np.eye(6)
            twist_cov[:3,:3] = P_agent[3:6,3:6]
            tw = Twist()
            tw.linear = Vector3(x_hat_agent[3],x_hat_agent[4],x_hat_agent[5])
            twc = TwistWithCovariance(tw, list(twist_cov.flatten()))
            
            h = Header(self.update_seq, timestamp, "odom")
            o = Odometry(h, "odom", pwc, twc)

            ae = AssetEstimate(o, asset)
            ne.assets.append(ae)
            self.asset_pub_dict[asset].publish(o)
        
        if self.red_agent_exists:
            asset = self.red_agent_name
            ind = self.blue_agent_names.index(asset)
            x_hat_agent, P_agent = self.kf.get_agent_states(ind)
            pose_cov[:3,:3] = P_agent[:3,:3]
            red_agent_depth = -0.7
            pose = Pose(Point(x_hat_agent[0],x_hat_agent[1],red_agent_depth), Quaternion(0,0,0,1))
            pose_cov[3:,3:] = -np.eye(3)
            pwc = PoseWithCovariance(pose, list(pose_cov.flatten()))
            twist_cov = -np.eye((6,6))
            twist_cov[:3,:3] = P_agent[3:6,3:6]
            tw = Twist()
            tw.linear = Vector3(x_hat_agent[3],x_hat_agent[4],x_hat_agent[5])
            twc = TwistWithCovariance(tw, list(twist_cov.flatten()))
            h = Header(self.update_seq, timestamp, "odom")
            o = Odometry(h, "odom", pwc, twc)
            ae = AssetEstimate(o, asset)
            ne.assets.append(ae)
            self.asset_pub_dict[asset].publish(o)

        self.network_pub.publish(ne)

    def meas_pkg_callback(self, msg):
        # Modem Meas taken by topside
        if msg.src_asset == self.topside_name:
            self.cuprint("Receiving Surface Modem Measurements")
            meas_indices = []
            modem_loc = None

            # Approximate all modem measurements as being taken at this time
            for meas in msg.measurements:
                if len(self.force_modem_pose) == 0:
                    modem_loc = meas.global_pose[:3]
                    modem_ori = meas.global_pose[3]
                else:
                    modem_loc = self.force_modem_pose[:3]
                    modem_ori = np.radians(self.force_modem_pose[3])
                # self.cuprint("Modem loc: {} Modem pose: {}".format(modem_loc, modem_ori))

                # meas_index = min(range(len(self.update_times)), key=lambda i: abs( (self.update_times[i]-meas.stamp).to_sec() ))
                meas_index = len(self.update_times) - 5
                if meas_index < 0:
                    meas_index = None
                meas_indices.append(meas_index)
                agent = meas.measured_asset
                agent_id = self.blue_agent_names.index(agent)

                # Approximate the fuse on the next update, so we can get other asset's position immediately
                if meas.meas_type == "modem_elevation":

                    rospy.logerr("Ignoring Modem Elevation Measurement since we have depth measurements")
                    continue

                elif meas.meas_type == "modem_azimuth" and agent != self.my_name:

                    meas.data += modem_ori
                    meas_value_rad = np.radians(meas.data)
                    R = self.meas_variances["modem_az"]
                    self.kf.filter_azimuth_from_untracked( meas_value_rad, R, modem_loc, agent_id, index=meas_index)

                elif meas.meas_type == "modem_range":
                    BIAS = 0.5
                    agent = meas.measured_asset
                    R = self.meas_variances["modem_range"]
                    self.kf.filter_range_from_untracked( meas.data - BIAS, R, modem_loc, agent_id, index=meas_index)

            if meas_indices and meas_index != None: # we received measurements
                min_index = min(meas_indices)
                my_id = self.blue_agent_names.index(self.my_name)
                self.kf.catch_up(min_index, modem_loc, self.position_process_noise, self.velocity_process_noise, my_id, fast_ci=False)
                self.correct_strapdown_next_seq = True

        elif self.is_deltatier:
            raise NotImplementedError("DT is not supported yet")
            # self.cuprint("receiving buffer")
            # # Loop through buffer and see if we've found the red agent
            # for i in range(len(msg.measurements)):
            #     if msg.measurements[i].measured_asset in self.red_asset_names and not self.red_asset_found:
            #         self.red_asset_found = True
            #         self.cuprint("Red asset measurement received!")
            
            # implicit_cnt, explicit_cnt = self.filter.receive_buffer(msg.measurements, msg.delta_multiplier, msg.src_asset)
            
            # implicit_cnt, explicit_cnt = self.filter.catch_up(msg.delta_multiplier, msg.measurements, self.Q, msg.all_measurements)

    def get_meas_pkg_callback(self, req):
        self.cuprint("pulling buffer")

        mult, share_buffer, explicit_cnt, implicit_cnt = self.kf.pull_buffer(
            self.delta_multipliers, 
            self.delta_codebook_table, 
            self.position_process_noise, 
            self.velocity_process_noise, 
            self.force_modem_pose[:3],
            self.buffer_size)
        print(mult)
        print(share_buffer)
        MEAS_COLUMNS = ["type", "index", "startx1", "startx2", "data", "R"]
        MEAS_TYPES_INDICES = ["modem_range", "modem_azimuth", "sonar_range", "sonar_azimuth", "sonar_range_implicit", "sonar_azimuth_implicit"]

        meas_list = []
        for m in share_buffer:
            meas_type = MEAS_TYPES_INDICES[ m[ MEAS_COLUMNS.index("type") ] ]
            index = m[ MEAS_COLUMNS.index("index") ]
            startx1 = m[ MEAS_COLUMNS.index("startx1") ]
            startx2 = m[ MEAS_COLUMNS.index("startx2") ]
            data  = m[ MEAS_COLUMNS.index("data") ]
            R = m[ MEAS_COLUMNS.index("R") ]
            x1_ind = self.kf.get_agent_index_from_state(startx1)
            x2_ind = self.kf.get_agent_index_from_state(startx2)
            agent1 = self.blue_agent_names[x1_ind]
            if x2_ind == len(self.blue_agent_names): # red agent
                agent2 = self.red_agent_name
            else:
                agent2 = self.blue_agent_names[x2_ind]
            new_meas = Measurement(meas_type, index, agent1, agent2, data, R, [], 0.0)
            meas_list.append(new_meas)

        mp = MeasurementPackage()
        mp.delta_multiplier = mult
        mp.src_asset = self.my_name
        mp.measurements = meas_list

        return mp

if __name__ == "__main__":
    rospy.init_node("etddf_node")
    et_node = ETDDF_Node()

    debug = False
    if not debug:
        rospy.spin()
    else:
        o = Odometry()
        o.pose.pose.orientation.w = 1
        et_node.use_strapdown = False
        rospy.sleep(2)
        t = rospy.get_rostime()
        et_node.nav_filter_callback(o)

        mp = MeasurementPackage()
        m = Measurement("modem_range", t, "topside", "guppy", 6, 0, [], 0)
        mp.measurements.append(m)
        m = Measurement("modem_azimuth", t, "topside", "guppy", 45, 0, [],0)
        mp.measurements.append(m)
        mp.src_asset = "topside"
        et_node.meas_pkg_callback(mp)

        rospy.sleep(1)
        et_node.kf._filter_artificial_depth(0.0)
        et_node.nav_filter_callback(o)

        stl = SonarTargetList()
        st = SonarTarget()
        st.id = "guppy"
        st.bearing_rad = np.random.normal(0,0.01)
        st.range_m = 5.0 + np.random.normal(0,0.1)
        stl.targets.append(st)
        et_node.sonar_callback(stl)
        rospy.sleep(1)
        et_node.nav_filter_callback(o)

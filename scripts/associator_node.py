#!/usr/bin/env python

from copy import deepcopy
import tf
import rospy
from nav_msgs.msg import Odometry
from minau.msg import SonarTargetList, SonarTarget
from ping360_sonar.msg import SonarSettings
from cuprint.cuprint import CUPrint
from deltatier.associator import Associator
import numpy as np
from std_msgs.msg import UInt16
from deltatier.sonar_controller import scan_control
from deltatier.normalize_angle import normalize_angle

"""
The architecture is a bit complex
intfc for sonar should be in gradians for consistency, need to convert on my end

We provide upper and lower bounds, SOO this node should be configured with the scan size

CHANGES TO SONAR NODE
1) Change from rosservice to pub/sub
2) Pub when complete a scan
3) Add Correct message type to this script for sonar configuration
4) HITL test

Scenario
1 configured for 360 degree scan
2 at some point we want to scan an agent
3 provide upper and lower bounds
4 I should check if anything changed and only update then? 

# TODO try first deltatier simple with stack

"""

class SonarAssociator:

    def __init__(self):
        
        self.my_name = rospy.get_namespace()[:-1].strip("/")
        self.cuprint = CUPrint("{}/associator_node".format(self.my_name))

        # Associator Params
        time_to_drop = rospy.get_param("~time_to_drop")
        self.lost_agent_unc = rospy.get_param("~lost_agent_unc")
        proto_track_points = rospy.get_param("~proto_track_points")
        process_noise = rospy.get_param("~position_process_noise")
        proto_Q = np.array([[process_noise, 0],[0, process_noise]])
        self.associator = Associator(time_to_drop, self.lost_agent_unc, proto_track_points, proto_Q)

        self.bearing_var = rospy.get_param("~force_sonar_az_var")
        self.range_var = rospy.get_param("~force_sonar_range_var")

        # Subscribe to all blue team poses
        blue_team = rospy.get_param("~blue_team_names")
        self.agent_poses = {}
        for b in blue_team:
            if b == "surface":
                continue
            rospy.Subscriber("etddf/estimate/" + b, Odometry, self.blue_team_callback, callback_args=b)

        # Get my pose
        self.my_name = rospy.get_namespace()[:-1].strip("/")
        pose_topic = "odometry/filtered/odom"
        rospy.Subscriber(pose_topic, Odometry, self.pose_callback)
        self.cuprint("Waiting for orientation")
        rospy.wait_for_message(pose_topic, Odometry) # TODO add back in

        # Debug purposes
        # o = Odometry() # main agent at zero-zero
        # cov = np.eye(6)
        # o.pose.covariance = list( cov.flatten() )
        # o.pose.pose.orientation.w = 1
        # self.pose_callback(o)
        # self.cuprint("Orientation found")

        self.red_agent_name = rospy.get_param("~red_agent_name")
        if self.red_agent_name != "":
            rospy.Subscriber("etddf/estimate/" + self.red_agent_name, Odometry, self.red_agent_callback)

        self.pub = rospy.Publisher("sonar_processing/target_list/associated", SonarTargetList, queue_size=10)

        # Sonar Controller Params
        self.enable_sonar_control = rospy.get_param("~enable_sonar_control")
        if self.enable_sonar_control:
            self.sonar_control_pub = rospy.Publisher("ping360_node/sonar/set_scan", SonarSettings, queue_size=10)
            self.scan_size_deg = rospy.get_param("~scan_size_deg")
            self.ping_thresh = rospy.get_param("~ping_thresh")
            self.scan_angle = None
            self.prototrack = None
            rospy.Subscriber("ping360_node/sonar/scan_complete", UInt16, self.scan_angle_callback)
            self.cuprint("Waiting for scan to complete")
            rospy.wait_for_message( "ping360_node/sonar/scan_complete", UInt16 )

        sonar_topic = "sonar_processing/target_list"
        rospy.Subscriber(sonar_topic, SonarTargetList, self.sonar_callback)

        self.cuprint("Loaded")
    
    def scan_angle_callback(self, msg):
        last_scan_angle_rad = self._gradian2radian( msg.data )
        self.scan_angle = last_scan_angle_rad # ending scan angle
        self.update_sonar_control()

    def update_sonar_control(self):
        if self.enable_sonar_control:
            agent_dict = self._get_agent_dict()
            my_pos = agent_dict[self.my_name][0]
            # Update the scan control
            scan_size_rad = np.radians(self.scan_size_deg)
            angle, scan_360 = scan_control(
                self.scan_angle, my_pos, agent_dict, self.prototrack, 
                scan_size_rad, self.ping_thresh, self.lost_agent_unc)
            if scan_360:
                print("Scanning Full Sweep 360")
            self.set_sonar_scan(angle, scan_360)

    """ These functions align with the real gradian angle in which the ping360 scans """
    def _gradian2radian(self, grad):
        return normalize_angle( (np.pi / 200.0)*(200 - grad) )
    def _radian2gradian(self, rad):
        return np.mod( 200 - (200/np.pi)*rad, 400)

    def set_sonar_scan(self, start_angle, scan_360):
        start_grad = int(self._radian2gradian(start_angle))
        if scan_360:
            end_grad = int(np.mod( start_grad + 399, 400))
        else:
            scan_size_grad = (200 / np.pi) * np.radians(self.scan_size_deg)
            end_grad = int(np.mod( start_grad + scan_size_grad, 400 ))

        print("New sonar configuration: {}".format([start_grad, end_grad]))
        self.sonar_control_pub.publish( SonarSettings(start_grad, end_grad) )
        
    def _get_agent_dict(self):
        # Construct agent_dict
        agent_dict = {}
        for a in self.agent_poses:
            if "red" in a:
                continue
            position = self.agent_poses[a].pose.position
            position = np.array([[position.x],[position.y],[position.z]])
            cov = np.reshape(self.agent_poses[a].covariance, (6,6))
            cov = cov[:3,:3]
            agent_dict[a] = [position, cov]
        return agent_dict

    def pose_callback(self, msg):
        ori = msg.pose.pose.orientation
        (r, p, y) = tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.orientation_rad = y
        self.agent_poses[self.my_name] = msg.pose
    
    def red_agent_callback(self, msg):
        if self.red_agent_name != "":
            self.agent_poses[self.red_agent_name] = msg.pose
        else:
            self.cuprint("Associator received pose for red agent but was not configured to associate with it")

    def blue_team_callback(self, msg, agent_name):
        self.agent_poses[agent_name] = msg.pose

    def sonar_callback(self, msg):
        self.cuprint("Message received")

        agent_dict = self._get_agent_dict().copy()
        my_pos = self.agent_poses[self.my_name].pose.position
        del agent_dict[self.my_name] # so as to not associate with ourselves

        # Construct meas np array, linearizing
        new_msg = deepcopy(msg)
        new_msg.targets = []
        for st in msg.targets:
            inertial_bearing = st.bearing_rad + self.orientation_rad
            self.cuprint("Associating r: {} az: {}".format(round(st.range_m,1), round(np.degrees(inertial_bearing),1)))

            meas_x = st.range_m * np.cos(inertial_bearing)
            meas_y = st.range_m * np.sin(inertial_bearing)
            meas = np.array([[meas_x], [meas_y]])
            self.cuprint("World coords: {}".format(meas.flatten()))

            bearing_std = np.sqrt( self.bearing_var )
            unc_x = ( st.range_m * bearing_std ) ** 2
            R = np.array( [[unc_x, 0],[0, unc_x]]) # TODO actual approximation with rotated covariance
            t = msg.header.stamp

            agent = self.associator.associate(agent_dict, meas, R, t.secs, association_sigma=4)
            self.prototrack = self.associator.get_proto()

            # print("Sonar meas information")
            # print(self.orientation_rad)
            # print(st)
            # print(meas)
            # print(agent_dict)

            if agent != "none" and agent != "proto":
                self.cuprint("Meas associated: {}".format(agent))
                st.associated = True
                st.id = agent
                st.bearing_variance = self.bearing_var
                st.range_variance = self.range_var
                new_msg.targets.append( st )
            
            self.scan_angle = st.bearing_rad

            # Update the scan control
            self.update_sonar_control()

        # Publish new msg
        if new_msg.targets:
            self.pub.publish( new_msg )

if __name__ == "__main__":
    rospy.init_node("sonar_association")
    d = SonarAssociator()

    debug = False
    if not debug:
        rospy.spin()
    else:

        # LAUNCH TESTS
        print("Launching tests")

        # Test associating the measurement with a blue agent

        # 2nd blue agent at 5,5
        o2 = Odometry()
        o2.pose.pose.position.x = 5
        o2.pose.pose.position.y = 5
        cov = np.eye(6)
        o2.pose.covariance = list( cov.flatten() )
        d.blue_team_callback(o2, "bluerov2_5")

        o3 = Odometry()
        o3.pose.pose.position.x = 0
        o3.pose.pose.position.y = 0
        cov = np.eye(6) * 10000
        o3.pose.covariance = list( cov.flatten() )
        d.red_agent_callback(o3)

        # d.scan_angle_callback( UInt16( np.arctan2(5,5) ) )

        # Test measurement generation
        stl = SonarTargetList()
        stl.header.stamp = rospy.get_rostime()
        st = SonarTarget()
        st.range_m = np.linalg.norm([5,5])
        st.bearing_rad = np.arctan2(5,5)
        st.range_variance = d.range_var
        st.bearing_variance = d.bearing_var
        stl.targets.append(st)

        d.sonar_callback(stl) # CHECK THAT WE ASSOCIATED  WITH BLUEROV2_5

        # Test the prototrack
        stl.targets = []
        st = SonarTarget()
        st.range_m = np.linalg.norm([-5,-5])
        st.bearing_rad = np.arctan2(-5,-5)
        st.range_variance = d.range_var
        st.bearing_variance = d.bearing_var
        stl.targets.append(st) # Start a prototrack
        stl.targets.append(st)
        stl.targets.append(st) # Check that we associate with red agent
        d.sonar_callback(stl)

        rospy.spin()

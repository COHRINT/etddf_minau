#!/usr/bin/python3

from copy import deepcopy
import tf
import rospy
from nav_msgs.msg import Odometry
from minau.msg import SonarTargetList, SonarTarget
from cuprint.cuprint import CUPrint
from deltatier.associator import Associator
import numpy as np

class SonarAssociator:

    def __init__(self):
        
        self.cuprint = CUPrint("SonarAssociator")
        self.cuprint("Loading")

        # Load params
        time_to_drop = rospy.get_param("~time_to_drop")
        lost_agent_unc = rospy.get_param("~lost_agent_unc")
        proto_track_points = rospy.get_param("~proto_track_points")
        process_noise = rospy.get_param("~process_noise/blueteam/x")
        proto_Q = np.array([[process_noise, 0],[0, process_noise]])
        self.associator = Associator(time_to_drop, lost_agent_unc, proto_track_points, proto_Q)

        self.bearing_var = rospy.get_param("~bearing_var")
        self.range_var = rospy.get_param("~range_var")

        # Subscribe to all blue team poses
        blue_team = rospy.get_param("~blue_team_names")
        self.agent_poses = {}
        for b in blue_team:
            if b == "surface":
                continue
            rospy.Subscriber("etddf/estimate/" + b, Odometry, self.blue_team_callback, callback_args=b)

        # Get my pose
        self.my_name = rospy.get_namespace()[:-1].strip("/")
        pose_topic = "etddf/estimate" + rospy.get_namespace()[:-1]
        rospy.Subscriber(pose_topic, Odometry, self.pose_callback)
        self.cuprint("Waiting for orientation")
        # rospy.wait_for_message(pose_topic, Odometry) # TODO add back in
        self.cuprint("Orientation found")

        self.red_agent_name = rospy.get_param("~red_agent_name")
        rospy.Subscriber("etddf/estimate/" + self.red_agent_name, Odometry, self.red_agent_callback)

        self.pub = rospy.Publisher("sonar_processing/target_list/associated", SonarTargetList, queue_size=10)

        sonar_topic = "sonar_processing/target_list"
        rospy.Subscriber(sonar_topic, SonarTargetList, self.sonar_callback)

        self.cuprint("Loaded")

    def pose_callback(self, msg):
        ori = msg.pose.pose.orientation
        (r, p, y) = tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.orientation_rad = y
        self.agent_poses[self.my_name] = msg.pose
    
    def red_agent_callback(self, msg):
        self.agent_poses[self.red_agent_name] = msg.pose

    def blue_team_callback(self, msg, agent_name):
        self.agent_poses[agent_name] = msg.pose

    def sonar_callback(self, msg):
        self.cuprint("Message received")

        # Construct agent_dict
        agent_dict = {}
        for a in self.agent_poses:
            position = self.agent_poses[a].pose.position
            position = np.array([[position.x],[position.y],[position.z]])
            cov = np.reshape(self.agent_poses[a].covariance, (6,6))
            cov = np.eye(6) # TODO remove
            cov = cov[:3,:3]
            agent_dict[a] = [position, cov]

        # Construct meas np array, linearizing
        new_msg = deepcopy(msg)
        new_msg.targets = []
        for st in msg.targets:
            meas_x = st.range_m * np.cos(st.bearing_rad + self.orientation_rad)
            meas_y = st.range_m * np.sin(st.bearing_rad + self.orientation_rad)
            meas = np.array([[meas_x], [meas_y]])

            bearing_std = np.sqrt( self.bearing_var )
            unc_x = ( st.range_m * bearing_std ) ** 2
            R = np.array( [[unc_x, 0],[0, unc_x]])
            t = msg.header.stamp

            agent, proto = self.associator.associate(agent_dict, meas, R, t)

            if agent != "none" and not proto:
                self.cuprint("Meas associated: {}".format(agent))
                st.associated = True
                st.id = agent
                new_msg.targets.append( st )

        # Publish new msg
        if new_msg.targets:
            print("Publishing")
            self.pub.publish( new_msg )

            # Add the sonar controller into the mix!

if __name__ == "__main__":
    rospy.init_node("sonar_association")
    d = SonarAssociator()
    # rospy.spin()

    # LAUNCH TESTS
    print("Launching tests")

    # Test associating the measurement with a blue agent
    o = Odometry() # main agent at zero-zero
    o.pose.pose.orientation.w = 1
    d.pose_callback(o)

    # 2nd blue agent at 5,5
    o2 = Odometry()
    o2.pose.pose.position.x = 5
    o2.pose.pose.position.y = 5
    d.blue_team_callback(o2, "bluerov2_5")

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

    rospy.spin()

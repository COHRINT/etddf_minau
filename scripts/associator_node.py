#!/usr/bin/python3

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
        self.time_to_drop = rospy.get_param("~time_to_drop")
        self.lost_agent_unc = rospy.get_param("~lost_agent_unc")
        self.proto_track_points = rospy.get_param("~proto_track_points")
        process_noise = rospy.get_param("~process_noise/blueteam/x")
        self.proto_Q = np.array([[process_noise, 0],[0, process_noise]])

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
        self.pose = msg.pose
        self.agent_poses[self.my_name] = msg.pose
    
    def red_agent_callback(self, msg):
        self.agent_poses[self.red_agent_name] = msg.pose

    def blue_team_callback(self, msg, agent_name):
        self.agent_poses[agent_name] = msg.pose

    def sonar_callback(self, msg):
        self.cuprint("Message received")
        pass

if __name__ == "__main__":
    rospy.init_node("sonar_association")
    d = SonarAssociator()

    # Test associating the measurement with a blue agent
    

    # Test the prototrack

    rospy.spin()

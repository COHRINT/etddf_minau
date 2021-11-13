#!/usr/bin/python3

import rospy
from nav_msgs.msg import Odometry
from minau.msg import SonarTargetList, SonarTarget
from cuprint.cuprint import CUPrint
from deltatier.associator import Associator

class SonarAssociator:

    def __init__(self):
        
        self.cuprint = CUPrint("SonarAssociator")
        self.cuprint("Loading")

        # pose_topic = "etddf/estimate" + rospy.get_namespace()[:-1]
        pose_topic = rospy.get_namespace()[:-1] + "/pose_gt"
        rospy.Subscriber(pose_topic, Odometry, self.pose_callback)
        rospy.wait_for_message(pose_topic, Odometry)

        self.red_team_name = rospy.get_param("~red_team_name")
        blue_team = rospy.get_param("~blue_team_names")
        self.blue_team = {}
        for b in blue_team:
            if b == "surface":
                continue
            rospy.Subscriber("etddf/estimate/" + b, Odometry, self.blue_team_callback, callback_args=b)
            rospy.wait_for_message(pose_topic, Odometry)

        self.pub = rospy.Publisher("sonar_processing/target_list/associated", SonarTargetList, queue_size=10)

        sonar_topic = "sonar_processing/target_list"
        rospy.Subscriber(sonar_topic, SonarTargetList, self.sonar_callback)

        self.cuprint("Loaded")

    def pose_callback(self, msg):
        self.pose = msg.pose

    def blue_team_callback(self, msg, agent_name):
        self.blue_team[agent_name] = msg.pose

    def sonar_callback(self, msg):
        self.cuprint("Message received")
        pass

if __name__ == "__main__":
    rospy.init_node("sonar_association")
    d = SonarAssociator()
    rospy.spin()

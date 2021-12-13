#!/usr/bin/env python
"""
This file subscribes to the robot localization topics and republishes them with a new timestamp
"""

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from std_msgs.msg import Header
import rospy

class MeasRepub:

    def __init__(self):
        self.ori_pub = rospy.Publisher("ori/repub", PoseWithCovarianceStamped, queue_size=10)
        self.baro_pub = rospy.Publisher("baro/repub", PoseWithCovarianceStamped, queue_size=10)
        self.dvl_pub = rospy.Publisher("dvl/repub", TwistWithCovarianceStamped, queue_size=10)

        rospy.Subscriber("ori", PoseWithCovarianceStamped, self.ori_callback)
        rospy.Subscriber("baro", PoseWithCovarianceStamped, self.baro_callback)
        rospy.Subscriber("dvl", TwistWithCovarianceStamped, self.dvl_callback)

    def ori_callback(self, msg):
        h = Header()
        h.frame_id = msg.header.frame_id
        h.stamp = rospy.get_rostime()
        msg.header = h
        self.ori_pub.publish(msg)

    def baro_callback(self, msg):
        h = Header()
        h.frame_id = msg.header.frame_id
        h.stamp = rospy.get_rostime()
        msg.header = h
        self.baro_pub.publish(msg)

    def dvl_callback(self, msg):
        h = Header()
        h.frame_id = msg.header.frame_id
        h.stamp = rospy.get_rostime()
        msg.header = h
        self.dvl_pub.publish(msg)

rospy.init_node("meas_repub")
mr = MeasRepub()
rospy.spin()

#!/usr/bin/env python

"""
Simulates two red agents in the sonar scan
"""

from minau.msg import SonarTargetList, SonarTarget
import rospy
from nav_msgs.msg import Odometry
import tf
import numpy as np

class FakeRedAgents:

    def __init__(self):
        self.yaw = None
        strap_topic = "odometry/filtered/odom"
        rospy.Subscriber( strap_topic, Odometry, self.ori_callback, queue_size=1)
        rospy.wait_for_message(strap_topic, Odometry)

    def ori_callback(self, msg):
        ori = msg.pose.pose.orientation
        (r, p, y) = tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.yaw = y

if __name__ == "__main__":

    rospy.init_node("fake_red_agents")
    fra = FakeRedAgents()

    pub = rospy.Publisher("sonar_processing/target_list/associated", SonarTargetList, queue_size=1)

    # Assume my_loc is [4,1]

    sleep_secs = 3.0

    while not rospy.is_shutdown():

        # Publish first agent
        print("Publishing red3")
        st = SonarTarget()
        st.id = "red3"
        st.bearing_rad = fra.yaw + np.radians(45)
        st.range_m = 4.0
        st.associated = True
        stl = SonarTargetList()
        stl.targets.append(st)
        stl.header.stamp = rospy.get_rostime()
        stl.header.frame_id = "base_link"
        pub.publish(stl)

        rospy.sleep(sleep_secs)

        # Publish second agent
        # print("Publishing red4")
        # st = SonarTarget()
        # st.id = "red4"
        # st.bearing_rad = fra.yaw + np.radians(135)
        # st.range_m = 4.0
        # st.associated = True
        # stl = SonarTargetList()
        # stl.targets.append(st)
        # stl.header.stamp = rospy.get_rostime()
        # stl.header.frame_id = "base_link"
        # pub.publish(stl)

        # rospy.sleep(sleep_secs)
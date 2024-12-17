#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Bool


class PedestrianPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('pedestrian_publisher', String, queue_size=20)
        rospy.init_node('pedestrian_publisher', anonymous=True)
        self.rate = rospy.Rate(10)


    def publish(self, message):
        stop = False
        while not rospy.is_shutdown() and not stop:
            self.publisher.publish(message)
            stop = True


class PedestrianLeftPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_p_left', Bool, queue_size=20)
        # rospy.init_node('topic_p_left', anonymous=True)
        self.rate = rospy.Rate(10)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)


class PedestrianRightPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_p_right', Bool, queue_size=20)
        # rospy.init_node('topic_p_right', anonymous=True)
        self.rate = rospy.Rate(10)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)


class VehicleLeftPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_t_left', Bool, queue_size=20)
        # rospy.init_node('topic_t_left', anonymous=True)
        self.rate = rospy.Rate(10)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)


class VehicleRightPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_t_right', Bool, queue_size=20)
        # rospy.init_node('topic_t_right', anonymous=True)
        self.rate = rospy.Rate(10)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)




if __name__ == '__main__':
    try:
        publisher = PedestrianPublisher()
        publisher.publish("Hello ROS 4!")
    except rospy.ROSInterruptException:
        pass
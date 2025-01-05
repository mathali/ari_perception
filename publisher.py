#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


class PedestrianPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('pedestrian_publisher', String, queue_size=5)
        rospy.init_node('pedestrian_publisher', anonymous=True)
        self.rate = rospy.Rate(20)


    def publish(self, message):
        stop = False
        while not rospy.is_shutdown() and not stop:
            self.publisher.publish(message)
            stop = True

class FrontImagePublisher:
    def __init__(self):
        self.publisher = rospy.Publisher('front_camera_image_detections', Image, queue_size=5)
        self.br = CvBridge()
        self.rate = rospy.Rate(20)

    def publish(self, image: np.ndarray) -> None:
        """
        Publish front camera image
        
        Args:
            image: numpy array containing image data
        """
        try:
            img_msg = self.br.cv2_to_imgmsg(image)
            self.publisher.publish(img_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing front camera image: {e}")

class BackImagePublisher:
    def __init__(self):
        self.publisher = rospy.Publisher('back_camera_image_detections', Image, queue_size=5)
        self.br = CvBridge()
        self.rate = rospy.Rate(20)

    def publish(self, image: np.ndarray) -> None:
        """
        Publish back camera image
        
        Args:
            image: numpy array containing image data
        """
        try:
            img_msg = self.br.cv2_to_imgmsg(image)
            self.publisher.publish(img_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing back camera image: {e}")



class PedestrianLeftPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_p_left', Bool, queue_size=5)
        # rospy.init_node('topic_p_left', anonymous=True)
        self.rate = rospy.Rate(20)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)


class PedestrianRightPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_p_right', Bool, queue_size=5)
        # rospy.init_node('topic_p_right', anonymous=True)
        self.rate = rospy.Rate(20)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)


class VehicleLeftPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_t_left', Bool, queue_size=5)
        # rospy.init_node('topic_t_left', anonymous=True)
        self.rate = rospy.Rate(20)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)


class VehicleRightPublisher():

    def __init__(self):
        self.publisher = rospy.Publisher('topic_t_right', Bool, queue_size=5)
        # rospy.init_node('topic_t_right', anonymous=True)
        self.rate = rospy.Rate(20)

    def publish(self, flag: bool) -> None:
        """Publish a boolean flag to the topic"""
        self.publisher.publish(flag)




if __name__ == '__main__':
    try:
        publisher = PedestrianPublisher()
        publisher.publish("Hello ROS 4!")
    except rospy.ROSInterruptException:
        pass
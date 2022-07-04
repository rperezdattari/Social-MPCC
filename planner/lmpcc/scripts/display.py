#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
class Display:

    def __init__(self):
        print('Display Module: Init')

        self.image_pub_ = rospy.Publisher('/lmpcc/view', Image, queue_size=1)
        self.bridge = CvBridge()
        # Also a listener to republish the goal
        self.camera_sub = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/front/image_color", Image, self.image_callback )
        self.vel_sub = rospy.Subscriber("/rodrigo", Float64, self.vref_callback)
        self.drive_sub = rospy.Subscriber("/drive", Float64, self.drive_callback)
        self.human_action_sub = rospy.Subscriber("/human_action", Float64, self.human_action_callback)

        self.max_velocity = 8
        self.velocity_reference = 0
        self.human_action = 0
        self.drive = 1

    def vref_callback(self,msg):
        self.velocity_reference = msg.data
        
    def drive_callback(self,msg):
        self.drive = msg.data
        
    def human_action_callback(self,msg):
        self.human_action = msg.data

    def image_callback(self,msg):
        self.camera_view_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def update(self):

        if hasattr(self,"camera_view_img"):
            # represents the top left corner of rectangle
            start_point = (5, 5)
            
            # Red bar first
            end_point = (int(100*self.velocity_reference/self.max_velocity+6),100)

            # Red color in BGR
            color = (0, 0, 255)

            # Fill rectangle
            thickness = -1

            self.output_image = cv2.rectangle(self.camera_view_img, start_point, end_point, color, thickness)
            
            # Line thickness of 2 px
            thickness = 2
            
            # First plot human action
            color = (255, 255, 0)  # cyan
            
            end_point = (int(100*self.human_action/self.max_velocity+6),100)
            
            self.output_image = cv2.rectangle(self.output_image, start_point, end_point, color, thickness)
            
            # Now plot blue rectangle
            # represents the bottom right corner of rectangle
            end_point = (100, 100)
    
            # Blue color in BGR
            color = (255, 0, 0)

            self.output_image = cv2.rectangle(self.output_image, start_point, end_point, color, thickness)


            # Show drive bool
            thickness = -1
            start_point = (150, 5)
            end_point = (200, 50)
            if self.drive == 1: 
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            self.output_image = cv2.rectangle(self.output_image, start_point, end_point, color, thickness)

            # Convert to ROS format
            self.output_image = self.bridge.cv2_to_imgmsg(self.output_image, encoding="bgr8")

            self.image_pub_.publish(self.output_image)

if __name__ == '__main__':

    rospy.init_node("display", anonymous=True)

    display_module = Display()
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():

        display_module.update()

        rate.sleep()

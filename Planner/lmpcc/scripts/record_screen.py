#!/usr/bin/env python3

import cv2
import numpy as np
from mss import mss
from PIL import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as image_ros
import rospy

class RecordScreen:

    def __init__(self):
        print('RecordScreen Module: Init')

        self.bridge = CvBridge()

        self.image_pub_ = rospy.Publisher('/lmpcc/processed_view', image_ros, queue_size=1)

        # Manual tuning to record part of the screen
        self.mon = {'left': 260, 'top': 500, 'width': 640, 'height': 480}

    def update(self):
        with mss() as sct:
            screenShot = sct.grab(self.mon)

            img = Image.frombytes(
                'RGB',
                (screenShot.width, screenShot.height),
                screenShot.rgb,
            )
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

            # Uncomment to debug
            #cv2.imshow('test', frame)

            # Convert to ROS format
            self.output_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            self.image_pub_.publish(self.output_image)

if __name__ == '__main__':

    rospy.init_node("record_screen", anonymous=True)

    display_module = RecordScreen()
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():

        display_module.update()

        rate.sleep()


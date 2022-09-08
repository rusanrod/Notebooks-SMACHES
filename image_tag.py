#!/usr/bin/env python3
    
import numpy as np
import rospy
import ros_numpy
import tf2_ros
import os
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

from std_msgs.msg import String


"""
import sys
print (sys.path)
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
"""

import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

def callback(img_msg):

    cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    
    cv2.imshow('xtion rgb  cam'	, cv2_img)
    # Process any keyboard commands
    #keystroke = cv2.waitkey(5)
  
     #print req
    img_name = f'test.jpg'
    cv2.imwrite(img_name,cv2_img)
    print('SAve requested',len(os.listdir('images')))

    keystroke = cv2.waitKey(1)
    if 32 <= keystroke and keystroke < 128:
        cc = chr(keystroke).lower()
        print (cc)
        #if cc =='s':

    
        #    cv2.imwrite('images/cap'+str(len(os.listdir('images')) +1 ) + '.png',cv2_img)
        #    print('SAve requested',len(os.listdir('images')))
        #    #print(res)
        
        if cc =='q':
            rospy.signal_shutdown("User hit q key to quit.")



#cv2_img_depth = bridge.imgmsg_to_cv2(res.segmented_objects_array.table_objects_array[i].depth_image_array[j] )
#cv2_img = bridge.imgmsg_to_cv2(res.segmented_objects_array.table_objects_array[i].rgb_image_array[j] )


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a uniques
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    global req,service_client , res
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, callback)
    
        
    #rospy.Subscriber("/hsrb/base_scan", LaserScan, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()


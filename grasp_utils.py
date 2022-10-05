# -*- coding: utf-8 -*-
import cv2 
import tf as tf
import rospy
import numpy as np
import ros_numpy
from std_msgs.msg import String
from geometry_msgs.msg import Twist , PointStamped , Point, WrenchStamped, PoseStamped, Quaternion, TransformStamped, Twist, Pose
from sensor_msgs.msg import Image as ImageMsg, LaserScan, PointCloud2
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import trajectory_msgs.msg

#import math as m
#import moveit_commander
#import moveit_msgs.msg

#Class to get XTION camera info (head)
class RGBD():
    def __init__(self):
        self._br = tf.TransformBroadcaster()
        self._cloud_sub = rospy.Subscriber(
            "/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
            PointCloud2, self._cloud_cb)
        self._points_data = None
        self._image_data = None
        self._h_image = None
        self._region = None
        self._h_min = 0
        self._h_max = 0
        self._xyz = [0, 0, 0]
        self._frame_name = None

    def _cloud_cb(self, msg):
        self._points_data = ros_numpy.numpify(msg)
        self._image_data = \
        self._points_data['rgb'].view((np.uint8, 4))[..., [2, 1, 0]]
        hsv_image = cv2.cvtColor(self._image_data, cv2.COLOR_RGB2HSV_FULL)
        self._h_image = hsv_image[..., 0]
        self._region = \
        (self._h_image > self._h_min) & (self._h_image < self._h_max)
        if not np.any(self._region):
            return
            
        (y_idx, x_idx) = np.where(self._region)
        x = np.average(self._points_data['x'][y_idx, x_idx])
        y = np.average(self._points_data['y'][y_idx, x_idx])
        z = np.average(self._points_data['z'][y_idx, x_idx])
        self._xyz = [y, x, z]
        if self._frame_name is None:
            return

        self._br.sendTransform(
        (x, y, z), tf.transformations.quaternion_from_euler(0, 0, 0),
        rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs),
        self._frame_name,
        msg.header.frame_id)

    def get_image(self):
        return self._image_data

    def get_points(self):
        return self._points_data

    def get_h_image(self):
        return self._h_image

    def get_region(self):
        return self._region

    def get_xyz(self):
        return self._xyz

    def set_h(self, h_min, h_max):
        self._h_min = h_min
        self._h_max = h_max

    def set_coordinate_name(self, name):
        self._frame_name = name
        
#Color segmentator
    def color_segmentator(self, color = "orange"):
        image = self.get_image()
        if(color == "blue"):
            lower_threshold = (100,120,100)
            upper_threshold = (150,220,240)
        else:
            lower_threshold = (102,95,97)
            upper_threshold = (115,255,255)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, lower_threshold, upper_threshold)
        res = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        pos = []
        pixels = cv2.findNonZero(mask)
        pixels = list(cv2.mean(pixels))
        pos.append(pixels[:2])
        return pos

#Class to get hand camera images(RGB)
class HAND_RGB():
    def __init__(self):
        self.cam_sub = rospy.Subscriber(
            '/hsrb/hand_camera/image_raw',
            ImageMsg, self._callback)
        self._points_data = None
        self._image_data = None
        
    def _callback(self, msg):
        self._image_data = ros_numpy.numpify(msg)
        
    def get_image(self):
        image = self._image_data
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
#Color segmentator
    def color_segmentator(self, color = "orange"):
        image = self.get_image()
        if(color == "blue"):
            lower_threshold = (100,120,100)
            upper_threshold = (150,220,240)
        else:
            lower_threshold = (102,95,97)
            upper_threshold = (115,255,255)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, lower_threshold, upper_threshold)
        res = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        pos = []
        pixels = cv2.findNonZero(mask)
        pixels = list(cv2.mean(pixels))
        pos.append(pixels[:2])
        return pos


#Class to get wrist sensor info (Force and torque)
class WRIST_SENSOR():
    def __init__(self):
        self._cam_sub = rospy.Subscriber(
            '/hsrb/wrist_wrench/compensated',
            WrenchStamped, self._callback)
        self.force = None
        self.torque = None
         
    def _callback(self, msg):
        self.force = msg.wrench.force
        self.torque = msg.wrench.torque

    def get_force(self):
        force = [self.force.x, self.force.y, self.force.z]
        return force
    
    def get_torque(self):
        torque = [self.torque.x, self.torque.y, self.torque.z]
        return torque

#Class to handle end effector (gripper)
class GRIPPER():
    def __init__(self):
        self._grip_cmd_pub = rospy.Publisher('/hsrb/gripper_controller/command',
                               trajectory_msgs.msg.JointTrajectory, queue_size=100)
        self._joint_name = "hand_motor_joint"
        self._position = 0.5
        self._velocity = 0.5
        self._effort = 0.0
        self._duration = 1

    def _manipulate_gripper(self):
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = [self._joint_name]
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = [self._position]
        p.velocities = [self._velocity]
        p.accelerations = []
        p.effort = [self._effort]
        p.time_from_start = rospy.Duration(self._duration)
        traj.points = [p]
        self._grip_cmd_pub.publish(traj)
        
    def change_velocity(self, newVel):
        self._velocity = newVel
    
    def open(self):
        self._position = 1.23
        self._effort = 0.2
        self._manipulate_gripper()
    
    def close(self):
        self._position = -0.82
        self._effort = 0.2
        self._manipulate_gripper()

class OMNIBASE():
    def __init__(self):
        self._base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)
        self.velX = 0
        self.velY = 0
        self.velT = 0
        self.timeout = 0.5
        self.MAX_VEL = 0.03
        self.MAX_VEL_THETA = 0.5
    def _move_base_vel(self):
            twist = Twist()
            twist.linear.x = self.velX
            twist.linear.y = self.velY
            twist.angular.z = self.velT
            self._base_vel_pub.publish(twist)

    def _move_base_time(self):
            start_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - start_time < self.timeout:  
                self._move_base_vel()

    def tiny_move(self, velX = 0, velY = 0, velT = 0, std_time = 0.5, MAX_VEL = 0.03, MAX_VEL_THETA = 0.5):
        self.MAX_VEL = MAX_VEL
        self.MAX_VEL_THETA = MAX_VEL_THETA
        self.timeout = std_time
        if abs(velX) > MAX_VEL: 
            self.velX =  MAX_VEL * (velX / abs(velX))
        else:
            self.velX = velX
        if abs(velY) > MAX_VEL:
            self.velY = MAX_VEL * (velY / abs(velY))
        else:
            self.velY = velY
        if abs(velT) > MAX_VEL_THETA:
            self.velT = MAX_VEL_THETA * (velT / abs(velT))
        else:
            self.velT = velT
        self._move_base_time()


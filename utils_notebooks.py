# -*- coding: utf-8 -*-
import tf as tf
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist , PointStamped , Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image as ImageMsg
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker , MarkerArray
import numpy as np
import pandas as pd
import ros_numpy
from gazebo_ros import gazebo_interface
from sklearn.decomposition import PCA
import math as m
import moveit_commander
import moveit_msgs.msg
import actionlib
import subprocess

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist , Pose
from IPython.display import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import LaserScan, PointCloud2
import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2 


class RGBD():
    u"""RGB-Dデータを扱うクラス"""

    def __init__(self):
        self._br = tf.TransformBroadcaster()
        # ポイントクラウドのサブスクライバのコールバックに_cloud_cbメソッドを登録
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
        # ポイントクラウドを取得する
        self._points_data = ros_numpy.numpify(msg)

        # 画像を取得する
        self._image_data = \
            self._points_data['rgb'].view((np.uint8, 4))[..., [2, 1, 0]]

        # 色相画像を作成する
        hsv_image = cv2.cvtColor(self._image_data, cv2.COLOR_RGB2HSV_FULL)
        self._h_image = hsv_image[..., 0]

        # 色相の閾値内の領域を抽出する
        self._region = \
            (self._h_image > self._h_min) & (self._h_image < self._h_max)

        # 領域がなければ処理を終える
        if not np.any(self._region):
            return

        # 領域からxyzを計算する
        (y_idx, x_idx) = np.where(self._region)
        x = np.average(self._points_data['x'][y_idx, x_idx])
        y = np.average(self._points_data['y'][y_idx, x_idx])
        z = np.average(self._points_data['z'][y_idx, x_idx])
        self._xyz = [y, x, z]
        #self._xyz = [x, y, z]

        # 座標の名前が設定されてなければ処理を終える
        if self._frame_name is None:
            return

        # tfを出力する
        self._br.sendTransform(
            (x, y, z), tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs),
            self._frame_name,
            msg.header.frame_id)

    def get_image(self):
        u"""画像を取得する関数"""
        return self._image_data

    def get_points(self):
        u"""ポイントクラウドを取得する関数"""
        return self._points_data

    def get_h_image(self):
        u"""色相画像を取得する関数"""
        return self._h_image

    def get_region(self):
        u"""抽出領域の画像を取得する関数"""
        return self._region

    def get_xyz(self):
        u"""抽出領域から計算されたxyzを取得する関数"""
        return self._xyz

    def set_h(self, h_min, h_max):
        u"""色相の閾値を設定する関数"""
        self._h_min = h_min
        self._h_max = h_max

    def set_coordinate_name(self, name):
        u"""座標の名前を設定する関数"""
        self._frame_name = name

class HandRGB():
    def __init__(self):
        self.cam_sub = rospy.Subscriber(
            '/hsrb/hand_camera/image_raw',
            ImageMsg, self._callback)
        self._points_data = None
        self._image_data = None
        
    def _callback(self, msg):
        
#         self._points_data = ros_numpy.numpify(msg)
        self._image_data = ros_numpy.numpify(msg)
#         cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
        
    def get_image(self):
        image = self._image_data
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def delete_object(name):
    u"""Gazeboの物体を消す関数

    引数:
        name (str): 物体の名前

    """

    cmd = ['rosservice', 'call', 'gazebo/delete_model',
           '{model_name: ' + str(name) + '}']
    subprocess.call(cmd)

    
    
    

def spawn_object(gazebo_name, name, x, y, z, yaw,roll=0.0 , pitch=0.0):
    global _path_xml, _path_model
    _path_xml = '/home/oscar/Codes/catkin_mio_ws/src/tmc_wrs_gazebo_world/models/MODEL_NAME/model-1_4.sdf'
    _path_model = '/home/oscar/Codes/catkin_mio_ws/src/tmc_wrs_gazebo_world/models/'
    #_path_xml = '/home/roboworks/Codes/catkin_mio/src/tmc_wrs_gazebo_world/models/MODEL_NAME/model-1_4.sdf'
    #_path_model = '/home/oscar/Codes/catkin_mio_ws/src/tmc_wrs_gazebo_world/models/'
    
    rospy.loginfo('Spawn: {0}'.format(name))
    initial_pose = Pose()
    initial_pose.position.x = x
    initial_pose.position.y = y
    initial_pose.position.z = z
    
    
    q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    initial_pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
    rospy.loginfo('Spawn: {0}'.format(q))

    path_xml = _path_xml.replace('MODEL_NAME', name)

    with open(path_xml, "r") as f:
        model_xml = f.read()

    model_xml = model_xml.replace('PATH_TO_MODEL', _path_model)

    gazebo_interface.spawn_sdf_model_client(gazebo_name, model_xml, rospy.get_namespace(),
                                            initial_pose, "", "/gazebo")
def gazebo_2_world(x,y):

    x_world= x+2.1
    y_world= -(y-1.2)
    return (x_world,y_world)

def world_2_gazebo(y_world , x_world):

    x= ( x_world - 2.1)
    y= (-y_world + 1.2) 
    return (x , y)
def find_2nd_biggest_contour_ix(contours):
    if (len(contours) >= 2):
        
        areas=[]
        for c in contours:
            M= cv2.moments(c)
            areas.append(M['m00'])
        ser=pd.Series(areas)
        ser.sort_values(ascending=False,inplace=True)
        return ser.index[1]
    else:
        print('only one contour')
        return (0)
def pad_digit_num(num,length):
    text_num=''
    for i in range(length -len(str(num))):
        text_num=text_num+'0'
    return(text_num+str(num))




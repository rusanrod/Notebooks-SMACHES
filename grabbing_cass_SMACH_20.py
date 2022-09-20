#!/usr/bin/env python3
from std_srvs.srv import Empty, Trigger, TriggerRequest
import smach
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Point , Quaternion, Twist
from actionlib_msgs.msg import GoalStatus
import moveit_commander
import moveit_msgs.msg
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import controller_manager_msgs.srv
import rospy
import trajectory_msgs.msg
import geometry_msgs.msg
#from object_classification.srv import *
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge, CvBridgeError

import cv2 as cv
import numpy as np

from utils_notebooks import *
from utils_takeshi import *

def color_segmentator(plot = False):
    image = hand_cam.get_image()
# print(image)
# image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    umbral_bajo = (102,95,97)
    umbral_alto = (115,255,255)
# hacemos la mask y filtramos en la original
    mask = cv2.inRange(img_hsv, umbral_bajo, umbral_alto)
    res = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(res)
        plt.show()
    pos = []
    pixels = cv.findNonZero(mask)
#     print([pixels])
    pixels = list(cv.mean(pixels))
    pos.append(pixels[:2])
    return pos
def get_line(camera):
    if camera == 'hand':
         img = hand_cam.get_image()
    elif camera == 'head':
        img = rgbd.get_image()
#     cv.imwrite('table.jpg',img)
#     img = cv.imread(cv.samples.findFile('table.jpg'))
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize = 3)
    lines = cv.HoughLines(edges, 1, np.pi/180,200)
#     if len(lines)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
#         if theta > 0.707 and theta < 2 and (y1 or y2)>480/2:
#             cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)    
#     cv.imwrite('TableLines.jpg',img)
    return lines

def tf2_obj_2_arr(transf):
    trans = []
    trans.append(transf.transform.translation.x)
    trans.append(transf.transform.translation.y)
    trans.append(transf.transform.translation.z)
    
    rot = []
    rot.append(transf.transform.rotation.x)
    rot.append(transf.transform.rotation.y)
    rot.append(transf.transform.rotation.z)
    rot.append(transf.transform.rotation.w)
    
    return [trans, rot]
    
    
def correct_points(low_plane=.0,high_plane=0.2):

    #Corrects point clouds "perspective" i.e. Reference frame head is changed to reference frame map
    data = rospy.wait_for_message('/hsrb/head_rgbd_sensor/depth_registered/rectified_points', PointCloud2)
    np_data = ros_numpy.numpify(data)
    
#   new implementation to use only tf2
    transf = tfbuff.lookup_transform('map', 'head_rgbd_sensor_gazebo_frame', rospy.Time())
    [trans, rot] = tf2_obj_2_arr(transf)
    
    eu = np.asarray(tf.transformations.euler_from_quaternion(rot))
    t = TransformStamped()
    rot = tf.transformations.quaternion_from_euler(-eu[1], 0, 0)
    t.header.stamp = data.header.stamp
    
    t.transform.rotation.x = rot[0]
    t.transform.rotation.y = rot[1]
    t.transform.rotation.z = rot[2]
    t.transform.rotation.w = rot[3]

    cloud_out = do_transform_cloud(data, t)
    np_corrected = ros_numpy.numpify(cloud_out)
    corrected = np_corrected.reshape(np_data.shape)

    img = np.copy(corrected['y'])

    img[np.isnan(img)] = 2
    #img3 = np.where((img>low)&(img< 0.99*(trans[2])),img,255)
    img3 = np.where((img>0.99*(trans[2])-high_plane)&(img< 0.99*(trans[2])-low_plane),img,255)
    return img3

def plane_seg_square_imgs(lower=500, higher=50000, reg_ly= 30, reg_hy=600, plt_images=True, low_plane=.0, high_plane=0.2):

    #Segment  Plane using corrected point cloud
    #Lower, higher = min, max area of the box
    #reg_ly= 30,reg_hy=600    Region (low y  region high y ) Only centroids within region are accepted
    
    image = rgbd.get_h_image()
    iimmg = rgbd.get_image()
    points_data = rgbd.get_points()
    img = np.copy(image)
    img3 = correct_points(low_plane,high_plane)
    
#     cv2 on python 3
    contours, hierarchy = cv2.findContours(img3.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    cents=[]
    points=[]
    images=[]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > lower and area < higher :
            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            boundRect = cv2.boundingRect(contour)
            #just for drawing rect, dont waste too much time on this
            img = cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]), (0,0,0), 2)
            # calculate moments for each contour
            if (cY > reg_ly and cY < reg_hy  ):
                image_aux = iimmg[boundRect[1]:boundRect[1]+max(boundRect[2],boundRect[3]),boundRect[0]:boundRect[0]+max(boundRect[2],boundRect[3])]
                images.append(image_aux)
                cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
                cv2.putText(img, f'centroid_{i}_{cX},{cY}',    (cX - 25, cY - 25)   ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                #print ('cX,cY',cX,cY)
                xyz = []

                for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                    for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                        aux = (np.asarray((points_data['x'][ix,jy],points_data['y'][ix,jy],points_data['z'][ix,jy])))
                        if np.isnan(aux[0]) or np.isnan(aux[1]) or np.isnan(aux[2]):
                            'reject point'
                        else:
                            xyz.append(aux)

                xyz = np.asarray(xyz)
                cent = xyz.mean(axis=0)
                cents.append(cent)
#                 print (cent)
                points.append(xyz)
#             else:
#                 print ('cent out of region... rejected')
    sub_plt = 0
    if plt_images:
        for image in images:
           
            sub_plt += 1
            ax = plt.subplot(5, 5, sub_plt)
          
            plt.imshow(image)
            plt.axis("off")

    cents=np.asarray(cents)
    ### returns centroids found and a group of 3d coordinates that conform the centroid
    return(cents,np.asarray(points), images)

def seg_square_imgs(lower=2000, higher=50000, reg_ly=0, reg_hy=1000, reg_lx=0, reg_hx=1000, plt_images=True): 

#     Using kmeans for image segmentation find
#     Lower, higher = min, max area of the box
#     reg_ly= 30,reg_hy=600,reg_lx=0,reg_hx=1000, 
#     Region (low  x,y  region high x,y ) Only centroids within region are accepted
    image = rgbd.get_h_image()
    iimmg = rgbd.get_image()
    points_data = rgbd.get_points()
    values = image.reshape((-1,3))
    values = np.float32(values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER  ,1000,0.1)
    k = 6
    _ , labels , cc = cv2.kmeans(values, k, None, criteria, 30, cv2.KMEANS_RANDOM_CENTERS)
    cc = np.uint8(cc)
    segmented_image = cc[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    th3 = cv2.adaptiveThreshold(segmented_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    im4 = cv2.erode(th3, kernel, iterations = 4)
    plane_mask = points_data['z']
    cv2_img = plane_mask.astype('uint8')
    img = im4
    contours, hierarchy = cv2.findContours(im4.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    cents = []
    points = []
    images = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > lower and area < higher :
            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            boundRect = cv2.boundingRect(contour)
            #just for drawing rect, dont waste too much time on this
            image_aux = iimmg[boundRect[1]:boundRect[1] + max(boundRect[3],boundRect[2]),boundRect[0]:boundRect[0]+max(boundRect[3],boundRect[2])]
            images.append(image_aux)
            img=cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]), (0,0,0), 2)
            #img=cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+max(boundRect[2],boundRect[3]), boundRect[1]+max(boundRect[2],boundRect[3])), (0,0,0), 2)
            # calculate moments for each contour
            if (cY > reg_ly and cY < reg_hy and  cX > reg_lx and cX < reg_hx   ):
                cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
                cv2.putText(img, f'centroid_{i}_{cX},{cY}', (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                #print ('cX,cY',cX,cY)
                xyz = []
                for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                    for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                        aux=(np.asarray((points_data['x'][ix,jy],points_data['y'][ix,jy],points_data['z'][ix,jy])))
                        if np.isnan(aux[0]) or np.isnan(aux[1]) or np.isnan(aux[2]):
                            'reject point'
                        else:
                            xyz.append(aux)
                xyz = np.asarray(xyz)
                cent = xyz.mean(axis=0)
                cents.append(cent)
                #print (cent)
                points.append(xyz)
            else:
                #print ('cent out of region... rejected')
                images.pop()
    sub_plt = 0
    if plt_images:
        for image in images:

            sub_plt+=1
            ax = plt.subplot(5, 5, sub_plt )
            plt.imshow(image)
            plt.axis("off")
    cents=np.asarray(cents)
    #images.append(img)
    return(cents,np.asarray(points), images)

def __manipulate_gripper(pos = 0.5, vel = 0.5, effort = 0.2):
    grip_cmd_pub = rospy.Publisher('/hsrb/gripper_controller/command',
                               trajectory_msgs.msg.JointTrajectory, queue_size=100)
    traj = trajectory_msgs.msg.JointTrajectory()
    traj.joint_names = ["hand_motor_joint"]
    p = trajectory_msgs.msg.JointTrajectoryPoint()
    p.positions = [pos]
    p.velocities = [vel]
    p.accelerations = []
    p.effort = [effort]
    p.time_from_start = rospy.Duration(1)
    traj.points = [p]

    grip_cmd_pub.publish(traj)

def open_gripper(eff=0.5):
    __manipulate_gripper(pos=1.23, vel=0.5, effort=eff)
    
def close_gripper(eff=0.5):
    __manipulate_gripper(pos=-0.831, vel=-0.5, effort=-eff)

def static_tf_publish(cents):
#     Publish tfs of the centroids obtained w.r.t. head sensor frame and references them to map (static)
    transf = tfbuff.lookup_transform('map', 'base_link', rospy.Time(0))
    [trans, rot] = tf2_obj_2_arr(transf)
#     closest_centroid_index=  np.argmin(np.linalg.norm(trans-cents, axis=1))##CLOSEST CENTROID
    closest_centroid_index = 0
    min_D_to_base = 10
    for  i, cent  in enumerate(cents):
        x, y, z = cent
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            print('nan')
        else:
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "head_rgbd_sensor_link"
            t.child_frame_id = f'Object{i}'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = rot[0]
            t.transform.rotation.y = rot[1]
            t.transform.rotation.z = rot[2]
            t.transform.rotation.w = rot[3]
            broad.sendTransform(t)
#             broad.sendTransform((x,y,z), rot, rospy.Time.now(), 'Object'+str(i), "head_rgbd_sensor_link")
            rospy.sleep(0.5)
            transf = tfbuff.lookup_transform('map', f'Object{i}', rospy.Time(0))
            [xyz_map, cent_quat] = tf2_obj_2_arr(transf)
            D_to_base = np.linalg.norm(np.asarray(trans)[:2] - np.asarray(xyz_map)[:2])
            if D_to_base <= min_D_to_base:
                min_D_to_base = D_to_base
                closest_centroid_index = i
                closest_centroid_height = xyz_map[2]
            print ('Distance: base to obj - ', i, np.linalg.norm(np.asarray(trans)[:2] - np.asarray(xyz_map)[:2]))
    i = closest_centroid_index
    transf = tfbuff.lookup_transform('map', f'Object{i}', rospy.Time(0))
    [xyz_map, cent_quat] = tf2_obj_2_arr(transf)
    print('Height closest centroid map', xyz_map[2])
    map_euler = tf.transformations.euler_from_quaternion(cent_quat)
    rospy.sleep(.5)
#     FIXING TF TO MAP ( ODOM REALLY)    
    static_ts = TransformStamped()
    static_ts.header.stamp = rospy.Time.now()
    static_ts.header.frame_id = "map"
    static_ts.child_frame_id = 'cassette'
    static_ts.transform.translation.x = float(xyz_map[0])
    static_ts.transform.translation.y = float(xyz_map[1])
    static_ts.transform.translation.z = float(xyz_map[2])
#     quat = tf.transformations.quaternion_from_euler(-euler[0],0,1.5)
    static_ts.transform.rotation.x = 0#-quat[0]#trans.transform.rotation.x
    static_ts.transform.rotation.y = 0#-quat[1]#trans.transform.rotation.y
    static_ts.transform.rotation.z = 0#-quat[2]#trans.transform.rotation.z
    static_ts.transform.rotation.w = 1#-quat[3]#trans.transform.rotation.w
    print ('xyz_map', xyz_map)
    tf_static_broad.sendTransform(static_ts)
    return closest_centroid_height, closest_centroid_index

def static_publish_ARmarker():
    transf = tfbuff.lookup_transform('map', 'ar_marker/201', rospy.Time(0))
    [xyz_map, cent_quat] = tf2_obj_2_arr(transf)
    print('Height closest centroid map', xyz_map[2])
    map_euler = tf.transformations.euler_from_quaternion(cent_quat)
    rospy.sleep(.5)
#     FIXING TF TO MAP ( ODOM REALLY)    
    static_ts = TransformStamped()
    static_ts.header.stamp = rospy.Time.now()
    static_ts.header.frame_id = "map"
    static_ts.child_frame_id = 'cassette'
    static_ts.transform.translation.x = float(xyz_map[0])
    static_ts.transform.translation.y = float(xyz_map[1])
    static_ts.transform.translation.z = float(xyz_map[2])
#     quat = tf.transformations.quaternion_from_euler(-euler[0],0,1.5)
    static_ts.transform.rotation.x = 0#-quat[0]#trans.transform.rotation.x
    static_ts.transform.rotation.y = 0#-quat[1]#trans.transform.rotation.y
    static_ts.transform.rotation.z = 0#-quat[2]#trans.transform.rotation.z
    static_ts.transform.rotation.w = 1#-quat[3]#trans.transform.rotation.w
    print ('xyz_map', xyz_map)
    tf_static_broad.sendTransform(static_ts)

def tiny_move_base(x = 0, y = 0, theta = 0, std_time = 0.5, MAX_VEL = 0.03):
    MAX_VEL = 0.03
    velX = x/std_time
    velY = y/std_time
    time = std_time
    if abs(velX) > MAX_VEL or abs(velY) > MAX_VEL:
        newVelX =  MAX_VEL * np.sign(velX)
        newVelY = MAX_VEL * np.sign(velY)
#         timeX = x / MAX_VEL
#         timeY = y / MAX_VEL
#         if timeX > timeY:
#             time = timeX
#         else:
#             time = timeY
    else :
        newVelX = velX
        newVelY = velY
    move_base(newVelX, newVelY, theta/std_time, time)

def move_base_vel(vx, vy, vw):
    twist = Twist()
    twist.linear.x = vx
    twist.linear.y = vy
    twist.angular.z = vw 
    base_vel_pub.publish(twist)

def move_base(x,y,yaw,timeout=0.2):
    start_time = rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec() - start_time < timeout:  
        move_base_vel(x, y, yaw)
        
def table_alignment():
#     hcp = head.get_current_joint_values()
#     hcp[0] = 0.0
#     hcp[1] = -0.5
#     head.set_joint_value_target(hcp)
#     head.go()
    threshold = 0.05
    while True:
        lin = get_line('head')
        suma = 0
        for el in lin:
            suma += el[0][1]
        prom = suma / len(lin)
        e = 1.5707 - prom
        print(e)
        if abs(e) < threshold:
            break
        else:    
            move_base(0.0,0.0,0.7*e,0.2)
            hcp = [0,0]
            hcp[1] = -0.05*e
            head.set_joint_value_target(hcp)
            head.go()
            
            ########## Functions for takeshi states ##########
class Proto_state(smach.State):###example of a state definition.
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : PROTO_STATE')

        if self.tries==3:
            self.tries=0 
            return'tries'
        if succ:
            return 'succ'
        else:
            return 'failed'
        global trans_hand
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            return'tries'
   
        

    ##### Define state INITIAL #####
#Estado inicial de takeshi, neutral
class Initial(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        
        rospy.loginfo('STATE : robot neutral pose')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
        clear_octo_client()
        stopper.call()
        scene.remove_world_object()
        #Takeshi neutral
        arm.set_named_target('go')
        arm.go()
        head.set_named_target('neutral')
        succ = head.go()
        starter.call()
        if succ:
            return 'succ'
        else:
            return 'failed'
class Find_AR_marker(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : Find AR marker ')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
        clear_octo_client()
        scene.remove_world_object()
        #Takeshi looks for AR marker
        hcp = head.get_current_joint_values()
        hcp[0]=0.4
        hcp[1]= -0.2
        head.set_joint_value_target(hcp)
        head.go()
        succ = False
        last = 0
        while not succ:
            try:
                t = tfbuff.lookup_transform('base_link', 'ar_marker/201', rospy.Time(0) )
                rospy.sleep(0.3)
                trans, _ = tf2_obj_2_arr(t)
                distanceX = trans[0]
                print(distanceX)
                if distanceX < 0.60 and distanceX > 0.55:
                    hcp = head.get_current_joint_values()
                    hcp[0] += 0.3
                    hcp[1] = -0.2
                    head.set_joint_value_target(hcp)
                    head.go()
                if distanceX < 0.45:
                    succ = True
                else:
                    tiny_move_base(x=0.5,std_time=0.1)
            except:
                tiny_move_base(x=0.5,std_time=0.1)
        if succ:
            return 'succ'
        else:
            return 'failed'
class Pre_grasp_pose(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : pre grasp pose ')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
#         clear_octo_client()
        scene.remove_world_object()
        #static_tf_publish_furniture(-0.95,-0.5,0.0)
        # intento de vuelta
        succ = False
        THRESHOLD = 0.1
        hcp = [0.6,-0.1]
        flag = True
        while not succ:
            try:
                t = tfbuff.lookup_transform('base_link','ar_marker/201',rospy.Time(0))
                _, rot = tf2_obj_2_arr(t)
#     print(rot)
                euler = tf.transformations.euler_from_quaternion(rot)
                theta = euler[2]
                e = theta + 1.57
                print(e)
                if abs(e) < THRESHOLD:
                    print('estoy alineado krnal')
                    succ = True
                    hcp[0] = 0.0
                    head.set_joint_value_target(hcp)
                    head.go()
                else:
                    rospy.sleep(0.55)
                    tiny_move_base(theta = 0.02*e, std_time=0.2)
            except:
                hcp[0] -= 0.2
                head.set_joint_value_target(hcp)
                head.go()
        
        
        rospy.sleep(0.2)
        open_gripper()
        grasp_from_above_joints=[0.59,-1.3376,0,-1.8275,0.0,0.0]
        arm.set_joint_value_target(grasp_from_above_joints)
        succ = arm.go()
        if succ:
            return 'succ'
        else:
            return 'failed'
        
class AR_adjustment(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : GOTO_SHELF')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3: 
            return'tries'
        clear_octo_client()
        scene.remove_world_object()
        #Takeshi gets close to the shelf
        starter.call()
        succ = False
        X_OFFSET = 0.0
        Y_OFFSET = 0.19
        Z_OFFSET = 0.135

        THRESHOLD = 0.025

        hcp = head.get_current_joint_values()
        hcp[0] = -0.1
        hcp[1] = -0.5
        head.set_joint_value_target(hcp)
        head.go()
        while(True):
            try:
                t = tfbuff.lookup_transform('hand_palm_link', 'ar_marker/201', rospy.Time(0) )
#         t = tfbuff.lookup_transform('hand_palm_link', 'ar_marker/4000', rospy.Time(0) )
                traf = t.transform.translation
                rospy.sleep(.6)
        # tiny_move_base(y = 0.163)
                ex = x = traf.x + X_OFFSET
                ey = -traf.y + Y_OFFSET
                print(ex, ey)
                if abs(ex) > THRESHOLD:
                    tiny_move_base(x = ex)#, y = -traf.y + Y_OFFSET)
                if abs(ey) > THRESHOLD:
                    tiny_move_base(y = ey)
                if (abs(ex) <= THRESHOLD and abs(ey) <= THRESHOLD):
                    hcp[0] = 0
                    head.set_joint_value_target(hcp)
                    head.go()
                    succ = True
                    break
            except:
                hcp = head.get_current_joint_values()
                hcp[0] -= 0.1   
                print(hcp[0])
                head.set_joint_value_target(hcp)
                head.go()
                if hcp[0] < -1:
                    hcp[0] = 0.1
                    head.set_joint_value_target(hcp)
                    head.go()
                    print('Ive lost the reference')
                    succ = False
                    break
        if succ:
            stopper.call()
            return 'succ'
        else:
            return 'failed'
class Color_adjustment(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : color adjustment')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3: 
            return'tries'
        clear_octo_client()
        scene.remove_world_object()
        #Takeshi scans the shelf
        succ = False
        THRESHOLD = 15
        while(True):
            goalPos = [258.61,261.75]
            [currentPos] = color_segmentator()
#     print(currentPos)
            ex = -(goalPos[0]-currentPos[0]) 
            ey = (goalPos[1]-currentPos[1])
            print(ex, ey)
            if abs(ex) > THRESHOLD:
                tiny_move_base(x = ex, std_time=0.1, MAX_VEL=0.01)#, y = -traf.y + Y_OFFSET)
                rospy.sleep(0.5)
            if abs(ey) > THRESHOLD:
                tiny_move_base(y = ey, std_time=0.1, MAX_VEL=0.01)
                rospy.sleep(0.5)
            if (abs(ex) <= THRESHOLD and abs(ey) <= THRESHOLD):
                print('done')
                succ = True
                break
        if succ:
            return 'succ'
        else:
            return 'failed'

    
class Grasp_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : robot neutral pose')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
        scene.remove_world_object()
        #Takeshi neutral
        acp = arm.get_current_joint_values()
        acp[0] = 0.56
        arm.set_joint_value_target(acp)
        arm.go()
        close_gripper()
        rospy.sleep(0.5)
        acp = arm.get_current_joint_values()
        acp[0] = 0.69
        arm.set_joint_value_target(acp)
        arm.go()
        rospy.sleep(0.3)
        tiny_move_base(x = -0.5)
        arm.set_named_target('go')
        succ = arm.go()
        if succ:
            return 'succ'
        else:
            return 'failed'
        
#Initialize global variables and node
def init(node_name):

    global lis, broad, tf_static_broad, tfbuff,scene, rgbd, gripper, head, whole_body, hand_cam
    global arm, goal, navclient, clear_octo_client, service_client, base_vel_pub, starter, stopper

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('Grab_cassette')
    head = moveit_commander.MoveGroupCommander('head')
    gripper = moveit_commander.MoveGroupCommander('gripper')
    whole_body = moveit_commander.MoveGroupCommander('whole_body_light')
    arm =  moveit_commander.MoveGroupCommander('arm')
    
    tfbuff = tf2_ros.Buffer()
    lis = tf2_ros.TransformListener(tfbuff)
    broad = tf2_ros.TransformBroadcaster()
    tf_static_broad = tf2_ros.StaticTransformBroadcaster()
    whole_body.set_workspace([-6.0, -6.0, 6.0, 6.0]) 
    
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    rgbd = RGBD()
    hand_cam = HandRGB()
    goal = MoveBaseGoal()
    
    navclient = actionlib.SimpleActionClient('/move_base/move', MoveBaseAction)
    clear_octo_client = rospy.ServiceProxy('/clear_octomap', Empty)
    service_client = rospy.ServiceProxy('/segment_2_tf', Trigger)
    base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)
    starter = rospy.ServiceProxy('/marker/start_recognition',Empty)
    stopper = rospy.ServiceProxy('/marker/stop_recognition',Empty)
#     arm.set_max_velocity_scaling_factor
    head.set_planning_time(0.3)
    head.set_num_planning_attempts(1)
    gripper.set_planning_time(0.3)
    gripper.set_num_planning_attempts(1)

#Entry point    
if __name__== '__main__':
    print("Takeshi STATE MACHINE...")
    init("takeshi_smach_20")
    sm = smach.StateMachine(outcomes = ['END'])     #State machine, final state "END"
    sm.userdata.sm_counter = 0

    with sm:
        #State machine for grasping on Table
        smach.StateMachine.add("INITIAL",Initial(),transitions = {'failed':'INITIAL', 'succ':'FIND_AR_MARKER', 'tries':'END'}) 
        smach.StateMachine.add("FIND_AR_MARKER",Find_AR_marker(),transitions = {'failed':'END', 'succ':'PRE_GRASP_POSE', 'tries':'FIND_AR_MARKER'}) 
        smach.StateMachine.add("PRE_GRASP_POSE",Pre_grasp_pose(),transitions = {'failed':'END', 'succ':'AR_ADJUSTMENT', 'tries':'PRE_GRASP_POSE'}) 
        smach.StateMachine.add("AR_ADJUSTMENT",AR_adjustment(),transitions = {'failed':'END', 'succ':'COLOR_ADJUSTMENT', 'tries':'COLOR_ADJUSTMENT'}) 
        smach.StateMachine.add("COLOR_ADJUSTMENT",Color_adjustment(),transitions = {'failed':'END', 'succ':'GRASP_TABLE', 'tries':'END'})
        smach.StateMachine.add("GRASP_TABLE",Grasp_table(),transitions = {'failed':'END', 'succ':'END', 'tries':'GRASP_TABLE'})

      

    outcome = sm.execute()



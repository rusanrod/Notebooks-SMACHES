#!/usr/bin/env python3
import sys
import smach
import rospy
import cv2 as cv
import numpy as np
from std_srvs.srv import Empty
import moveit_commander
import moveit_msgs.msg
import tf2_ros as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from utils_takeshi import *
from grasp_utils import *

    
def correct_points(low_plane=0.0, high_plane=0.2):

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
            img = cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]), (0,0,0), 2)
            # calculate moments for each contour
            if (cY > reg_ly and cY < reg_hy  ):
                image_aux = iimmg[boundRect[1]:boundRect[1]+max(boundRect[2],boundRect[3]),boundRect[0]:boundRect[0]+max(boundRect[2],boundRect[3])]
                images.append(image_aux)
                cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
                cv2.putText(img, f'centroid_{i}_{cX},{cY}',    (cX - 25, cY - 25)   ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                #print ('cX,cY',cX,cY)
                xyz = []

                for jy in range (boundRect[0], boundRect[0] + boundRect[2]):
                    for ix in range(boundRect[1], boundRect[1] + boundRect[3]):
                        aux = (np.asarray((points_data['x'][ix,jy], points_data['y'][ix,jy], points_data['z'][ix,jy])))
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

    cents = np.asarray(cents)
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
        #global trans_hand
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
# State initial
        gripper.open()

        try:
            clear_octo_client()
        except:
            print('cant clear octomap')
        AR_stopper.call()
        #Takeshi neutral
        arm.set_named_target('go')
        arm.go()
        gripper.steady()
        head.set_named_target('neutral')
        succ = head.go()
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
        # State Find AR marker
        try:
            AR_starter.call()
            clear_octo_client()
        except:
            print('cant clear octomap')
        rospy.sleep(0.2)
        arm.set_named_target('go')
        arm.go()
        hcp = gaze.relative(1,0,0.7)
        head.set_joint_value_target(hcp)
        head.go()
        succ = False
        flag = 1
        while not succ:
            trans,rot = tf_man.getTF(target_frame='ar_marker/201', ref_frame='base_link')
            print(trans)
            succ = type(trans) is not bool
            if succ:
                tf_man.pub_static_tf(pos=trans, rot=rot, point_name='cassette', ref='base_link')
                rospy.sleep(0.5)
                while not tf_man.change_ref_frame_tf(point_name='cassette', new_frame='map'):
                    print('change reference frame is not done yet')
                arm.set_named_target('go')
                arm.go()
            else:
                if flag == 1:
                    hcp = gaze.relative(0.7,0.5,0.7)
                    head.set_joint_value_target(hcp)
                    head.go()
                    flag += 1
                    rospy.sleep(0.3)
                elif flag == 2:
                    hcp = gaze.relative(0.7,-0.5,0.7)
                    head.set_joint_value_target(hcp)
                    head.go()
                    flag += 1
                    rospy.sleep(0.3)
                else:
                    head.set_named_target('neutral')
                    head.go() 
                    return 'failed'
                
        if succ:
            return 'succ'
        else:
            return 'failed'
        
class AR_alignment(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : AR alignment ')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
        # State AR alignment
        AR_stopper.call()
        head.set_named_target('neutral')
        head.go()
        arm.set_named_target('go')
        arm.go()
        succ = False
        THRESHOLD = 0.08
        talk("I am going to align with the table")
        while not succ:
            try:
                trans, rot = tf_man.getTF(target_frame='cassette', ref_frame='base_link')
                euler = tf.transformations.euler_from_quaternion(rot)
                theta = euler[2]
                e = theta + 1.57
                print(e)
                if abs(e) < THRESHOLD:
#                     talk("ready")
                    succ = True
                else:
                    rospy.sleep(0.1)
                    grasp_base.tiny_move(velT = 0.4*e, std_time=0.1)
            except:
                hcp[0] -= 0.2
                if hcp[0] > -1.2:
                    hcp[0] = 0.0
                head.set_joint_value_target(hcp)
                head.go()
                
        
        if succ:
            return 'succ'
        else:
            return 'failed'

class Pre_grasp_pose(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : Pre grasp pose ')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
#         clear_octo_client()
        # State Pre grasp pose
        # talk("I will reach the cassette")
        gripper.open()
        arm.set_named_target('neutral')
        arm.go()
        result = False
        while not result:
            trans,_ = tf_man.getTF(target_frame='cassette', ref_frame='odom')
            pos, rot =tf_man.getTF(target_frame='hand_palm_link', ref_frame='odom')
            pos[2] = trans[2] + 0.1
            pose_goal = set_pose_goal(pos=pos, rot=rot)

            arm.set_start_state_to_current_state()
            arm.set_pose_target(pose_goal)
            result = arm.go()
        result = False
        tf_man.pub_static_tf(point_name='grasp', pos=[0,0,0.11], ref='hand_palm_link')
        while not result:
            pos, rot = tf_man.getTF(target_frame='grasp', ref_frame='odom')
            rospy.sleep(2)

            pose_goal = set_pose_goal(pos = pos, rot=rot)
            arm.set_start_state_to_current_state()
            arm.set_pose_target(pose_goal)
            result = arm.go()
        succ = False
        THRESHOLD = 0.01
        while not succ:
            trans,_ = tf_man.getTF(target_frame='cassette', ref_frame='hand_palm_link')
            print(trans)
            if abs(trans[1])>THRESHOLD:
                grasp_base.tiny_move(velY=-0.2*trans[1], std_time=0.2, MAX_VEL=0.3)
            else:
                succ = True
        succ = False
        while not succ:
            trans,_ = tf_man.getTF(target_frame='cassette', ref_frame='hand_palm_link')
            if abs(trans[2])>0.1:
                grasp_base.tiny_move(velX=0.2*trans[2], std_time=0.2 , MAX_VEL=0.3)
            else:
                succ = True
        return 'succ'
            
class Grasp_pose(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : robot grasp pose')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==4:
            return 'failed'
                # State grasp table

        gripper.open()
        # trans,rot= tf_man.getTF(target_frame='cassette', ref_frame='odom')
        rospy.sleep(0.5)
        succ = False
        # shift = 0.12
        # attempt = 0
        # flag = False
        """while not succ:
            attempt += 1
            shift -= 0.01
            if shift <= 0.0:
                shift = 0.15
                grasp_base.tiny_move(velX=-0.05, std_time=0.4, MAX_VEL=0.1)
                if flag :
                    break
                flag = True
            tf_man.pub_static_tf(pos=[0,0,shift] ,point_name='grasp1', ref='hand_palm_link')
            rospy.sleep(0.8)
            tf_man.change_ref_frame_tf(point_name='grasp1', new_frame='odom')
            rospy.sleep(0.8)
            trans, rot = tf_man.getTF(target_frame='grasp1', ref_frame='odom')
            # print(trans, trans1)
            pose_goal = set_pose_goal(pos=trans, rot=rot)

            arm.set_start_state_to_current_state()
            arm.set_pose_target(pose_goal)
            (succ, plan,_,_) = arm.plan()
            print(f'plan status: {succ} in attempt {attempt}')"""
        grasp_base.tiny_move(velX=0.03,std_time=0.4,MAX_VEL=0.03)
        succ = True
        if succ:
            # arm.execute(plan)
            return 'succ'
        else:
            print('plan failed')
            return 'failed'
        
class Grasp_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Grasp from table')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
        # State post grasp pose
        gripper.close()
        gripper.close()
        rospy.sleep(0.5)
        acp = arm.get_current_joint_values()
        while not acp[0]:
            acp = arm.get_current_joint_values()
        acp[0] += 0.03
        arm.set_joint_value_target(acp)
        arm.go()
        rospy.sleep(1)
        
        force = wrist.get_force()
        print(force)
        if abs(force[0]) > 1.5:
            rospy.sleep(0.5)
            talk('i have the cassette')
            return 'succ'
            # print('si')
        else:
            gripper.open()
            rospy.sleep(0.5)
            acp[0]-=0.03
            arm.set_joint_value_target(acp)
            arm.go()
            talk('i will try again')
            return 'tries'

class Post_grasp_pose(smach.State):###example of a state definition.
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Post grasp pose')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        eef_link = arm.get_end_effector_link()
        ##Adding objects to planning scene
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "hand_palm_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z =  0.20  # below the panda_hand frame
        box_name = "box"
        scene.add_box(box_name, box_pose, size=(0.075, 0.05, 0.075))
        rospy.sleep(0.7)
        ##attaching object to the gripper
        grasping_group = "gripper"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)
        arm.set_named_target('go')
        succ = arm.go()
        if succ:
            return 'succ'
        else:
            return 'failed'

class Forward_shift(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Forward shift')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        if self.tries==3:
            return 'tries'
# State forward shift
        grasp_base.tiny_move(velX = 0.02, std_time=0.5, MAX_VEL=0.03)
        succ = True
        if succ:
            return 'succ'
        else:
            return 'failed'
        
#Initialize global variables and node
def init(node_name):

    global head, wbw, wbl, arm, tf_man, gaze, robot, scene
    global rgbd, hand_cam, wrist, gripper, grasp_base, clear_octo_client, service_client, AR_starter, AR_stopper

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('Pruebas_de_graspeo_v2')
    head = moveit_commander.MoveGroupCommander('head')
    wbw = moveit_commander.MoveGroupCommander('whole_body_weighted')
    wbl = moveit_commander.MoveGroupCommander('whole_body_light')
    arm =  moveit_commander.MoveGroupCommander('arm')
    wbw.set_workspace([-6.0, -6.0, 6.0, 6.0]) 
    wbl.set_workspace([-6.0, -6.0, 6.0, 6.0])  
    
    tf_man = TF_MANAGER()
    rgbd = RGBD()
    hand_cam = HAND_RGB()
    wrist = WRIST_SENSOR()
    gripper = GRIPPER()
    grasp_base = OMNIBASE()
    gaze = GAZE()

    clear_octo_client = rospy.ServiceProxy('/clear_octomap', Empty)
    AR_starter = rospy.ServiceProxy('/marker/start_recognition',Empty)
    AR_stopper = rospy.ServiceProxy('/marker/stop_recognition',Empty)
    
    head.set_planning_time(0.3)
    head.set_num_planning_attempts(1)
#Entry point    
if __name__== '__main__':
    print("Takeshi STATE MACHINE...")
    init("takeshi_smach_20")
    sm = smach.StateMachine(outcomes = ['END'])     #State machine, final state "END"

    with sm:
        #State machine for grasping on Table
        smach.StateMachine.add("INITIAL",Initial(),transitions = {'failed':'INITIAL', 'succ':'FIND_AR_MARKER', 'tries':'END'}) 
        smach.StateMachine.add("FIND_AR_MARKER",Find_AR_marker(),transitions = {'failed':'END', 'succ':'AR_ALIGNMENT', 'tries':'FIND_AR_MARKER'}) 
        smach.StateMachine.add("AR_ALIGNMENT",AR_alignment(),transitions = {'failed':'AR_ALIGNMENT', 'succ':'PRE_GRASP_POSE', 'tries':'AR_ALIGNMENT'}) 
        smach.StateMachine.add("PRE_GRASP_POSE",Pre_grasp_pose(),transitions = {'failed':'END', 'succ':'GRASP_POSE', 'tries':'PRE_GRASP_POSE'}) 
        smach.StateMachine.add("FORWARD_SHIFT",Forward_shift(),transitions = {'failed':'FORWARD_SHIFT', 'succ':'GRASP_TABLE', 'tries':'FORWARD_SHIFT'}) 
        smach.StateMachine.add("GRASP_POSE",Grasp_pose(),transitions = {'failed':'END', 'succ':'GRASP_TABLE', 'tries':'GRASP_POSE'}) 
        smach.StateMachine.add("GRASP_TABLE",Grasp_table(),transitions = {'failed':'END', 'succ':'POST_GRASP_POSE', 'tries':'FORWARD_SHIFT'})
        smach.StateMachine.add("POST_GRASP_POSE",Post_grasp_pose(),transitions = {'failed':'END', 'succ':'END', 'tries':'END'})
      

    outcome = sm.execute()



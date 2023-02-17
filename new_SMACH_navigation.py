#!/usr/bin/env python3
import sys
import smach
import rospy
import cv2 as cv
import numpy as np
from std_srvs.srv import Empty
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped
import tf2_ros as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from utils_takeshi import *
from grasp_utils import *
# from goto import *
            
            ########## Functions for takeshi states ##########
        
    ##### Define state INITIAL #####
#Estado inicial de takeshi, neutral
class Initial(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : robot neutral pose')
        self.tries+=1
        rospy.loginfo(f'Try:{self.tries} of 5 attepmpts') 
        if self.tries==3:
            return 'tries'
        # State initial
        eef_link = arm.get_end_effector_link()
        scene.remove_attached_object(eef_link, name='box')
        rospy.sleep(0.5)
        ##Remove objects
        scene.remove_world_object('box')
        gripper.open()
        arm.clear_pose_targets()
        wb.clear_pose_targets()
        rospy.sleep(0.5)
        talk('I am ready to start')
        rospy.sleep(1.0)
        try:
            clear_octo_client()
            AR_stopper.call()
        except:
            rospy.loginfo('Cant clear octomap')
        #Takeshi neutral
        arm.set_named_target('go')
        arm.go()
        gripper.steady()
        head.set_named_target('neutral')
        succ = head.go()
        # status = goto('cassette_demo')
        status = 1
        move_base(known_location='cassette_demo')
        while status != 3:
            status = NS.get_status()
            # print(status)
            rospy.sleep(0.5)
        return 'succ'
class Find_AR_marker(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : Find AR marker ')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        if self.tries > 1:
            grasp_base.tiny_move(velX=0.05,std_time=0.5, MAX_VEL=0.05)
        # State Find AR marker
        try:
            AR_starter.call()
            clear_octo_client()
        except:
            rospy.loginfo('Cant clear octomap')
        rospy.sleep(0.8)
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
            if type(trans) is not bool:
                tf_man.pub_static_tf(pos=trans, rot=rot, point_name='cassette', ref='base_link')
                rospy.sleep(0.8)

                if not tf_man.change_ref_frame_tf(point_name='cassette', new_frame='map'):
                    rospy.sleep(0.8)
                    rospy.loginfo('Change reference frame is not done yet')
                succ = True
                return 'succ'
            else:
                gazeY = 0.5 
                if flag == 1:
                    flag += 1
                elif flag == 2:
                    gazeY = -0.5
                    flag += 1
                else:
                    head.set_named_target('neutral')
                    head.go()
                    talk('I did not find any marker, I will try again')
                    rospy.sleep(0.7)
                    return 'tries'
                hcp = gaze.relative(0.7,gazeY,0.7)
                head.set_joint_value_target(hcp)
                head.go()
                rospy.sleep(0.9)

class Set_position(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : AR alignment ')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        if self.tries == 1:
            talk('I will set the optimal position')
            rospy.sleep(0.7)
        AR_stopper.call()
        head.set_named_target('neutral')
        head.go()
        arm.set_named_target('go')
        arm.go()
        succ = False
        THRESHOLD = 0.04
        while not succ:
            trans,_ = tf_man.getTF(target_frame='cassette', ref_frame='arm_flex_link')
            _, rot = tf_man.getTF(target_frame='cassette', ref_frame='base_link')
            if type(trans) is not bool and type(rot) is not bool:
                eX, eY, _ = trans
                eX -= 0.45
                euler = tf.transformations.euler_from_quaternion(rot)
                theta = euler[2]
                eT = theta + 1.57
                rospy.loginfo("Distance to goal: {:.2f} m., {:.2f} m., {:.2f} rad.".format(eX, eY, eT))
                if abs(eY) < THRESHOLD:
                    eY = 0
                if abs(eX) < THRESHOLD:
                    eX = 0
                if abs(eT) < THRESHOLD:
                    eT = 0
                succ =  eX == 0 and eY == 0 and eT == 0
                grasp_base.tiny_move(velX=0.2*eX, velY=0.3*eY, velT = 0.4*eT, std_time = 0.2, MAX_VEL = 0.3) #Pending test
        return 'succ'

class Pre_grasp_pose(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : Pre grasp pose ')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        if self.tries==1:
            talk("I will reach the cassette")
        elif self.tries == 5:
            return 'failed'
        # State Pre grasp pose
        gripper.open()
        arm.set_named_target('neutral')
        arm.go()
        #Set pose
        trans = False
        pos = False
        while (type(trans) is bool) and (type(pos) is bool) :
            trans,_ = tf_man.getTF(target_frame='cassette', ref_frame='hand_palm_link')
            pos,_ = tf_man.getTF(target_frame='hand_palm_link', ref_frame='odom')
        tf_man.pub_static_tf(point_name='goal_pose', pos=[trans[0]+0.09, 0.0, 0.11], ref='hand_palm_link')
        rospy.sleep(0.5)
        pos, rot = tf_man.getTF(target_frame='goal_pose', ref_frame='odom')

        if type(pos) is not bool:
            pose_goal = set_pose_goal(pos=pos, rot=rot)
            arm.set_start_state_to_current_state() 
            arm.set_pose_target(pose_goal)
            succ, plan, _, _ = arm.plan()
            if succ:
                arm.execute(plan)
            else:
                rospy.loginfo('I could not plan to goal pose')
                return 'tries'
        else:
            return 'tries'
        #Align along X and Y axis
        succ = False
        THRESHOLD = 0.01
        while not succ:
            trans,_ = tf_man.getTF(target_frame='cassette', ref_frame='hand_palm_link')
            if type(trans) is not bool:
                _, eY, eX = trans
                rospy.loginfo("Distance to goal: {:.2f}, {:.2f}".format(eX, eY))
                if abs(eY) < THRESHOLD:
                    eY = 0
                if abs(eX) < THRESHOLD:
                    eX = 0
                succ =  eX == 0 and eY == 0
                    # grasp_base.tiny_move(velY=-0.4*trans[1], std_time=0.2, MAX_VEL=0.3)
                grasp_base.tiny_move(velX=0.3*eX, velY=-0.4*eY, std_time=0.2, MAX_VEL=0.3) #Pending test
        return 'succ'
            
class Grasp_pose(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : robot grasp pose')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        if self.tries==4:
            return 'failed'
                # State grasp table
        try:
            calibrate_wrist.call()
        except:
            rospy.loginfo('Wrist not calibrated')
        gripper.open()
        # trans,rot= tf_man.getTF(target_frame='cassette', ref_frame='odom')
        rospy.sleep(0.5)
        # succ = False
        grasp_base.tiny_move(velX=0.03,std_time=0.3,MAX_VEL=0.03)
        rospy.sleep(0.5)
        # succ = True
        return 'succ'
        
class Grasp_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Grasp from table')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 

        if self.tries==5:
            return 'failed'

        # State post grasp pose
        clear_octo_client()
        # rospy.loginfo(f'fuerza inicial {FI}')
        rospy.sleep(0.5)
        # gripper.steady()
        gripper.close()
        # succ = False
        trans, rot = tf_man.getTF(target_frame='hand_palm_link', ref_frame='odom') 
        trans[2] += 0.03
        # while not succ:
        pose_goal = set_pose_goal(pos = trans, rot =rot)
        # wb.set_start_state_to_current_state()
        # wb.set_pose_target(pose_goal)
        arm.set_start_state_to_current_state()
        arm.set_pose_target(pose_goal)
        # succ, plan, _, _ = wb.plan()
        succ, plan, _, _ = arm.plan()

        if succ:
            # wb.execute(plan)
            arm.execute(plan)
        else:
            return 'tries'
        rospy.sleep(0.8)
        force = wrist.get_force()
        force = np.array(force)
        weight = force[0]
        rospy.loginfo("Weight detected of {:.3f} Newtons".format(weight))
        if  weight >  0.01:
            talk('I have the cassette')
            rospy.sleep(0.7)
            return 'succ'
        else:
            trans, _tf_man.getTF(target_frame= 'hand_r_finger_tip_frame', ref_frame= 'hand_l_finger_tip_frame')
            t = np.array(trans)
            dist = np.linalg.norm(t)
            if dist > 0.005:
                talk('succeed')
                rospy.sleep(0.7)
                return 'succ'
            else:
                gripper.open()
                rospy.sleep(0.1)
                trans[2] -= 0.03
                pose_goal = set_pose_goal(pos = trans, rot=rot)
                arm.set_start_state_to_current_state()
                arm.set_pose_target(pose_goal)
                succ, plan, _, _ = arm.plan()
                if succ:
                    arm.execute(plan)
                talk('I will try again')
                rospy.sleep(0.7)
                return 'tries'

class Post_grasp_pose(smach.State):###example of a state definition.
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Post grasp pose')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        
        if self.tries==3:
            return 'tries'
        grasp_base.tiny_move(velX=-0.1,std_time=1.5, MAX_VEL=0.1)
        rospy.sleep(0.7)
        arm.set_named_target('go')
        succ = arm.go()
        head.set_named_target('neutral')
        head.go()
        if succ:
            talk('Now I will find the player')
            rospy.sleep(0.7)
            return 'succ'
        else:
            return 'tries'
class Attach_object(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Attach object')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        
        if self.tries==3:
            return 'tries'
        # State Delete objects
        eef_link = arm.get_end_effector_link()
        # Adding objects to planning scene
        box_pose = PoseStamped()
        box_pose.header.frame_id = "hand_palm_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z =  0.08  # below the panda_hand frame
        box_name = "box"
        scene.add_box(box_name, box_pose, size=(0.13, 0.05, 0.13))
        rospy.sleep(0.7)
        ##attaching object to the gripper
        grasping_group = "gripper"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)
        # grasp_base.tiny_move(velX = -0.05, std_time=1.0, MAX_VEL=0.05)
        # arm.set_named_target('go')
        # arm.go()
        trans, _= tf_man.getTF(target_frame='cassette', ref_frame='base_link')
        hcp = gaze.relative(trans[0],trans[1],trans[2])
        head.set_joint_value_target(hcp)
        succ = head.go()
        if succ:
            return 'succ'
        else:
            return 'tries'
class Delete_objects(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Delete objects')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        
        if self.tries==3:
            return 'tries'
        # State Delete objects
        ##Detaching objects
        eef_link = arm.get_end_effector_link()
        scene.remove_attached_object(eef_link, name='box')

        ##Remove objects
        scene.remove_world_object('box')
        gripper.open()
        grasp_base.tiny_move(velX = -0.05, std_time=1.0, MAX_VEL=0.05)
        arm.set_named_target('go')
        arm.go()
        return 'succ'

class Forward_shift(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Forward shift')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        
        if self.tries==3:
            return 'tries'
# State forward shift
        ##Detaching objects
        # scene.remove_attached_object(eef_link, name=box_name)

        ##Remove objects
        # scene.remove_world_object(box_name)
        grasp_base.tiny_move(velX = 0.02, std_time=0.5, MAX_VEL=0.03)
        succ = True
        if succ:
            return 'succ'
        else:
            return 'failed'

class Player_search(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Player search')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        
        if self.tries==3:
            return 'failed'
        # State Player AR search
        # status = goto('player_demo')
        status = 1
        move_base(known_location='player_demo')
        while status !=3:
            status = NS.get_status()
            # print(status)
            rospy.sleep(0.5)
        # return 'succ'
        try:
            AR_starter.call()
            clear_octo_client()
        except:
            rospy.loginfo('Cant clear octomap')
        rospy.sleep(0.2)
        hcp = gaze.relative(1,0,0.7)
        head.set_joint_value_target(hcp)
        head.go()
        succ = False
        flag = 1
        while not succ:
            trans,rot = tf_man.getTF(target_frame='ar_marker/7')
            print(trans)
            if type(trans) is not bool:
                tf_man.pub_static_tf(pos=trans, rot=rot, point_name='player')
                rospy.sleep(0.8)
                succ = True
                return 'succ'
            else:
                gazeY = -0.5 
                if flag == 1:
                    flag += 1
                elif flag == 2:
                    gazeY = 0.5
                    flag += 1
                else:
                    head.set_named_target('neutral')
                    head.go()
                    talk('I did not find any marker, I will try again') 
                    return 'tries'
                hcp = gaze.relative(0.5,gazeY,0.7)
                head.set_joint_value_target(hcp)
                head.go()
                rospy.sleep(0.3)

class Player_alignment(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : player alignment')
        self.tries+=1
        rospy.loginfo(f'Try:{self.tries} of 5 attepmpts') 
        if self.tries==3:
            return 'tries'
        #Player alignment
        arm.set_named_target('neutral')
        arm.go()
        try:
            acp = arm.get_current_joint_values()
            acp[4] = 1.60
            arm.set_joint_value_target(acp)
            arm.go()
        except:
            return 'tries'
        succ = False
        THRESHOLD = 0.01
        while not succ:
            trans,rot = tf_man.getTF(target_frame='player', ref_frame='hand_palm_link')
            euler = tf.transformations.euler_from_quaternion(rot)
            theta = euler[1] 
            if type(trans) is not bool:
                eY, _, eX = trans
                eY = -eY 
                eX -= 0.40
                eT = theta
                rospy.loginfo("Distance to goal: {:.3f}, {:.3f}".format(eX, eY))
                if abs(eY) < THRESHOLD:
                    eY = 0
                if abs(eX) < THRESHOLD:
                    eX = 0
                if abs(eT) < THRESHOLD:
                    # talk("I am aligned")
                    eT=0
                succ =  eX == 0 and eY == 0
#                 rospy.sleep(1.0)
                grasp_base.tiny_move(velX=0.2
                                     *eX, velY=0.4*eY, velT=-0.4*eT, std_time=0.2, MAX_VEL=0.3) #Pending test
        return 'succ'
class Pre_place_pose(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Pre place pose')
        self.tries+=1
        rospy.loginfo(f'Try:{self.tries} of 5 attepmpts') 
        # if self.tries > 1:
        clear_octo_client()
        if self.tries==5:
            return 'failed'
        head.set_named_target('neutral')
        head.go()
        arm.set_goal_tolerance(0.001)

        pos, rot = tf_man.getTF(target_frame='hand_palm_link', ref_frame='odom')
        trans,_ = tf_man.getTF(target_frame='player', ref_frame='hand_palm_link')
        pos[2]+=(-trans[1]+0.0135)
        tf_man.pub_static_tf(point_name='goal',pos=pos, rot=rot, ref='odom')
        tf_man.pub_static_tf(point_name='goal_pose', pos=[0,0,0.04], ref='goal')
        pos, rot = tf_man.getTF(target_frame='goal_pose', ref_frame='odom')
        pose_goal = set_pose_goal(pos=pos, rot = rot)
        arm.set_start_state_to_current_state()
        arm.set_pose_target(pose_goal)
        succ,plan,_,_=arm.plan()
        if succ:
            arm.execute(plan)
            return 'succ'
        else:
            return 'tries'
class Get_player_edge(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : robot neutral pose')
        self.tries+=1
        rospy.loginfo(f'Try:{self.tries} of 5 attepmpts') 
        if self.tries==3:
            return 'tries'
        succ = False
        while not succ:
            force = wrist.get_force()
            if force[2] <= 0.0:
                grasp_base.tiny_move(velX=0.05, std_time=0.01)
            else:
                vel = -0.01
                grasp_base.tiny_move(velX=vel, std_time=0.1)
                p,r = tf_man.getTF(target_frame = 'hand_r_finger_tip_frame')
                tf_man.pub_static_tf(point_name='goal_pose',pos=p, rot=r, ref='map')
                succ = True
        return 'succ'

class Pre_place_cassette(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Pre place cassette ')
        self.tries+=1
        rospy.loginfo(f'Try:{self.tries} of 5 attepmpts') 
        if self.tries==3:
            return 'tries'
        clear_octo_client()
        succ = False
        THRESHOLD = 0.01
        while not succ:
            trans,_ = tf_man.getTF(target_frame='player', ref_frame='hand_palm_link')
            if type(trans) is not bool:
                eY, _, eX = trans
                eY = -eY - 0.10
                eX -= 0.17
                rospy.loginfo("Distance to goal: {:.3f}, {:.3f}".format(eX, eY))
                if abs(eY) < THRESHOLD:
                    eY = 0
                if abs(eX) < THRESHOLD:
                    eX = 0
                succ =  eX == 0 and eY == 0
#                 rospy.sleep(1.0)
                grasp_base.tiny_move(velX=0.2
                                     *eX, velY=0.45*eY, std_time=0.25, MAX_VEL=0.3) #Pending test
        return 'succ'

class Place_cassette(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : robot neutral pose')
        self.tries+=1
        rospy.loginfo(f'Try:{self.tries} of 5 attepmpts') 
        if self.tries==3:
            return 'tries'
        succ = False
        e1= 1.0
        n1 = 0.015 * 0.9 #Amplificacion de la fuerza frontal
        n2 = 0.02       #Amplificacion de la fuerza lateral
        c = 1.05        #
        THRESHOLD = 0.001
        TIMEOUT = 8
        start_time = rospy.Time().now().to_sec()
        while not succ:
            d, _ = tf_man.getTF(target_frame='player', ref_frame='hand_r_finger_tip_frame')
            d_GP,_ = tf_man.getTF(target_frame='goal_pose', ref_frame='hand_r_finger_tip_frame')
            force = wrist.get_force()
            dist = np.array([c*(d[1]),0.0])
            if d_GP[1] >= 0.0:
                n2 = 0.025
            else:
                n2 = 0.001
            f = np.array([n2*force[2],-n1*force[0]])
            speed = e1*dist-f
            #print(f'Distance to AR x:{d[0]}, z:{d[1]}')
            succ =  abs(dist[0]) < THRESHOLD
            if not succ:
                grasp_base.tiny_move(velX = speed[0], velY = speed[1], std_time = 0.02, MAX_VEL = 0.03)
            if rospy.Time().now().to_sec() - start_time > TIMEOUT:
                return 'tries'
        return 'succ'

class Push_cassette(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : robot neutral pose')
        self.tries+=1
        rospy.loginfo(f'Try:{self.tries} of 5 attepmpts') 
        if self.tries==3:
            return 'tries'
        gripper.steady()
        succ = False
        THRESHOLD = 0.07
        #Put hand behind cassette
        while not succ:
            d, _ = tf_man.getTF(target_frame='player', ref_frame='hand_r_finger_tip_frame')
            succ = d[1] > THRESHOLD
            if not succ:
                grasp_base.tiny_move(velX = -0.04,std_time=0.2)
        gripper.close()
        succ = False
        THRESHOLD = 0.04
        while not succ:
            d, _ = tf_man.getTF(target_frame='player', ref_frame='hand_r_finger_tip_frame')
            succ = d[1] < THRESHOLD
            if not succ:
                grasp_base.tiny_move(velX = 0.04,std_time=0.2)
        gripper.steady()
        return 'succ'
#Initialize global variables and node
def init(node_name):

    global head, wb, arm, tf_man, gaze, robot, scene, calibrate_wrist #wbw, wbl
    global rgbd, hand_cam, wrist, gripper, grasp_base, clear_octo_client, service_client, AR_starter, AR_stopper, NS

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('takeshi_smach_20')

    head = moveit_commander.MoveGroupCommander('head')
    wb = moveit_commander.MoveGroupCommander('whole_body')
    arm =  moveit_commander.MoveGroupCommander('arm')
    # wbl = moveit_commander.MoveGroupCommander('whole_body_light')
    #wbw.set_workspace([-6.0, -6.0, 6.0, 6.0]) 
    #wbl.set_workspace([-6.0, -6.0, 6.0, 6.0])  
    wb.set_workspace([-6.0, -6.0, 6.0, 6.0])  
    
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    
    tf_man = TF_MANAGER()
    rgbd = RGBD()
    hand_cam = HAND_RGB()
    wrist = WRIST_SENSOR()
    gripper = GRIPPER()
    grasp_base = OMNIBASE()
    gaze = GAZE()
    NS = nav_status()

    clear_octo_client = rospy.ServiceProxy('/clear_octomap', Empty)
    calibrate_wrist = rospy.ServiceProxy('/hsrb/wrist_wrench/readjust_offset',Empty)
    AR_starter = rospy.ServiceProxy('/marker/start_recognition',Empty)
    AR_stopper = rospy.ServiceProxy('/marker/stop_recognition',Empty)
    
    head.set_planning_time(2.0)
    arm.set_planning_time(10.0)
    head.set_num_planning_attempts(1)
    wb.set_num_planning_attempts(10)
    # wb.allow
#Entry point    
if __name__== '__main__':
    print("Takeshi STATE MACHINE...")
    init("takeshi_smach_20")
    sm = smach.StateMachine(outcomes = ['END'])     #State machine, final state "END"

    with sm:
        #State machine for grasping on Table
        # smach.StateMachine.add("GET_PLAYER_EDGE",Get_player_edge(),transitions = {'failed':'GET_PLAYER_EDGE', 'succ':'PRE_PLACE_CASSETTE', 'tries':'GET_PLAYER_EDGE'})

        smach.StateMachine.add("INITIAL",Initial(),transitions = {'failed':'INITIAL', 'succ':'FIND_AR_MARKER', 'tries':'END'}) 
        
        smach.StateMachine.add("FIND_AR_MARKER",Find_AR_marker(),transitions = {'failed':'END', 'succ':'SET_POSITION', 'tries':'FIND_AR_MARKER'}) 
        smach.StateMachine.add("SET_POSITION",Set_position(),transitions = {'failed':'SET_POSITION', 'succ':'PRE_GRASP_POSE', 'tries':'SET_POSITION'})

        smach.StateMachine.add("PRE_GRASP_POSE",Pre_grasp_pose(),transitions = {'failed':'END', 'succ':'GRASP_POSE', 'tries':'PRE_GRASP_POSE'}) 
        smach.StateMachine.add("FORWARD_SHIFT",Forward_shift(),transitions = {'failed':'FORWARD_SHIFT', 'succ':'GRASP_TABLE', 'tries':'FORWARD_SHIFT'}) 
        smach.StateMachine.add("GRASP_POSE",Grasp_pose(),transitions = {'failed':'END', 'succ':'GRASP_TABLE', 'tries':'GRASP_POSE'}) 
        smach.StateMachine.add("GRASP_TABLE",Grasp_table(),transitions = {'failed':'DELETE_OBJECTS', 'succ':'ATTACH_OBJECT', 'tries':'GRASP_TABLE'})
        smach.StateMachine.add("ATTACH_OBJECT",Attach_object(),transitions = {'failed':'POST_GRASP_POSE', 'succ':'POST_GRASP_POSE', 'tries':'POST_GRASP_POSE'})
        smach.StateMachine.add("DELETE_OBJECTS",Delete_objects(),transitions = {'failed':'END', 'succ':'END', 'tries':'END'})
        smach.StateMachine.add("POST_GRASP_POSE",Post_grasp_pose(),transitions = {'failed':'END', 'succ':'PLAYER_SEARCH', 'tries':'POST_GRASP_POSE'})
        smach.StateMachine.add("PLAYER_SEARCH",Player_search(),transitions = {'failed':'PLAYER_SEARCH', 'succ':'PLAYER_ALIGNMENT', 'tries':'PLAYER_SEARCH'})
        smach.StateMachine.add("PLAYER_ALIGNMENT",Player_alignment(),transitions = {'failed':'PLAYER_ALIGNMENT', 'succ':'PRE_PLACE_POSE', 'tries':'PLAYER_ALIGNMENT'})
        smach.StateMachine.add("PRE_PLACE_POSE",Pre_place_pose(),transitions = {'failed':'PLAYER_SEARCH', 'succ':'GET_PLAYER_EDGE', 'tries':'PRE_PLACE_POSE'})

        smach.StateMachine.add("GET_PLAYER_EDGE",Get_player_edge(),transitions = {'failed':'GET_PLAYER_EDGE', 'succ':'PRE_PLACE_CASSETTE', 'tries':'GET_PLAYER_EDGE'})
        smach.StateMachine.add("PRE_PLACE_CASSETTE",Pre_place_cassette(),transitions = {'failed':'PRE_PLACE_CASSETTE', 'succ':'PLACE_CASSETTE', 'tries':'PRE_PLACE_CASSETTE'})
        smach.StateMachine.add("PLACE_CASSETTE",Place_cassette(),transitions = {'failed':'PLACE_CASSETTE', 'succ':'PUSH_CASSETTE', 'tries':'END'})
        smach.StateMachine.add("PUSH_CASSETTE",Push_cassette(),transitions = {'failed':'PUSH_CASSETTE', 'succ':'END', 'tries':'PUSH_CASSETTE'})
        # smach.StateMachine.add("PRE_PLACE_POSE",Pre_place_pose(),transitions = {'failed':'PLAYER_SEARCH', 'succ':'END', 'tries':'PRE_PLACE_POSE'})
        # 

    outcome = sm.execute()

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
# from numba import njit

            
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
        talk('I am ready to start')
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
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        if self.tries==3:
            return 'tries'
        # State Find AR marker
        try:
            AR_starter.call()
            clear_octo_client()
        except:
            rospy.loginfo('Cant clear octomap')
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
            if type(trans) is not bool:
                tf_man.pub_static_tf(pos=trans, rot=rot, point_name='cassette', ref='base_link')
                rospy.sleep(0.8)

                while not tf_man.change_ref_frame_tf(point_name='cassette', new_frame='map'):
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
                    return 'tries'
                hcp = gaze.relative(0.7,gazeY,0.7)
                head.set_joint_value_target(hcp)
                head.go()
                rospy.sleep(0.3)
        
class AR_alignment(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : AR alignment ')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        if self.tries == 1:
            talk("I am going to align with the table")
        # State AR alignment
        AR_stopper.call()
        head.set_named_target('neutral')
        head.go()
        arm.set_named_target('go')
        arm.go()
        succ = False
        THRESHOLD = 0.08
        while not succ:
            try:
                trans, rot = tf_man.getTF(target_frame='cassette', ref_frame='base_link')
                euler = tf.transformations.euler_from_quaternion(rot)
                theta = euler[2]
                e = theta + 1.57
                rospy.loginfo("Its missing to turn {:.2f} radians".format(e))
                if abs(e) < THRESHOLD:
                    talk("I am aligned")
                    succ = True
                    return 'succ'
                else:
                    rospy.sleep(0.1)
                    grasp_base.tiny_move(velT = 0.4*e, std_time=0.1)
            except:
                succ = True
                return 'tries'

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
        wb.set_start_state_to_current_state()
        wb.set_pose_target(pose_goal)
        succ, plan, _, _ = wb.plan()
        if succ:
            wb.execute(plan)
        else:
            return 'tries'
        rospy.sleep(1.0)
        force = wrist.get_force()
        force = np.array(force)
        weight = np.linalg.norm(force)
        rospy.loginfo("Weight detected of {:.3f} Newtons".format(weight))
        if  weight >  0.1:
            rospy.sleep(0.5)
            talk('I have the cassette')
            return 'succ'
        else:
            gripper.open()
            rospy.sleep(0.5)
            trans[2] -= 0.03
            pose_goal = set_pose_goal(pos = trans, rot=rot)
            arm.set_start_state_to_current_state()
            arm.set_pose_target(pose_goal)
            succ, plan, _, _ = arm.plan()
            if succ:
                arm.execute(plan)
            talk('I will try again')
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
            talk('I have finished my task')
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
class Far_AR_search(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('STATE : Far AR search')
        self.tries+=1
        print('Try',self.tries,'of 5 attepmpts') 
        
        if self.tries==3:
            return 'failed'
# State far ar search
        acp = arm.get_current_joint_values()
        if len(acp) > 0:
            acp[0] = 0.2
            succ = arm.go()
        if succ:
            return 'succ'
        else:
            return 'failed'
        
#Initialize global variables and node
def init(node_name):

    global head, wb, arm, tf_man, gaze, robot, scene, calibrate_wrist #wbw, wbl
    global rgbd, hand_cam, wrist, gripper, grasp_base, clear_octo_client, service_client, AR_starter, AR_stopper

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

    clear_octo_client = rospy.ServiceProxy('/clear_octomap', Empty)
    calibrate_wrist = rospy.ServiceProxy('/hsrb/wrist_wrench/readjust_offset',Empty)
    AR_starter = rospy.ServiceProxy('/marker/start_recognition',Empty)
    AR_stopper = rospy.ServiceProxy('/marker/stop_recognition',Empty)
    
    head.set_planning_time(0.3)
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
        smach.StateMachine.add("INITIAL",Initial(),transitions = {'failed':'INITIAL', 'succ':'FIND_AR_MARKER', 'tries':'END'}) 
        smach.StateMachine.add("FIND_AR_MARKER",Find_AR_marker(),transitions = {'failed':'END', 'succ':'AR_ALIGNMENT', 'tries':'FIND_AR_MARKER'}) 
        smach.StateMachine.add("FAR_AR_SEARCH",Far_AR_search(),transitions = {'failed':'FAR_AR_SEARCH', 'succ':'FIND_AR_MARKER', 'tries':'FIND_AR_MARKER'}) 
        smach.StateMachine.add("AR_ALIGNMENT",AR_alignment(),transitions = {'failed':'AR_ALIGNMENT', 'succ':'PRE_GRASP_POSE', 'tries':'AR_ALIGNMENT'}) 
        smach.StateMachine.add("PRE_GRASP_POSE",Pre_grasp_pose(),transitions = {'failed':'END', 'succ':'GRASP_POSE', 'tries':'PRE_GRASP_POSE'}) 
        smach.StateMachine.add("FORWARD_SHIFT",Forward_shift(),transitions = {'failed':'FORWARD_SHIFT', 'succ':'GRASP_TABLE', 'tries':'FORWARD_SHIFT'}) 
        smach.StateMachine.add("GRASP_POSE",Grasp_pose(),transitions = {'failed':'END', 'succ':'GRASP_TABLE', 'tries':'GRASP_POSE'}) 
        smach.StateMachine.add("GRASP_TABLE",Grasp_table(),transitions = {'failed':'DELETE_OBJECTS', 'succ':'ATTACH_OBJECT', 'tries':'GRASP_TABLE'})
        smach.StateMachine.add("ATTACH_OBJECT",Attach_object(),transitions = {'failed':'POST_GRASP_POSE', 'succ':'POST_GRASP_POSE', 'tries':'POST_GRASP_POSE'})
        smach.StateMachine.add("DELETE_OBJECTS",Delete_objects(),transitions = {'failed':'END', 'succ':'END', 'tries':'END'})
        smach.StateMachine.add("POST_GRASP_POSE",Post_grasp_pose(),transitions = {'failed':'END', 'succ':'END', 'tries':'POST_GRASP_POSE'})

    outcome = sm.execute()



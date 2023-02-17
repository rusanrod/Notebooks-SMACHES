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
        rospy.sleep(0.5)
        ##Remove objects
        h_search = [0.0,-0.80]
        arm_search = [0.0, 0.0, 1.65, -1.5707, 0.0, 0.0]
        # arm.set_joint_value_target(arm_search)
        arm.set_named_target('go')
        arm.go()
        head.set_joint_value_target(h_search)
        succ = head.go()
        if succ:
        	AR_starter.call()
        	return 'succ'
        else:
            return 'failed'
class Search_AR(smach.State):###example of a state definition.
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : PROTO_STATE')
        self.tries+=1
        floor_pose = [0.0,-2.47,0.0,0.86,-0.032,0.0]
        arm.set_joint_value_target(floor_pose)
        arm.go()
        if self.tries > 1:
        	grasp_base.tiny_move(velX=0.05, MAX_VEL=0.03)
        trans,_ = tf_man.getTF(target_frame='ar_marker/4', ref_frame='hand_palm_link')
        if type(trans) is not bool:
        	rospy.sleep(0.5)
        	return 'succ'
        return 'tries'

class Pre_grasp_pose(smach.State):###example of a state definition.
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : PROTO_STATE')
        clear_octo_client()
        gripper.open()
        succ = True
        if succ:
        	gripper.open()
        	return 'succ'
        else:
        	return 'tries'

class Grasp_floor(smach.State):###example of a state definition.
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : PROTO_STATE')
        succ = False
        THRESHOLD = 0.02
        while not succ:
        	trans,_ = tf_man.getTF(target_frame='ar_marker/4', ref_frame='hand_palm_link')
        	if type(trans) is not bool:
        		_, eY, eX = trans
        		eX -= 0.05
        		rospy.loginfo("Distance to goal: {:.3f}, {:.3f}".format(eX, eY))
        		if abs(eY) < THRESHOLD:
        			eY = 0
        		if abs(eX) < THRESHOLD:
        			eX = 0
        		succ =  eX == 0 and eY == 0
        		grasp_base.tiny_move(velX=0.2*eX, velY=-0.4*eY, std_time=0.2, MAX_VEL=0.3)
        	gripper.close()
        return 'succ'

class Neutral(smach.State):###example of a state definition.
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : PROTO_STATE')
        arm.set_named_target('neutral')
        arm.go()
        head.set_named_target('neutral')
        head.go()
        talk('done')
        return 'succ'

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
        smach.StateMachine.add("INITIAL",Initial(),transitions = {'failed':'INITIAL', 'succ':'SEARCH_AR', 'tries':'INITIAL'}) 
        smach.StateMachine.add("SEARCH_AR",Search_AR(),transitions = {'failed':'END', 'succ':'PRE_GRASP_POSE', 'tries':'SEARCH_AR'}) 
        smach.StateMachine.add("PRE_GRASP_POSE",Pre_grasp_pose(),transitions = {'failed':'END', 'succ':'GRASP_FLOOR', 'tries':'PRE_GRASP_POSE'}) 
        smach.StateMachine.add("GRASP_FLOOR",Grasp_floor(),transitions = {'failed':'END', 'succ':'NEUTRAL', 'tries':'GRASP_FLOOR'}) 
        smach.StateMachine.add("NEUTRAL",Neutral(),transitions = {'failed':'END', 'succ':'END', 'tries':'END'}) 

    outcome = sm.execute()



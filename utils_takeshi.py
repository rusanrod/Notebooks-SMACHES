# -*- coding: utf-8 -*-

#from utils import *
import moveit_msgs.msg
#from gazebo_ros import gazebo_interface
import matplotlib.pyplot as plt




##### KNOWN LOCATIONS #####
kl_mess1 = [1.04, 0.3, 90]
kl_tray =  [2.318411366833172, 0.09283744344925589, -90]
kl_box1 =  [-0.04168256822546347, 2.427268271720426, -90]
kl_table1 = [1.04, 1.2, 90]
kl_table2= [0 , 1.2,90] 
kl_drawers=  [0.06, 0.5, -90]


##### ARM #####
arm_grasp_from_above = [0.19263830140116414,
 -2.2668981568652917,
 -0.007358947463759424,
 -0.9939144210462025,
 -0.17365421548386273,
 0.0]
arm_grasp_from_above_table = [0.41349380130577407,
 -1.671584191489468,
 -0.02774372779356371,
 -1.5952436225825641,
 0.22362492457833927,
 0.0]
arm_grasp_table=[0.41349380130577407,
 -1.671584191489468,
 -0.02774372779356371,
 0.0,
 0.22362492457833927,
 0.0]
arm_grasp_floor = [-1.5151551103007697e-05,
 -2.4,
 -0.2620865401925543,
 0.7019536624449207,
 0.20120924571306453,
 0.0]
arm_train_pose = [0.033749214744071214,
 -2.1204421063180217,
 -1.3982377978814715,
 -1.7296544561013807,
 2.135675364707808,
 0.0]
arm_ready_to_place = [0.03999320441056991,
 -0.4729690540086997,
 0.19361475012179108,
 -1.5269847787383313,
 -0.009753879176134461,
 0.0]
arm_high_drawer=[0.2539946870715912,
 -1.6765040634258677,
 -0.02776609034055033,
 0.0726899567834991,
 1.5763667194117463,
 0.0]

##### GRASP 
ungrasped=[-0.00047048998088961014,
 -0.03874743486886725,
 -0.04825256513113274,
 0.038463464485261056,
 -0.03874743486886725]
grasped=[0.12814103131904275,
 -0.30672794406396453,
 0.21972794406396456,
 0.13252877558892262,
 -0.30672794406396453]


########## Util Functions ##########

def cart2spher(x,y,z):
    ro= np.sqrt(x**2+y**2+z**2)
    th=np.arctan2(y,x)
    phi=np.arctan2((np.sqrt(x**2+y**2)),z)
    return np.asarray((ro,th,phi))


def spher2cart(ro,th,phi):
    x= ro * np.cos(th)* np.sin(phi)
    y= ro * np.sin(th)* np.sin(phi)
    z= ro*  np.cos(th)
    return np.asarray((x,y,z))

def point_2D_3D(points_data, px_y, px_x):
    ##px pixels /2D world  P1 3D world
    P = np.asarray((points_data[px_y, px_x]['x'], points_data[px_y, px_x]['y'], points_data[px_y, px_x]['z']))
    return P

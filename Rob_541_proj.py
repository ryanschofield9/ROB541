#TO DO CHECK IF WE NEED ALL THE IMPORTS 
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from scipy.optimize import curve_fit

from geometry_msgs.msg import PoseStamped, TransformStamped, Point, TwistStamped, Vector3
from tf2_ros import TransformException

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController
import numpy as np

from std_msgs.msg import Float32, Int64, Bool

from sensor_msgs.msg import JointState
import time as t
import matplotlib.pyplot as plt
import math 

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
)
from rclpy.action import ActionClient

import datetime
import logging

from ground_truth.geomotion import (
    utilityfunctions as ut,
    rigidbody as rb)
from ground_truth import simplediffkinematicchain as sdkc



#from filterpy.kalman import KalmanFilter

class Proj(Node):
    def __init__(self):
        super().__init__('ROB541_Proj')
        #Create timer 
        self.pub_timer = self.create_timer(0.1, self.main_control)

        #Create tf buffer and listener 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #Create Callback group
        self.service_handler_group = ReentrantCallbackGroup()
        
        #Create clients 
        self.enable_servo = self.create_client(
            Trigger, "/servo_node/start_servo", callback_group=self.service_handler_group
        )
        self.disable_servo = self.create_client(
            Trigger, "/servo_node/stop_servo", callback_group=self.service_handler_group
        )
        self.switch_ctrl = self.create_client(
            SwitchController, "/controller_manager/switch_controller", callback_group=self.service_handler_group
        )

        self.moveit_planning_client = ActionClient(self, MoveGroup, "move_action")

        #inital states
        self.servo_active = False #if the servos have been activated 


        #constant variables 
        self.forward_cntr = 'forward_position_controller' #name of controller that uses velocity commands 
        self.joint_cntr = 'scaled_joint_trajectory_controller' # name of controller that uses joint commands 
        self.base_frame = 'base_link' #base frame that doesn't move 
        self.tool_frame = 'tool0' #frame that the end effector is attached to  

        #start servoing and switch to forward position controller 
        self.start_servo()
        self.switch_controller(self.joint_cntr, self.forward_cntr)

        self.G2 = rb.SE3 
        self.links = [self.G2.element([0.162, 0, 0, 0, 0, 0]), self.G2.element([0.425, 0, 0, 0, 0,0]), self.G2.element([0.392, 0, 0, 0, 0, 0]), self.G2.element([0.1016, 0, 0, 0, 0, 0]), self.G2.element([0.1016, 0, 0, 0, 0, 0]), self.G2.element([0.0508, 0, 0, 0, 0, 0])]
        self.joint_axes = [self.G2.Lie_alg_vector([0, 0, 0, 0, 0, 1])] * 6
        self.desired_vel = np.array([[0],[0],[0], [0], [0], [0.5]])
        # Create a kinematic chain
        self.kc = sdkc.DiffKinematicChain(self.links, self.joint_axes)
        self.kc.set_configuration([-0.157, -1.623, 1.728, -3.246, 0.157, 0.0])
        self.J_Ad = self.kc.Jacobian_Ad(6, self.G2,'world')
        self.joint_names = [ 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        
        
    def main_control (self): 

        new_joint_angles = self.kc.calc_new_joint_pose(0.1, self.desired_vel, self.kc)
        print(new_joint_angles)
        #print(f"New joint angles: {new_joint_angles}")
        self.J_Ad = self.kc.Jacobian_Ad(6, self.G2,'world')
        self.send_joint_pos(self.joint_names, new_joint_angles)
  
    
    def rotate_to_w(self, angle):
        #rotate the tool to the given angle  
        names = self.joint_names 
        pos = self.joints 

        #for all the joints, use the current angle for all joints, but wrist 3. Set wrist 3 to the given angle  
        for idx, vals in enumerate (names):
            if vals == "wrist_3_joint":
                pos[idx]= angle

        self.send_joint_pos(names, pos)  
        
                
    def start_servo(self):
        #start the arms servos 
        print("in start")
        if self.servo_active:
            #if the servo is already active 
            print ("Servo is already active")
        else:
            #if the servo has not yet been activated 
            self.enable_servo.call_async(Trigger.Request()) #make a service call to activate the servos 
            self.active = True #set the servo active flag as true 
            print("Servo has been activated") 
        return
    
    def switch_controller(self, act, deact):
        #activate the act controllers given and deactivate the deact controllers given 
        switch_ctrl_req = SwitchController.Request(
            activate_controllers = [act], deactivate_controllers= [deact], strictness = 2
            ) #create request for activating and deactivating controllers with a strictness level of STRICT 
        self.switch_ctrl.call_async(switch_ctrl_req) #make a service call to switch controllers 
        print(f"Activated: {act}  Deactivated: {deact}")
            
        return
    
    def send_joint_pos(self, joint_names, joints):
        #make a service call to moveit to go to given joint values 
        #for n, p in zip (joint_names, joints):
        #    print(f"{n}: {p}")
        #print(f"NOW SENDING GOAL TO MOVE GROUP")
        joint_constraints = [JointConstraint(joint_name=n, position=p) for n, p in zip(joint_names, joints)]
        kwargs = {"joint_constraints": joint_constraints}
        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest(
            group_name="ur_manipulator",
            goal_constraints=[Constraints(**kwargs)],
            allowed_planning_time=5.0,
        ) #create a service request of given joint values and an allowed planning time of 5 seconds 

        goal_msg.planning_options = PlanningOptions(plan_only=False)

        self.moveit_planning_client.wait_for_server() #wait for service to be active 
        future = self.moveit_planning_client.send_goal_async(goal_msg) #make service call 
        future.add_done_callback(self.goal_complete) #set done callback

    def goal_complete(self, future):
            #function that is called once a service call is made to moveit_planning 
            rez = future.result()
            if not rez.accepted:
                print("Planning failed!")
                return
            else:
                print("Plan succeeded!")

    def callback_joints(self,msg ):
        #function that saves the current joint names and positions 
        self.joint_names = msg.name
        self.joints= msg.position

   

def convert_tf_to_pose(tf: TransformStamped):
    #take the tf transform and turn that into a position  
    pose = PoseStamped() #create a pose which is of type Pose stamped
    pose.header = tf.header
    tl = tf.transform.translation
    pose.pose.position = Point(x=tl.x, y=tl.y, z=tl.z)
    pose.pose.orientation = tf.transform.rotation

    return pose


def main(args=None):
    rclpy.init(args=args)
    proj = Proj()
    rclpy.spin(proj)
    rclpy.shutdown ()

if __name__ == '__main__':
   main()
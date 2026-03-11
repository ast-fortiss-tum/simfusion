
import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"

import roslaunch.parent
import roslaunch.rlutil
import rospy
import math
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TwistStamped, PointStamped
from std_msgs.msg import Float32, ColorRGBA
from visualization_msgs.msg import MarkerArray
from autoware_msgs.msg import DetectedObjectArray, DetectedObject, Lane
from geometry_msgs.msg import Point32
import threading
import time
import subprocess
from os import kill
from shapely.geometry import Point, Polygon


from simulations.simple_sim.lanelet_operations import LaneletOperations
from simulations.simple_sim.launch_operations import LaunchOperations
from simulations.simple_sim.simout_operations import SimoutOperations
from simulations.simple_sim.object_operations import ObjectOperations, ObjectType
from simulations.simple_sim.subscribers import Subscribers

import logging


class AutowareHelper:
    """
    A helper class for interacting with Autoware via ROS. This class manages vehicle state,
    goals, obstacles, and collects simulation data for analysis.
    """

    def __init__(self):
        # ROS Node Initialization
        rospy.init_node('scenario_simulation', anonymous=True)


        self.object_operations = ObjectOperations()
        self.lanelet_operations = LaneletOperations()
        self.launch_operations = LaunchOperations()
        self.simout_operations = SimoutOperations()
        self.subscriber = Subscribers(self.simout_operations, self.lanelet_operations, self.object_operations)

    def publish_initial_pose(self, x, y, h):
        nearest_point = self.lanelet_operations.process_point((x, y))
        self.object_operations.publish_initial_pose(nearest_point[0], nearest_point[1], h)


    def publish_goal(self, x, y):
        nearest_point = self.lanelet_operations.process_point((x, y))
        self.object_operations.publish_goal(nearest_point[0], nearest_point[1])
        self.subscriber.set_sim_goal_pose(nearest_point[0], nearest_point[1])

    def publish_velocity(self, linear_velocity, angular_velocity):
        self.object_operations.publish_velocity(linear_velocity, angular_velocity)



    def start_object_publisher(self, sim_end_time):
        self.subscriber.set_start_time(rospy.Time.now().to_sec())
        self.subscriber.set_sim_end_time(sim_end_time)
        self.object_operations.start_object_publisher(sim_end_time)


    def create_object(self, x_start, y_start, z_start, obj_type :ObjectType, value=0, heading=0, x_end= 0,  y_end = 0, z_end=0.0, speed= 0): # External interface for use in autoware_simulation.py; start_time, sim_duration
        """
        Schedule an object to be added and removed based on start_time and duration
        """
        self.object_operations.create_object(x_start, y_start, z_start, obj_type, value, heading, x_end, y_end, z_end, speed)


    # Utility Methods
    def get_results(self):
        """
        Returns the current state of the ego vehicle and obstacles.
        """        
        return self.simout_operations.get_results()

    def clear_state(self):
        """
        Resets the state for a new simulation.
        """
        self.simout_operations.clear()
        self.object_operations.clear()
        #self.lanelet_operations.clear()
        self.launch_operations.clear()

    def kill_ros(self):
        subprocess.run(["pkill", "-f", "roslaunch"])
        subprocess.run(["pkill", "-f", "roscore"])
        subprocess.run(["pkill", "-f", "scenario_simulation"])

        logging.warning(f"ROS Services killed.")
    
    def check_ros(self):
        import rosgraph
        try:
            rosgraph.Master('/rosnode').getPid('/scenario_simulation')
            print("ROS master is alive")
        except:
            print("ROS master is not reachable")
import rospy
from autoware_msgs.msg import DetectedObjectArray, Lane
from geometry_msgs.msg import PoseStamped, TwistStamped
from visualization_msgs.msg import MarkerArray
from shapely.geometry import Polygon
from simulations.simple_sim.utility import Utility
from shapely.geometry import box, Point
from simulations.config import offset_x, offset_y
from opensbt.config import DEFAULT_CAR_LENGTH as CAR_LENGTH
from opensbt.config import DEFAULT_CAR_WIDTH as CAR_WIDTH
from datetime import datetime
import csv

class Subscribers:
    
    def __init__(self, simout_operations, lanelet_operations, object_operations):    
        self.simout_operations = simout_operations
        self.lanelet_operations = lanelet_operations
        self.object_operations = object_operations
        self.sim_end_time = 0
        self.sim_start_time = 0
        self.latest_ego_location_msg = None
        self.latest_obstacle_msg = None 
        self.latest_ego_velocity = None
        self.goal_x = 0
        self.goal_y = 0

        self.ego_csv_file = None
        self.ego_csv_writer = None
        self.ego_log_filename = None
    
        self.start_subscribers()

    def _close_ego_csv(self):
        """Close CSV file"""
        if self.ego_csv_file:
            self.ego_csv_file.close()
            rospy.loginfo(f"Ego CSV log closed: {self.ego_log_filename}")
    
    def _setup_ego_csv(self):
        """Create CSV file for ego tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ego_log_filename = f"ego_log_{timestamp}.csv"
        
        self.ego_csv_file = open(self.ego_log_filename, 'w', newline='')
        self.ego_csv_writer = csv.writer(self.ego_csv_file)
        
        # Write header
        self.ego_csv_writer.writerow([
            'timestamp', 'ros_timestamp', 'sim_time',
            'ego_x', 'ego_y', 'ego_z',
            'velocity', 'within_sim_window'
        ])
        
        rospy.loginfo(f"Ego CSV log created: {self.ego_log_filename}")
    
    def __del__(self):
        #self.target_sub.unregister()
        #self.waypoints_sub.unregister()
        self.lanlet_sub.unregister()
        self.current_pose_sub.unregister()
        self.current_velocity_sub.unregister()
        self.detected_objects_sub.unregister()

        self._close_ego_csv()

    def start_subscribers(self):
        #self.target_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.target_callback)
        #self.waypoints_sub = rospy.Subscriber('/planning/global_path', Lane, self.waypoints_callback)
        self.lanlet_sub = rospy.Subscriber('/planning/lanelet2_map_markers', MarkerArray, self.lanelet_callback)
        self.current_pose_sub = rospy.Subscriber('/localization/current_pose', PoseStamped, self.pose_callback)
        self.current_velocity_sub = rospy.Subscriber('/localization/current_velocity', TwistStamped, self.velocity_callback)
        self.detected_objects_sub = rospy.Subscriber('/detection/detected_objects', DetectedObjectArray, self.obstacle_pose_callback)

    def set_start_time(self, sim_start_time):
        self.sim_start_time = sim_start_time
        #self._setup_ego_csv()  # ← Called here automatically!

    def set_sim_end_time(self, sim_end_time):
        self.sim_end_time = self.sim_start_time + sim_end_time
        self.latest_ego_location_msg = None
        self.latest_obstacle_msg= None
    
    def set_sim_goal_pose(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y


    def obstacle_pose_callback(self, msg):
        """
        Updates the state of obstacles detected by the ego vehicle.
        """
        if Utility.extract_timestamp(msg.header.stamp.secs, msg.header.stamp.nsecs) < self.sim_start_time:
            return
        if Utility.extract_timestamp(msg.header.stamp.secs, msg.header.stamp.nsecs) > self.sim_end_time:
            return
        if len(msg.objects) == 0:
            return
        self.latest_obstacle_msg = msg
        self.simout_operations.add_obstacle_message(msg)

    def pose_callback(self, msg):
        """
        Updates the ego vehicle's state, including location and yaw.
        """
        timestamp = Utility.extract_timestamp(msg.header.stamp.secs, msg.header.stamp.nsecs)
        
        # Check if within simulation window
        before_sim = timestamp < self.sim_start_time
        after_sim = timestamp > self.sim_end_time
        within_window = not (before_sim or after_sim)
        
        # LOG TO CSV (log everything, even filtered messages)
        if self.ego_csv_writer:
            sim_time = timestamp - self.sim_start_time if self.sim_start_time else 0
            self.ego_csv_writer.writerow([
                f"{rospy.Time.now().to_sec():.4f}",  # Wall-clock time
                f"{timestamp:.4f}",                   # Message timestamp
                f"{sim_time:.4f}",                    # Time since sim start
                f"{msg.pose.position.x:.4f}",
                f"{msg.pose.position.y:.4f}",
                f"{msg.pose.position.z:.4f}",
                f"{self.latest_ego_velocity:.4f}" if self.latest_ego_velocity else "0.0000",
                within_window                         # True/False if used
            ])

      
        # Filter based on time window (your original logic)
        if before_sim:
            return
        if after_sim:
            return

        self.object_operations.set_ego_pose((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))

        self.simout_operations.add_ego_pose_message(msg)

        self.latest_ego_location_msg = msg

        self.check_collision_if_synced()
        self.check_reach_goal_pose()
        self.check_passed_obstacle()

    def velocity_callback(self, msg):
        """
        Updates the ego vehicle's velocity, speed, and acceleration.
        """
        # if Utility.extract_timestamp(msg.header.stamp.secs, msg.header.stamp.nsecs) < self.sim_start_time:
        #     return
        # if Utility.extract_timestamp(msg.header.stamp.secs, msg.header.stamp.nsecs) > self.sim_end_time:
        #     return
        
        self.simout_operations.add_ego_velocity_message(msg)
        self.latest_ego_velocity = msg.twist.linear.x

    # Map Boundary Functions
    def lanelet_callback(self, msg):
        """
        Callback to process lanelet boundaries from a ROS topic.
        Converts lanelet markers into polygons.
        """
        for marker in msg.markers:
            if len(marker.points) > 2:
                polygon_points = [(point.x, point.y) for point in marker.points]
                self.lanelet_operations.lanelet_boundaries.append(Polygon(polygon_points))
        rospy.loginfo(f"Loaded {len(self.lanelet_operations.lanelet_boundaries)} lanelet boundaries.")

    def check_collision(self):
        """
        Checks if the ego vehicle collides with any detected object.
        Returns True if a collision is detected, otherwise False.
        """
        ego_pose = self.latest_ego_location_msg
        if ego_pose is None:
            return False  # Ego pose not initialized

        ego_x = ego_pose.pose.position.x
        ego_y = ego_pose.pose.position.y
        # TODO set x,y displacement appropriately
        ego_box = box(ego_x - offset_x * CAR_LENGTH, ego_y - offset_y * CAR_WIDTH, ego_x + (1 - offset_x) * CAR_LENGTH, ego_y + (1 - offset_y) * CAR_WIDTH)  

        for obstacle in self.latest_obstacle_msg.objects:
            obs_x, obs_y, obs_z = obstacle.pose.position.x, obstacle.pose.position.y, obstacle.pose.position.z  # Take latest location
            dim_x, dim_y = obstacle.dimensions.x, obstacle.dimensions.y
            obs_box = box(obs_x - dim_x/2, obs_y - dim_y/2, obs_x + dim_x/2, obs_y + dim_y/2)

            if ego_box.intersects(obs_box):
                print("obs_x", obs_x)
                print("obs_y", obs_y)
                
                print("dim_x:", dim_x)
                print("dim_y:", dim_y)

                print("obs_box", obs_box)

                print("ego_box", ego_box)
            
                rospy.loginfo(f"Collision detected with object ID {obstacle.id}")
                return self.object_operations.stop_object_publisher()

        return False
    

    def check_collision_if_synced(self, tolerance_sec=0.5): # 50ms tolerans
        if self.latest_ego_location_msg is None or self.latest_obstacle_msg is None:
            return False

        t_ego = self.latest_ego_location_msg.header.stamp.to_sec()
        t_obs = self.latest_obstacle_msg.header.stamp.to_sec()

        if abs(t_ego - t_obs) < tolerance_sec:
            return self.check_collision()
        return False
    
    def check_reach_goal_pose(self):
        dx = self.goal_x - self.latest_ego_location_msg.pose.position.x
        dy = self.goal_y - self.latest_ego_location_msg.pose.position.y
        dist = Utility.calculate_magnitude(dx, dy, 0)

        distance_threshold = CAR_LENGTH
        velocity_threshold = 0.2

        if dist < distance_threshold and self.latest_ego_velocity < velocity_threshold:
            rospy.loginfo("Goal reached and vehicle stopped.")
            print("Goal reached and vehicle stopped.")
            return self.object_operations.stop_object_publisher()
        
    def check_passed_obstacle(self):
        ego_x = self.latest_ego_location_msg.pose.position.x
        ego_y = self.latest_ego_location_msg.pose.position.y

        for obstacle in self.latest_obstacle_msg.objects:
            obs_x = obstacle.pose.position.x
            obs_y = obstacle.pose.position.y
            obs_z = obstacle.pose.position.z  # Currently unused, but may be useful for 3D logic
            
            # TODO need to change when scenario is different
            tolerance_x = 1 #1
            tolerance_y = 10 #1.5
            if ego_y - tolerance_y > obs_y or ego_x + tolerance_x < obs_x:
                rospy.logwarn("Passed obstacle at ({}, {}).".format(obs_x, obs_y))
                rospy.logwarn("Stopping scenario.")
                print("Obstacled passed, stopping scenario.")
                self.object_operations.stop_object_publisher()
                return True
        rospy.logwarn("Not passed obstacle yet at ({}, {}).".format(obs_x, obs_y))
        return False

    
"""
    def target_callback(self, msg): # gets overwritten if called a second time (should not happen)
        self.simout_operations.target_dict["location"] = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self.simout_operations.target_dict["orientation"] = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
    
    def waypoints_callback(self, msg): # should only be called once
        if len(msg.waypoints) == 0:
            now = rospy.Time.now()
            now_exact = now.to_sec() + now.to_nsec() / 1e9
            if now_exact >= self.sim_start_time and now_exact <= self.sim_end_time:
                rospy.logwarn("empty waypoints encountered!")
            return
        for waypoint in msg.waypoints:
            self.simout_operations.waypoints_dict["location"].append((waypoint.pose.pose.position.x, waypoint.pose.pose.position.y, waypoint.pose.pose.position.z))
            self.simout_operations.waypoints_dict["orientation"].append((waypoint.pose.pose.orientation.x, waypoint.pose.pose.orientation.y, waypoint.pose.pose.orientation.z, waypoint.pose.pose.orientation.w))
"""

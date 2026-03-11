import rospy
import math
from std_msgs.msg import ColorRGBA
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from geometry_msgs.msg import PoseWithCovarianceStamped, Point32, PoseStamped, TwistStamped
import threading
from enum import Enum
from simulations.simple_sim.utility import Utility
import csv
from datetime import datetime

DISTANCE_TOLERANCE = 5
OBJECT_ID = 0

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class ObjectType(Enum):
    DURATION = "duration"
    DISTANCE = "distance"
    STATIC = "static"

class Object:
    def __init__(self, pose_x, pose_y, pose_z, obj_type: ObjectType, value = 0, heading=0, goal_x = 0, goal_y = 0, goal_z = 0, speed=0):
        global OBJECT_ID
        OBJECT_ID += 1

        self.object_type = obj_type
        self.value = value
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_z = goal_z
        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0

        if self.object_type == ObjectType.STATIC:
            self.status = Status.ACTIVE
        else:
            self.status = Status.PENDING
            x_dist = goal_x - pose_x
            y_dist = goal_y - pose_y
            z_dist = goal_z - pose_z
            if heading != 0:
                self.x_vel = speed * math.cos(-heading)
                self.y_vel = speed * math.sin(-heading)
                self.z_vel = 0
            else:
                distance = Utility.calculate_magnitude(x_dist, y_dist, z_dist)
                if distance != 0:
                    self.x_vel = speed * (x_dist / distance) 
                    self.y_vel = speed * (y_dist / distance)
                    self.z_vel = speed * (z_dist / distance)

        self.obj = DetectedObject()
        self.obj.header.frame_id = "map"

        self.obj.id = OBJECT_ID
        self.obj.label = "obstacle"
        self.obj.color = ColorRGBA(0.0, 255.0, 0.0, 1.) # ColorRGBA(252., 15., 192., 1.)
        self.obj.valid = True

        self.obj.space_frame = "map"
        self.obj.pose.position.x = pose_x
        self.obj.pose.position.y = pose_y
        self.obj.pose.position.z = pose_z
        self.obj.pose.orientation.x = 0.0
        self.obj.pose.orientation.y = 0.0
        self.obj.pose.orientation.z = 0.0
        self.obj.pose.orientation.w = 1.0
        if heading != 0:
            self.obj.pose.orientation.z = math.sin(heading/2)
            self.obj.pose.orientation.w = math.cos(heading/2)
        self.obj.dimensions.x = 1.0
        self.obj.dimensions.y = 1.0
        self.obj.dimensions.z = 1.0
        self.obj.pose_reliable = True
        self.obj.velocity.linear.x = 0 #x_vel / 20 
        self.obj.velocity.linear.y = 0 #y_vel / 20
        self.obj.velocity.linear.z = 0 #z_vel / 20
        self.obj.velocity_reliable = True
        self.obj.acceleration.linear.x = 0
        self.obj.acceleration.linear.y = 0
        self.obj.acceleration.linear.z = 0
        self.obj.acceleration_reliable = True
        self.obj.behavior_state = 0

        self.obj.header.stamp = rospy.Time.now()

        self.obj.convex_hull.polygon.points = [
            Point32(self.obj.pose.position.x - self.obj.dimensions.x/2, self.obj.pose.position.y - self.obj.dimensions.y/2, self.obj.pose.position.z),
            Point32(self.obj.pose.position.x - self.obj.dimensions.x/2, self.obj.pose.position.y + self.obj.dimensions.y/2, self.obj.pose.position.z),
            Point32(self.obj.pose.position.x + self.obj.dimensions.x/2, self.obj.pose.position.y + self.obj.dimensions.y/2, self.obj.pose.position.z),
            Point32(self.obj.pose.position.x + self.obj.dimensions.x/2, self.obj.pose.position.y - self.obj.dimensions.y/2, self.obj.pose.position.z)
        ]

        self.last_update_time = None  # Track last update

    def get_object_publishing_infos(self):
        return self.obj

    # def _moving_object(self):
    #         self.status = Status.ACTIVE
    #         self.obj.pose.position.x += self.x_vel / 20 
    #         self.obj.pose.position.y += self.y_vel / 20
    #         self.obj.pose.position.z += self.z_vel / 20
    def _moving_object(self):
            current_time = rospy.Time.now().to_sec()
            
            if self.last_update_time is None:
                dt = 1.0 / 20.0  # First iteration, assume 20 Hz
            else:
                dt = current_time - self.last_update_time
            
            self.last_update_time = current_time
            
            # Move based on actual elapsed time
            self.obj.pose.position.x += self.x_vel * dt
            self.obj.pose.position.y += self.y_vel * dt
            self.obj.pose.position.z += self.z_vel * dt

            self.obj.velocity.linear.x = self.x_vel 
            self.obj.velocity.linear.y = self.y_vel
            self.obj.velocity.linear.z = self.z_vel

            self.obj.convex_hull.polygon.points = [
            Point32(self.obj.pose.position.x - self.obj.dimensions.x/2, self.obj.pose.position.y - self.obj.dimensions.y/2, self.obj.pose.position.z),
            Point32(self.obj.pose.position.x - self.obj.dimensions.x/2, self.obj.pose.position.y + self.obj.dimensions.y/2, self.obj.pose.position.z),
            Point32(self.obj.pose.position.x + self.obj.dimensions.x/2, self.obj.pose.position.y + self.obj.dimensions.y/2, self.obj.pose.position.z),
            Point32(self.obj.pose.position.x + self.obj.dimensions.x/2, self.obj.pose.position.y - self.obj.dimensions.y/2, self.obj.pose.position.z)
        ]

    def _continue_pending(self):
        pass

    def _check_if_reached_goal(self):
        self.obj.velocity.linear.x = 0 
        self.obj.velocity.linear.y = 0
        self.obj.velocity.linear.z = 0
        
        return Utility.calculate_magnitude(
            self.obj.pose.position.x - self.goal_x,
            self.obj.pose.position.y - self.goal_y,
            self.obj.pose.position.z - self.goal_z
        ) <= DISTANCE_TOLERANCE
            
    def update_based_on_duration(self, duration):
        if self.object_type == ObjectType.DURATION:
            if self.status == Status.PENDING:
                if duration >= self.value:
                    self._moving_object()
                else:
                    self._continue_pending()
            elif self.status == Status.ACTIVE:
                if self._check_if_reached_goal():
                    self.status = Status.INACTIVE
                else:
                    self._moving_object()
        else:
            raise TypeError("This object is distance-based, update_based_on_duration() cannot be used.")

    def update_based_on_distance(self, pose):
        if self.object_type == ObjectType.DISTANCE:
            if self.status == Status.PENDING:
                if  self.value >= Utility.calculate_magnitude(self.obj.pose.position.x - pose[0], self.obj.pose.position.y - pose[1], self.obj.pose.position.z - pose[2]):
                    self._moving_object()
                else:
                    self._continue_pending()
            elif self.status == Status.ACTIVE:
                if self._check_if_reached_goal():
                    self.status = Status.INACTIVE
                else:
                    self._moving_object()
        else:
            raise TypeError("This object is duration-based, update_based_on_distance() cannot be used.")



class ObjectOperations:
    def __init__(self, pub_obj_rate=20): #20
        self.pub_obj_id = 0
        self.pub_obj_rate = pub_obj_rate
        self.stop_publishing = False
        self.obstacles = []
        self.rate = rospy.Rate(self.pub_obj_rate)
        self.lock = threading.Lock()
        self.ego_pose = (0, 0, 0)
        
        self.pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.velocity_pub = rospy.Publisher('/initialvelocity', TwistStamped, queue_size=10)
        self.obstacles_pub = rospy.Publisher('detection/detected_objects', DetectedObjectArray, queue_size=10)
        self.pedestrian_csv_file = None
        self.pedestrian_csv_writer = None

    def clear(self):
        self.pub_obj_id = 0
        self.obstacles = []

    def set_ego_pose(self, ego_pose):
        with self.lock:
            self.ego_pose = ego_pose

    def _get_ego_pose(self):
        return self.ego_pose

    def publish_initial_pose(self, x, y, h):
        """
        Publish the initial pose of the ego vehicle.
        """
        pose_msg = PoseWithCovarianceStamped()
        self.is_sim_done = False

        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = rospy.Time.now()

        pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y = (x, y)
        pose_msg.pose.pose.position.z = 0.0
        if h != 0:
            pose_msg.pose.pose.orientation.z = math.sin(h/2)
            pose_msg.pose.pose.orientation.w = math.cos(h/2)
            pose_msg.pose.pose.orientation.x = 0.0

        rospy.sleep(1)
        self.pose_pub.publish(pose_msg)

    def publish_goal(self, x, y):
        """
        Publish the goal position of the ego vehicle.
        """
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x, goal_msg.pose.position.y = x,y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        rospy.sleep(1)
        self.goal_pub.publish(goal_msg)

    def publish_velocity(self, linear_velocity, angular_velocity):
        """
        Publish the initial velocity of the ego vehicle.
        """
        velocity_msg = TwistStamped()
        velocity_msg.header.frame_id = "base_link"
        velocity_msg.header.stamp = rospy.Time.now()
        velocity_msg.twist.linear.x = linear_velocity
        velocity_msg.twist.angular.z = angular_velocity

        rospy.sleep(1)
        self.velocity_pub.publish(velocity_msg)

    def _setup_pedestrian_csv(self):
        """Create CSV file for pedestrian tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pedestrian_log_{timestamp}.csv"
        
        self.pedestrian_csv_file = open(filename, 'w', newline='')
        self.pedestrian_csv_writer = csv.writer(self.pedestrian_csv_file)
        
        # Write header
        self.pedestrian_csv_writer.writerow([
            'timestamp', 'step', 'ped_id', 'status', 
            'ped_x', 'ped_y', 'ped_z',
            'ego_x', 'ego_y', 'ego_z',
            'distance', 'trigger_distance',
            'vel_x', 'vel_y', 'dt'
        ])
        
        rospy.loginfo(f"Pedestrian CSV log created: {filename}")
        return filename

    def publish_objects(self):
        """
        Publish all static obstacles
        """
        msg = DetectedObjectArray()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.objects = [object.get_object_publishing_infos() for object in self.obstacles]
        self.obstacles_pub.publish(msg)

    def start_object_publisher(self, sim_time):
        """
        Initializes and runs the obstacle publishing simulation.
        This function sets up a separate thread for ROS spinning, clears obstacle states, 
        and runs an update loop to manage scheduled obstacles.
        """

        # Initialize ROS thread for subscribers
        self.t = threading.Thread(target=rospy.spin)
        self.t.daemon = True
        self.stop_publishing = False
        self.t.start()
        self.sim_time = sim_time

#    '     # Set ROS publishing rate
#         self.rate = rospy.Rate(self.pub_obj_rate)
#         self.steps = self.sim_time * self.pub_obj_rate 

#         # Record simulation start time
#         ros_time = rospy.Time.now()
#         self.start_time = ros_time.to_sec()
#         self.simulation_end_time = self.start_time + self.sim_time

#         rospy.loginfo("Obstacle simulation started.")'
        self.steps = self.sim_time * self.pub_obj_rate 
        ros_time = rospy.Time.now()
        self.start_time = ros_time.to_sec()
        self.simulation_end_time = self.start_time + self.sim_time
        
        # ped_csv_filename = self._setup_pedestrian_csv()
        
        rospy.loginfo("Obstacle simulation started.")
        print("Steps for simulation: ", self.steps)
        
        for step in range(self.steps + 1):
            if self.stop_publishing:
                rospy.logwarn("Simulation stopped at step %d/%d", step, self.steps)
                break
            
            duration = rospy.Time.now().to_sec() - self.start_time
            current_ego_pose = self._get_ego_pose()
            
            # Log to CSV
            # self._log_pedestrians_to_csv(step, current_ego_pose)
            
            self.update_objects(duration, current_ego_pose)
            self.publish_objects()
            self.rate.sleep()
        
        # Close CSV
        self._close_pedestrian_csv()
        
        actual_duration = rospy.Time.now().to_sec() - self.start_time
        rospy.loginfo("Obstacle simulation completed. Steps: %d, Actual: %.2fs", 
                      step + 1, actual_duration)
        # rospy.loginfo(f"Pedestrian log saved to: {ped_csv_filename}")
    
        # print("Steps lofi simulation: ", self.steps)
        # # Main loop for publishing obstacles
        # for step in range(self.steps + 1):
        #     duration = rospy.Time.now().to_sec() - self.start_time
        #     if duration >= self.sim_time or self.stop_publishing:
        #         print("Stopping obstacle publisher at step as over sim_time", step)                
        #         break
        #     self.update_objects(duration, self._get_ego_pose())
        #     self.publish_objects()
        #     self.rate.sleep()

        # # Ensure thread cleanup
        # # self.t.join(0.1)

        # rospy.loginfo("Obstacle simulation terminated at %ds", rospy.Time.now().to_sec() - self.start_time)


    def create_object(self, x_start, y_start, z_start, obj_type :ObjectType, value=0, heading=0, x_end= 0,  y_end = 0, z_end=0.0, speed= 0):
        """
        Add and publish a single static obstacle at the given coordinates (at a given time)
        returns id of object
        """
        new_object = Object(x_start, y_start, z_start, obj_type, value, heading, x_end, y_end, z_end, speed)
        print("Object created with object type: ", new_object.object_type, x_start, y_start)

        #with self.lock:
        self.obstacles.append(new_object)
        self.publish_objects()
        
    def stop_object_publisher(self):
        rospy.logwarn("Collision detected and simulation is stopped.")
        self.stop_publishing = True

    def update_objects(self, now, pose):
        for object in self.obstacles:
            if object.object_type == ObjectType.DURATION:
                object.update_based_on_duration(now)
            elif object.object_type == ObjectType.DISTANCE:
                object.update_based_on_distance(pose)

    def _close_pedestrian_csv(self):
        """Close CSV file"""
        if self.pedestrian_csv_file:
            self.pedestrian_csv_file.close()
            rospy.loginfo("Pedestrian CSV log closed")

    def _log_pedestrians_to_csv(self, step, ego_pose):
        """Log current state of all pedestrians"""
        current_time = rospy.Time.now().to_sec()
        
        for obstacle in self.obstacles:
            distance = Utility.calculate_magnitude(
                obstacle.obj.pose.position.x - ego_pose[0],
                obstacle.obj.pose.position.y - ego_pose[1],
                obstacle.obj.pose.position.z - ego_pose[2]
            )
            
            dt = 0.0
            if hasattr(obstacle, 'last_update_time') and obstacle.last_update_time is not None:
                dt = current_time - obstacle.last_update_time
            
            self.pedestrian_csv_writer.writerow([
                f"{current_time:.4f}",
                step,
                obstacle.obj.id,
                obstacle.status.value,
                f"{obstacle.obj.pose.position.x:.4f}",
                f"{obstacle.obj.pose.position.y:.4f}",
                f"{obstacle.obj.pose.position.z:.4f}",
                f"{ego_pose[0]:.4f}",
                f"{ego_pose[1]:.4f}",
                f"{ego_pose[2]:.4f}",
                f"{distance:.4f}",
                f"{obstacle.value:.4f}",
                f"{obstacle.x_vel:.4f}",
                f"{obstacle.y_vel:.4f}",
                f"{dt:.4f}"
            ])

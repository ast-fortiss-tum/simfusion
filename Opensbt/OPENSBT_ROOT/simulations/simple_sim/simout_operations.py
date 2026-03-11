from simulations.simple_sim.utility import Utility
import rospy
from bisect import bisect_left
from simulations.config import simout_messages_interval as SIMOUT_MESSAGES_INTERVAL
import numpy as np
import rospy
# Interval for simout messages that coming from ros 

class SimoutOperations:

    def __init__(self):

        self.__ego_pose_messages = []
        self.__ego_velocity_messages = []
        self.__detected_obstacle_messages = []


        self.__ego_information_dict = {
            "location" : [],
            "velocity" : [],
            "speed" : [],
            "acceleration" : [],
            "yaw" : []
        }

        self.__obstacle_dict = {
            "location" : [],
            "velocity" : [],
            "speed" : [],
            "acceleration" : [],
            "yaw" : []
        }

    def clear(self):

        self.__ego_pose_messages = []
        self.__ego_velocity_messages = []
        self.__detected_obstacle_messages = []


        self.__ego_information_dict = {
            "location" : [],
            "velocity" : [],
            "speed" : [],
            "acceleration" : [],
            "yaw" : []
        }

        self.__obstacle_dict = {
            "location" : [],
            "velocity" : [],
            "speed" : [],
            "acceleration" : [],
            "yaw" : []
        }


    def __add_message_periodically(self, list, msg):

        if len(list) == 0:
            list.append(msg)
            return True
        else:
            previous_time = Utility.extract_timestamp(list[-1].header.stamp.secs, list[-1].header.stamp.nsecs)
            current_time = Utility.extract_timestamp(msg.header.stamp.secs, msg.header.stamp.nsecs)
            if current_time - previous_time >= SIMOUT_MESSAGES_INTERVAL:
                list.append(msg)
                return True
        return False
    
    def add_ego_pose_message(self, msg):
        """
        Add ego pose message to the list if the time interval is greater than SIMOUT_MESSAGES_INTERVAL
        """
        self.__add_message_periodically(self.__ego_pose_messages, msg)

    def add_ego_velocity_message(self, msg):
        """
        Add ego velocity message to the list if the time interval is greater than SIMOUT_MESSAGES_INTERVAL
        """
        self.__add_message_periodically(self.__ego_velocity_messages, msg)
    
    def add_obstacle_message(self, msg):
        """
        Add obstacle message to the list if the time interval is greater than SIMOUT_MESSAGES_INTERVAL
        """
        self.__add_message_periodically(self.__detected_obstacle_messages, msg)

    def process_messages(self):
        """Process ROS messages and interpolate to common timeline"""
        
        # Validate clean state
        assert len(self.__ego_information_dict["location"]) == 0, "Ego dict not empty! Call clear() first!"
        assert len(self.__obstacle_dict["location"]) == 0, "Obstacle dict not empty! Call clear() first!"
        
        # Initialize timestamp lists
        if "timestamp" not in self.__ego_information_dict:
            self.__ego_information_dict["timestamp"] = []
        if "timestamp" not in self.__obstacle_dict:
            self.__obstacle_dict["timestamp"] = []
        
        # ===== COLLECT RAW EGO DATA =====
        pose_map = {msg.header.stamp.to_sec(): msg for msg in self.__ego_pose_messages}
        velocity_map = {msg.header.stamp.to_sec(): msg for msg in self.__ego_velocity_messages}
        
        if not velocity_map or not pose_map:
            rospy.logwarn("No pose or velocity messages available")
            self.__ego_pose_messages.clear()
            self.__ego_velocity_messages.clear()
            self.__detected_obstacle_messages.clear()
            return
        
        pose_times = sorted(pose_map.keys())
        vel_times = sorted(velocity_map.keys())
        
        # Build velocity interpolation data
        vel_data = {
            'x': [velocity_map[t].twist.linear.x for t in vel_times],
            'y': [velocity_map[t].twist.linear.y for t in vel_times],
            'z': [velocity_map[t].twist.linear.z for t in vel_times]
        }
        
        # Store raw ego data with pose timestamps
        ego_raw_times = []
        ego_raw_locations = []
        ego_raw_velocities = []
        ego_raw_yaws = []
        
        for timestamp in pose_times:
            pose_msg = pose_map[timestamp]
            
            # Interpolate velocity
            vx = np.interp(timestamp, vel_times, vel_data['x'])
            vy = np.interp(timestamp, vel_times, vel_data['y'])
            vz = np.interp(timestamp, vel_times, vel_data['z'])
            velocity = (vx, vy, vz)
            
            ego_raw_times.append(timestamp)
            ego_raw_locations.append((pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z))
            ego_raw_velocities.append(velocity)
            ego_raw_yaws.append(pose_msg.pose.orientation.z)
        
        # ===== COLLECT RAW OBSTACLE DATA =====
        obs_raw_times = []
        obs_raw_locations = []
        obs_raw_velocities = []
        obs_raw_yaws = []
        
        for obstacle_msg in self.__detected_obstacle_messages:
            if len(obstacle_msg.objects) == 0:
                continue
            
            obstacle = obstacle_msg.objects[0]
            timestamp = obstacle_msg.header.stamp.to_sec()
            
            obs_raw_times.append(timestamp)
            obs_raw_locations.append((obstacle.pose.position.x, obstacle.pose.position.y, obstacle.pose.position.z))
            obs_raw_velocities.append((obstacle.velocity.linear.x, obstacle.velocity.linear.y, obstacle.velocity.linear.z))
            obs_raw_yaws.append(obstacle.pose.orientation.z)
        
        if len(obs_raw_times) == 0:
            rospy.logwarn("No obstacle data collected")
            self.__ego_pose_messages.clear()
            self.__ego_velocity_messages.clear()
            self.__detected_obstacle_messages.clear()
            return
        
        # ===== CREATE COMMON TIMELINE =====
        # Find overlapping time range
        t_min = max(ego_raw_times[0], obs_raw_times[0])
        t_max = min(ego_raw_times[-1], obs_raw_times[-1])
        
        common_times = np.arange(t_min, t_max, 0.02)
        
        rospy.loginfo(f"Common timeline: {len(common_times)} points from {t_min:.2f}s to {t_max:.2f}s")
        
        # ===== INTERPOLATE EGO TO COMMON TIMELINE =====
        ego_loc_x = [loc[0] for loc in ego_raw_locations]
        ego_loc_y = [loc[1] for loc in ego_raw_locations]
        ego_loc_z = [loc[2] for loc in ego_raw_locations]
        ego_vel_x = [vel[0] for vel in ego_raw_velocities]
        ego_vel_y = [vel[1] for vel in ego_raw_velocities]
        ego_vel_z = [vel[2] for vel in ego_raw_velocities]
        
        for t in common_times:
            # Interpolate location
            x = np.interp(t, ego_raw_times, ego_loc_x)
            y = np.interp(t, ego_raw_times, ego_loc_y)
            z = np.interp(t, ego_raw_times, ego_loc_z)
            location = (x, y, z)
            
            # Interpolate velocity
            vx = np.interp(t, ego_raw_times, ego_vel_x)
            vy = np.interp(t, ego_raw_times, ego_vel_y)
            vz = np.interp(t, ego_raw_times, ego_vel_z)
            velocity = (vx, vy, vz)
            
            # Interpolate yaw
            yaw = np.interp(t, ego_raw_times, ego_raw_yaws)
            
            # Calculate speed
            speed = Utility.calculate_magnitude(vx, vy, vz)
            
            # Calculate acceleration
            if len(self.__ego_information_dict["timestamp"]) > 0:
                v_prev = self.__ego_information_dict["velocity"][-1]
                t_prev = self.__ego_information_dict["timestamp"][-1]
                dt = t - t_prev
                if dt > 0:
                    a_vector = tuple((vc - vp) / dt for vc, vp in zip(velocity, v_prev))
                    a_magnitude = Utility.calculate_magnitude(*a_vector)
                else:
                    a_magnitude = 0.0
            else:
                a_magnitude = 0.0
            
            # Store
            self.__ego_information_dict["timestamp"].append(t)
            self.__ego_information_dict["location"].append(location)
            self.__ego_information_dict["velocity"].append(velocity)
            self.__ego_information_dict["speed"].append(speed)
            self.__ego_information_dict["yaw"].append(yaw)
            self.__ego_information_dict["acceleration"].append(a_magnitude)
        
        # ===== INTERPOLATE OBSTACLE TO COMMON TIMELINE =====
        obs_loc_x = [loc[0] for loc in obs_raw_locations]
        obs_loc_y = [loc[1] for loc in obs_raw_locations]
        obs_loc_z = [loc[2] for loc in obs_raw_locations]
        obs_vel_x = [vel[0] for vel in obs_raw_velocities]
        obs_vel_y = [vel[1] for vel in obs_raw_velocities]
        obs_vel_z = [vel[2] for vel in obs_raw_velocities]
        
        for t in common_times:
            # Interpolate location
            x = np.interp(t, obs_raw_times, obs_loc_x)
            y = np.interp(t, obs_raw_times, obs_loc_y)
            z = np.interp(t, obs_raw_times, obs_loc_z)
            location = (x, y, z)
            
            # Interpolate velocity
            vx = np.interp(t, obs_raw_times, obs_vel_x)
            vy = np.interp(t, obs_raw_times, obs_vel_y)
            vz = np.interp(t, obs_raw_times, obs_vel_z)
            velocity = (vx, vy, vz)
            
            # Interpolate yaw
            yaw = np.interp(t, obs_raw_times, obs_raw_yaws)
            
            # Calculate speed
            speed = Utility.calculate_magnitude(vx, vy, vz)
            
            # Store (acceleration placeholder)
            self.__obstacle_dict["timestamp"].append(t)
            self.__obstacle_dict["location"].append(location)
            self.__obstacle_dict["velocity"].append(velocity)
            self.__obstacle_dict["speed"].append(speed)
            self.__obstacle_dict["yaw"].append(yaw)
            self.__obstacle_dict["acceleration"].append(0)
        
        # ===== VALIDATION =====
        ego_count = len(self.__ego_information_dict["timestamp"])
        obs_count = len(self.__obstacle_dict["timestamp"])
        
        assert ego_count == obs_count == len(common_times), \
            f"Interpolation failed: ego={ego_count}, obs={obs_count}, common={len(common_times)}"
        
        rospy.loginfo(f"Processed: {ego_count} frames for both ego and obstacle")
        
        # Clear messages
        self.__ego_pose_messages.clear()
        self.__ego_velocity_messages.clear()
        self.__detected_obstacle_messages.clear()

    def get_results(self):
        """
        Returns the current state of the ego vehicle and obstacles.
        """        
        # Process messages
        self.process_messages()
        return self.__ego_information_dict, self.__obstacle_dict
    
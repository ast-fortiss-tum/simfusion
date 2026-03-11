#!/usr/bin/env python3
import rospy

def main():
    # Initialize a temporary node
    rospy.init_node("read_params_node", anonymous=True)

    try:
        speed_Kp = rospy.get_param("/carla_ackermann_control_ego_vehicle/speed_Kp")
        speed_Ki = rospy.get_param("/carla_ackermann_control_ego_vehicle/speed_Ki")
        speed_Kd = rospy.get_param("/carla_ackermann_control_ego_vehicle/speed_Kd")
        print(f"speed_Kp={speed_Kp}, speed_Ki={speed_Ki}, speed_Kd={speed_Kd}")
    except KeyError as e:
        print(f"Parameter not found: {e}")
    except rospy.ROSException as e:
        print(f"Could not connect to ROS master: {e}")

if __name__ == "__main__":
    main()
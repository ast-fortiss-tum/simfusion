import rospy
from geometry_msgs.msg import PoseStamped, Twist
import os
import sys

class CarlaHelper:
    def __init__(self):
        # ROS node initialization
        os.environ["ROS_MASTER_URI"] = "http://localhost:11312"
        rospy.init_node('carla_helper_node', anonymous=True)

        # Publisher for setting goal pose and velocity
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher('/carla/ego_vehicle/control/set_target_velocity', Twist, queue_size=10)
        rospy.sleep(1)  # Allow publishers to register

    def set_maximal_speed(self, speed):
        import rospy
        rospy.set_param("/planning/lanelet2_global_planner/custom_speed_limit", speed)
        print("Maximal speed set to: ", speed)
    
    def set_goal_pose(self, x, y, z):
        goal_msg = PoseStamped()
        goal_msg.header.seq = 0
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"

        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z

        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        rospy.loginfo("Goal has been set successfully!")

    def publish_velocity(self, linear_x):
        velocity_msg = Twist()
        velocity_msg.linear.x = linear_x
        velocity_msg.linear.y = 0.0
        velocity_msg.linear.z = 0.0
        velocity_msg.angular.x = 0.0
        velocity_msg.angular.y = 0.0
        velocity_msg.angular.z = 0.0

        self.vel_pub.publish(velocity_msg)
        rospy.loginfo(f"Velocity {linear_x} m/s published.")


if __name__ == "__main__":
    helper = CarlaHelper()

    # Örnek koordinatlar
    #x = 92.80921173095703
    #y = -35.9668083190918
    #z = 0.0

    # x = 108.818428
    # y = -55.4220199 
    x = 102
    y = -55
    z = 0
    helper.set_goal_pose(x, y, z)

    # Komut satırından hız parametresi al
    try:
        velocity = float(sys.argv[1])
    except (IndexError, ValueError):
        velocity = 5.0  # Varsayılan hız

    helper.publish_velocity(velocity)

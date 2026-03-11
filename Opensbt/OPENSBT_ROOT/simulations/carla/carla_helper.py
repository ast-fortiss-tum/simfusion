
from geometry_msgs.msg import PoseStamped


class CarlaHelper:

    def __init__(self):
        # ROS node initialization
        pass

    def set_goal_pose(self, x, y, z):
        import os
        os.environ["ROS_MASTER_URI"] = "http://localhost:11312"

        import rospy


        rospy.init_node('carla_helper_node', anonymous=True)

        # Publisher for setting goal pose
        self.pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        goal_msg = PoseStamped()
        goal_msg.header.seq = 0
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        
        goal_msg.pose.position.x = 100.649673
        goal_msg.pose.position.y = -55.557228
 
        #goal_msg.pose.position.x = 92.80921173095703
        #goal_msg.pose.position.y = -35.9668083190918

        #goal_msg.pose.position.x = x 
        #goal_msg.pose.position.y = y
        #goal_msg.pose.position.z = z

        #goal_msg.pose.position.x = 153.992538
        #goal_msg.pose.position.y = -455.661591
        #goal_msg.pose.position.z = 0.0
        
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.pub.publish(goal_msg)
        rospy.loginfo("Goal has been set successfully!")

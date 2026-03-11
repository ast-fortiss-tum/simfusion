import rospy
from shapely.geometry import Point, Polygon
from visualization_msgs.msg import MarkerArray
from simulations.simple_sim.subscribers import Subscribers

class LaneletOperations:
    def __init__(self):
        self.lanelet_boundaries = []

    def clear(self):
        self.lanelet_boundaries = []
        
    def is_point_in_map(self, point):
        """
        Check if a point is within the map boundaries.

        Args:
            point (tuple): The point (x, y) to check.

        Returns:
            bool: True if the point is inside the map, otherwise False.
        """
        point_geometry = Point(point)
        return any(boundary.contains(point_geometry) for boundary in self.lanelet_boundaries)

    def find_closest_point(self, point):
        """
        Find the closest point on the map boundaries to the given point.

        Args:
            point (tuple): The point (x, y) to find the closest map point to.

        Returns:
            tuple: Closest point (x, y) on the map boundaries.
        """
        point_geometry = Point(point)
        closest_point = None
        min_distance = float('inf')

        for boundary in self.lanelet_boundaries:
            boundary_point = boundary.exterior.interpolate(boundary.exterior.project(point_geometry))
            distance = point_geometry.distance(boundary_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = (boundary_point.x, boundary_point.y)

        return closest_point

    def process_point(self, target_point):
        """
        Validate a point lies within the map boundaries, and find the closest valid point if outside.

        Args:
            target_point (tuple): Target point (x, y).

        Returns:
            tuple: Validated or adjusted point (x, y).
        """
        if not self.lanelet_boundaries:
            rospy.logwarn("Map boundaries not yet available. Using input point as default.")
            return target_point

        if self.is_point_in_map(target_point):
            return target_point
        else:
            closest_point = self.find_closest_point(target_point)
            return closest_point

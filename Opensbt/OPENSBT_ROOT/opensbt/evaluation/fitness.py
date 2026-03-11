import copy
import sys
from typing import Tuple
from opensbt.simulation.simulator import SimulationOutput
import numpy as np
import math
from opensbt.utils import geometric
from opensbt.config import DEFAULT_CAR_LENGTH
import numpy as np
from scipy.interpolate import interp1d
import logging as log

class Fitness():
    """This class defines an interface for concrete fitness functions. 
    """
    
    
    @property
    def min_or_max(self):
        """Defines for each objective if it is minimized or maximized. Returns a tuple, where each element holds 
           the value "min" or "max".
        """
        pass

    @property
    def name(self):     
        """Defines the name of the fitness funtion as a tuple, where each element corresponds to the name of the objective
        """
        pass
 
    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:    
        """Returns the fitness value of a given SimulationOutput instance.
    
        :param simout: SimulationOutput instance.
        :type simout: SimulationOutput 
        :param kwargs: Further optional variables needed for fitness evaluation. 
        :type kwargs: **Dict 
        :return:: Return the fitness value for the given simulation results. Each tuple element corresponds to value for the specific fitness dimension.
        :rtype Tuple[float]
        """
        pass

class MockFitness():
    @property
    def min_or_max(self):
        return "min","min"

    @property
    def name(self):
        return "dimension_1","dimension_2"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        return (0,0)
        
class FitnessMinDistance(Fitness):
    @property
    def min_or_max(self):
        return ("min",)

    @property
    def name(self):
        return "Min distance"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        if "distance" in simout.otherParams:
            dist = simout.otherParams["distance"]
            result = min(dist)
        else:
            trace_ego = simout.location["ego"]
            trace_ped = simout.location["adversary"]
            result = np.min(geometric.distPair(trace_ego, trace_ped))
        return result

class FitnessMinDistanceVelocity(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min distance", "Velocity at min distance"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        trace_ego = simout.location["ego"]
        trace_ped = simout.location[name_adversary]

        ind_min_dist = np.argmin(geometric.distPair(trace_ego, trace_ped))

        # distance between ego and other object
        distance = np.min(geometric.distPair(trace_ego, trace_ped))

        # speed of ego at time of the minimal distance
        speed = simout.speed["ego"][ind_min_dist]

        return (distance, speed)
    
def min_distance_with_index(trace1, trace2):
    """
    Compute the minimum Euclidean distance between two traces and return the index pair.

    Args:
        trace1: np.ndarray of shape (N, 3) — e.g., ego trace
        trace2: np.ndarray of shape (M, 3) — e.g., pedestrian trace

    Returns:
        min_dist: float — minimum Euclidean distance
        idx_pair: tuple — indices (i, j) such that distance(trace1[i], trace2[j]) is minimal
    """
    trace1 = np.asarray(trace1)
    trace2 = np.asarray(trace2)

    # Only consider x and y dimensions (if z is not relevant)
    pos1 = trace1[:, :2]
    pos2 = trace2[:, :2]

    # Compute pairwise distances
    dists = np.linalg.norm(pos1[:, None, :] - pos2[None, :, :], axis=2)

    # Find minimum distance and corresponding indices
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_dist = dists[min_idx]

    return min_dist, min_idx

def interpolate_trace(trace, time_src, time_target):
    """
    Interpolates a trace (x, y, z) from source times to target times.
    
    Args:
        trace: (N, 3) array
        time_src: (N,) array of timestamps
        time_target: (M,) array to which we want to interpolate
        
    Returns:
        trace_interp: (M, 3) array interpolated at time_target
    """
    trace = np.asarray(trace)
    time_src = np.asarray(time_src)
    time_target = np.asarray(time_target)
    
    f_interp = interp1d(time_src, trace, axis=0, bounds_error=False, fill_value="extrapolate")
    return f_interp(time_target)

def compute_aligned_distances(trace_ego, time_ego, trace_ped, time_ped):
    """
    Interpolates pedestrian trace to ego's time base and computes Euclidean distances.
    
    Returns:
        dists: (N,) array of distances at each time_ego point
        min_index: index of the minimum distance in time_ego
        min_dist: the minimum distance value
    """
    trace_ped_interp = interpolate_trace(trace_ped, time_ped, time_ego)
    dists = np.linalg.norm(trace_ego[:, :2] - trace_ped_interp[:, :2], axis=1)
    min_index = np.argmin(dists)
    min_dist = dists[min_index]
    return dists, min_index, min_dist
    
class FitnessMinDistanceVelocityFrontOnly(Fitness):
    def __init__(self, 
                    offset_x=0.5 * DEFAULT_CAR_LENGTH,
                    offset_y=0):
        super().__init__()
        
        self.offset_x = offset_x
        self.offset_y = offset_y

        print("In fitness min distance front only.")

    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min distance", "Velocity at min distance"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        name_adversary = "adversary" if "adversary" in simout.location else "other"

        trace_ego_raw = np.asarray(simout.location["ego"])
        trace_ped_raw = np.asarray(simout.location[name_adversary])

        # Handle missing or empty timestamps with fallback
        time_ego = (
            np.asarray(simout.timestamps["ego"])
            if hasattr(simout, 'timestamps') and simout.timestamps and "ego" in simout.timestamps and len(simout.timestamps["ego"]) > 0
            else np.asarray(simout.times) if hasattr(simout, 'times') and len(simout.times) > 0
            else np.arange(len(trace_ego_raw)) * 0.02
        )

        time_ped = (
            np.asarray(simout.timestamps[name_adversary])
            if hasattr(simout, 'timestamps') and simout.timestamps and name_adversary in simout.timestamps and len(simout.timestamps[name_adversary]) > 0
            else np.asarray(simout.times) if hasattr(simout, 'times') and len(simout.times) > 0
            else np.arange(len(trace_ped_raw)) * 0.02
        )

        print("time ego first entry: ", time_ego[0])
        print("time ped first entry: ", time_ped[0])

        # --- Define common time grid ---
        t_min = max(time_ego.min(), time_ped.min())
        t_max = min(time_ego.max(), time_ped.max())
        dt = 0.02
        time_common = np.arange(t_min, t_max, dt)

        print(f"[TIME] t_min={t_min}, t_max={t_max}, dt={dt}, N={len(time_common)}")

        # --- Interpolate trajectories ---
        trace_ego = interpolate_trace(trace_ego_raw, time_ego, time_common)
        trace_ped = interpolate_trace(trace_ped_raw, time_ped, time_common)

        print("offset_x:", self.offset_x)
        print("offset_y:", self.offset_y)

        # --- Apply x and y offsets to ego trace ---
        trace_ego[:, 0] -= self.offset_x
        trace_ego[:, 1] -= self.offset_y
        log.info(f"[OFFSET] Applied offsets: x={self.offset_x}, y={self.offset_y}")

        # --- Compute distance ---
        dists = np.linalg.norm(trace_ego[:, :2] - trace_ped[:, :2], axis=1)
        
        ind_min = np.argmin(dists)
        min_dist = dists[ind_min]
        min_time = time_common[ind_min]

        # --- Speed at minimum distance ---
        speed_ego_raw = np.asarray(simout.speed["ego"])
        speed_interp = np.interp(time_common, time_ego, speed_ego_raw)
        speed_at_min = speed_interp[ind_min]

        # --- Logging ---
        log.info(f"[MIN DIST] {min_dist:.4f}m at t={min_time:.4f}s, speed={speed_at_min:.4f}m/s")
        log.info(f"[POSITION] ego=({trace_ego[ind_min, 0]:.4f}, {trace_ego[ind_min, 1]:.4f}), "
                f"ped=({trace_ped[ind_min, 0]:.4f}, {trace_ped[ind_min, 1]:.4f})")

        print(f"[MIN DIST] index={ind_min}")
        print(f"[MIN DIST] value={min_dist}")
        print(f"[MIN DIST] time={time_common[ind_min]}")
        print(
            "[POSITION FITNESS] ego: (%s, %s), ped: (%s, %s)" % (
                trace_ego[ind_min, 0],
                trace_ego[ind_min, 1],
                trace_ped[ind_min, 0],
                trace_ped[ind_min, 1],
            )
        )

        return (min_dist, speed_at_min)
# class FitnessMinDistanceVelocityFrontOnly(Fitness):
#     def __init__(self, offset = 0.5 * DEFAULT_CAR_LENGTH):
#         super().__init__()
#         self.offset = offset

#     @property
#     def min_or_max(self):
#         return "min", "max"

#     @property
#     def name(self):
#         return "Min distance", "Velocity at min distance"

#     def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
#         if "adversary" in simout.location:
#             name_adversary = "adversary"
#         else:
#             name_adversary = "other"
                    
#         offset = self.offset

#         # 1. displace by offset x 
#         trace_ego = np.asarray(simout.location["ego"])
#         # log.info(f"BEFORE: {trace_ego}")
#         trace_ego[:, 0] = trace_ego[:, 0] - offset
#         trace_ego = copy.deepcopy(trace_ego.tolist())

#         # 2a. compute distance ego, adversary
#         time_ego = simout.timestamps["ego"] if len(simout.timestamps["ego"])> 0 else simout.times 
#         time_ped = simout.timestamps[name_adversary] if len(simout.timestamps["adversary"]) > 0 else simout.times 

#         _, ind_min_dist, distance = compute_aligned_distances(time_ego=np.asarray(time_ego),
#                                   time_ped=np.asarray(time_ped),
#                                   trace_ped=np.asarray(simout.location[name_adversary]),
#                                   trace_ego=np.asarray(trace_ego)
#                                   )
#         trace_ped = simout.location[name_adversary]
#         # # distance, ind_min_dist = min_distance_with_index(trace_ego, trace_ped)
#         # # ind_min_dist = ind_min_dist[0]
#         log.info(f"IND_INTERPOL: {ind_min_dist}")
#         log.info(f"DISTANCE_INTERPOL: {distance}")

#         # 2b. compute distance ego, adversary

#         ##################
#         ind_min_dist_my = np.argmin(geometric.distPair(trace_ego, trace_ped))
#         # approx distance between ego's front and other object
#         distance_my = np.min(geometric.distPair(trace_ego, trace_ped))
#         log.info(f"IND_MY: {ind_min_dist_my}")
#         log.info(f"DISTANCE_MY:  {distance_my}")
#         ###################

#         # ind_min_dist = ind_min_dist_my
#         # distance = distance_my

#         # speed of ego at time of the minimal distance
#         speed = simout.speed["ego"][ind_min_dist]
#         # value scenarios worse if pedestrian is not in front of the car
#         FITNESS_WORSE = 1000
        
#         log.info("[POSITION X FITNESS] ego: %s, ped: %s", trace_ego[ind_min_dist][0], trace_ped[ind_min_dist][0])
#         log.info("[POSITION Y FITNESS] ego: %s, ped: %s", trace_ego[ind_min_dist][1], trace_ped[ind_min_dist][1])
        
#         # if (trace_ego[ind_min_dist][0] -  trace_ped[ind_min_dist][0] < 0 and 
#         #     trace_ego[ind_min_dist][1] >  trace_ped[ind_min_dist][1]):
#         #     distance = FITNESS_WORSE # favor closer results
#         #     speed = -FITNESS_WORSE 
#         log.info(f"SPEED: {speed}")
#         return (distance, speed)

class FitnessMinTTC(Fitness):
    @property
    def min_or_max(self):
        return "min",

    @property
    def name(self):
        return "Min TTC",

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        all_ttc = []
        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        for i in range(2, len(simout.times)):
            ego_location = simout.location["ego"]
            adv_location = simout.location[name_adversary]

            colpoint = geometric.intersection(
                (ego_location[i], ego_location[i-1]), (adv_location[i], adv_location[i-1]))

            if colpoint == []:
                all_ttc.append(sys.maxsize)
            else:
                velocity_ego = simout.speed["ego"]
                velocity_adv = simout.speed[name_adversary]
                dist_ego_colpoint = geometric.dist(colpoint, ego_location[i])
                dist_adv_colpoint = geometric.dist(colpoint, adv_location[i])

                if colpoint == []:
                    all_ttc.append(sys.maxsize)
                    continue
                if velocity_ego[i] == 0 or velocity_adv[i] == 0:
                    all_ttc.append(sys.maxsize)
                    continue

                t_col_ego = dist_ego_colpoint/velocity_ego[i]
                t_col_ped = dist_adv_colpoint/velocity_adv[i]
                t_tolerance = 0.5  # time tolerance for missed collision
                if abs(t_col_ego - t_col_ped) < t_tolerance:
                    all_ttc.append(t_col_ego)
                else:
                    all_ttc.append(t_col_ego)

        min_ttc = np.min(all_ttc)
        return min_ttc

class FitnessMinTTCVelocity(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min TTC", "Critical Velocity"

    def eval(self, simout: SimulationOutput, **kwargs) -> float:
        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        for i in range(2, len(simout.times)):
            ego_location = simout.location["ego"]
            adv_location = simout.location[name_adversary]

            colpoint = geometric.intersection(
                (ego_location[i], ego_location[i-1]), (adv_location[i], adv_location[i-1]))
            all_ttc = []
            # If no collision, return huge value
            if colpoint == []:
                min_ttc = sys.maxsize
                velocity_min_ttc = sys.maxsize
            else:
                velocity_ego = simout.speed["ego"]
                velocity_adv = simout.speed[name_adversary]

                dist_ego_colpoint = geometric.dist(colpoint, ego_location[i])
                dist_adv_colpoint = geometric.dist(colpoint, adv_location[i])

                if colpoint == []:
                    all_ttc.append(sys.maxsize)
                    continue
                if velocity_ego[i] == 0 or velocity_adv[i] == 0:
                    all_ttc.append(sys.maxsize)
                    continue

                t_col_ego = dist_ego_colpoint/velocity_ego[i]
                t_col_ped = dist_adv_colpoint/velocity_adv[i]

                t_tolerance = 0.5  # time tolerance for missed collision
                if abs(t_col_ego - t_col_ped) < t_tolerance:
                    all_ttc.append(t_col_ego)
                else:
                    all_ttc.append(t_col_ego)

                time_min_ttc = np.argmin(all_ttc)
                min_ttc = all_ttc[time_min_ttc]
                velocity_min_ttc = velocity_ego[time_min_ttc]
        result = (min_ttc, velocity_min_ttc)
        return result

class FitnessAdaptedDistSpeedRelVelocity(Fitness):
    @property
    def min_or_max(self):
        return "max", "max", "max"

    @property
    def name(self):
        return "Critical Adapted distance", "Velocity at critical distance", "Relative velocity at critical distance"

    ''' Fitness function to resolve front and rear collisions'''

    def fitness_parallel(self, z_parallel, car_length):
        """
        Input:
            z_parallel, which is a projection of relative position of a car's front bumper and a pedestrian to the
            axis, parallel to a car velocity.
            car_length, which is a length of a car.
        Returns:
            a value between 0 and 1, indicating severeness of the relative position of a car and a pedestrian in the
            parallel direction. The higher the value - the more severe a scenario is.

        The fitness function is composed of exponential functions and constant functions. step_back characterizes
        steepness of decay of the fitness function behind the front bumper. step_front characterizes steepness of decay
        of the fitness function in front of the front bumper. The value of the fitness function for positions of a
        pedestrian behind the back bumper is 0.
        """
        steep_back = 10
        steep_front = 2
        # z = yPed - yEgo - car_length/2
        n = len(z_parallel)
        result = np.zeros(n)
        for i in range(n):
            if z_parallel[i] < -car_length:
                result[i] = 0
            elif z_parallel[i] < 0:
                result[i] = (np.exp(steep_back * (z_parallel[i] + car_length)) - np.exp(0)) / np.exp(
                    steep_back * car_length)
            else:
                result[i] = np.exp(-steep_front * z_parallel[i]) / np.exp(0)
        return result

    def fitness_perpendicular(self, z_perpendicular, car_width):
        """
        Input:
            z_perpendicular, which is a projection of relative position of a car's center and a pedestrian to the axis,
            perpendicular to a car velocity.
            car_width, which is a width of a car.

        Returns:
            a value between 0 and 1, indicating severeness of the relative position of a car and a pedestrian in the
            perpendicular direction. The higher the value - the more severe a scenario is.

        The fitness function is composed of a "bell-shaped" function segments on the sides, and a constant function in
        the middle equal to 1. A "bell-shaped" function is a gaussian function. sigma is proportional to the width of a
        "bell". A constant segment has a length of the car_width. The parallel fitness function, combined with the
        perpendicular fitness function, e.g. by multiplication, results in a proximity function, which defines a
        severeness of the relative position of a car and a pedestrian.
        """
        sigma = 0.05
        # z = xPed - xEgo
        n = len(z_perpendicular)
        result = np.zeros(n)
        for i in range(n):
            if abs(z_perpendicular[i]) < car_width / 2:
                result[i] = 1
            else:
                result[i] = np.exp(-(0.5 / sigma) *
                                   (abs(z_perpendicular[i]) - car_width / 2) ** 2)
        return result

    def eval(self, simout: SimulationOutput, **kwargs) -> float:
        if "car_length" in simout.otherParams:
            car_length = float(simout.otherParams["car_length"])
        else:
            car_length = float(4.3)

        if "car_width" in simout.otherParams:
            car_width = float(simout.otherParams["car_width"])
        else:
            car_width = float(1.8)

        if "adversary" in simout.location:
            name_adversary = "adversary"
        else:
            name_adversary = "other"

        # time series of Ego position
        trace_ego = np.array(simout.location["ego"])
        # time series of Ped position
        trace_adv = np.array(simout.location[name_adversary])

        # time series of Ego position
        velocity_ego = np.array(simout.velocity["ego"])
        # time series of Ped position
        velocity_adv = np.array(simout.velocity[name_adversary])
        velocity_relative = velocity_adv - velocity_ego

        # time series of Ego velocity
        speed_ego = np.array(simout.speed["ego"])
        yaw_ego = np.array(simout.yaw["ego"])  # time series of Ego velocity

        # Global coordinates 
        x_ego = trace_ego[:, 0]
        y_ego = trace_ego[:, 1]
        x_adv = trace_adv[:, 0]
        y_adv = trace_adv[:, 1]

        # Coordinates, with respect to ego: e2 is parallel to the direction of ego
        e2_x = np.cos(yaw_ego * math.pi / 180)
        e2_y = np.sin(yaw_ego * math.pi / 180)
        e1_x = e2_y
        e1_y = -e2_x

        z_parallel = (x_adv - x_ego) * e2_x + \
            (y_adv - y_ego) * e2_y - car_length / 2
        z_perpendicular = (x_adv - x_ego) * e1_x + (y_adv - y_ego) * e1_y

        f_1 = self.fitness_parallel(z_parallel, car_length)
        f_2 = self.fitness_perpendicular(z_perpendicular, car_width)
        critical_iteration = np.argmax(f_1 * f_2)

        vector_fitness = (f_1[critical_iteration] * f_2[critical_iteration],
                          speed_ego[critical_iteration],
                          np.linalg.norm(velocity_relative[critical_iteration]))

        # Maybe speed and relative velocity should not be taken at the critical iteration? But at some steps before that.
        # For some reason, when the front collision happens,
        # F1 is not equal to 1 for carla simulation, but somewhere in between 0.5 and 1.0.
        # It happens for f_1. It happens because of poor sampling at the moment of collision.
        # The parallel fitness function should be modified to account for that, or carla settings.
        return vector_fitness


class FitnessAdaptedDistanceSpeed(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Critical adapted distance", "Velocity at critical distance"

    def eval(self, simout: SimulationOutput, **kwargs) -> float:
        # use only adapted distance and velocity of the fitness comupation of existing function
        vector_fitness_all = FitnessAdaptedDistSpeedRelVelocity().eval(simout)
        adapted_distance = vector_fitness_all[0]
        speed = vector_fitness_all[1]
        return adapted_distance, speed


class FitnessAdaptedDistanceSpeedTTC(Fitness):
    @property
    def min_or_max(self):
        return "min", "max", "min"

    @property
    def name(self):
        return "Critical adapted distance", "Velocity at critical distance", "Min TTC"

    def eval(self, simout: SimulationOutput, **kwargs) -> float:
        min_ttc = FitnessMinTTC().eval(simout)
        pos_crit = FitnessAdaptedDistanceSpeed().eval(simout)
        return pos_crit[0], pos_crit[1], min_ttc


import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

log = logging.getLogger(__name__)


# ============================================================
# 1) Geometry helpers for OBB (Oriented Bounding Box) in 2D
# ============================================================

def _rotmat(theta: float) -> np.ndarray:
    """
    Build a 2x2 rotation matrix for angle theta [rad].
    Used to rotate local rectangle corners into world coordinates.
    """
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def _obb_corners(center_xy: np.ndarray, yaw: float, L: float, W: float) -> np.ndarray:
    """
    Compute the 4 corners of a vehicle rectangle in world coordinates.

    Inputs:
      center_xy: (2,) -> vehicle center position (x,y)
      yaw: heading angle [rad]
      L: vehicle length [m]
      W: vehicle width  [m]

    Output:
      corners: (4,2) array of world points

    How it is computed:
      - Define rectangle corners in vehicle-local frame:
          (+L/2,+W/2), (+L/2,-W/2), (-L/2,-W/2), (-L/2,+W/2)
      - Rotate by yaw, then translate by center.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    halfL, halfW = 0.5 * L, 0.5 * W

    local = np.array([
        [ halfL,  halfW],
        [ halfL, -halfW],
        [-halfL, -halfW],
        [-halfL,  halfW],
    ], dtype=float)

    R = _rotmat(yaw)
    return local @ R.T + np.array([cx, cy], dtype=float)


def _project_polygon(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """
    Project polygon vertices onto an axis and return [min,max] interval.
    Used by SAT (Separating Axis Theorem).
    """
    vals = poly @ axis
    return float(np.min(vals)), float(np.max(vals))


def _sat_overlap(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    Separating Axis Theorem overlap test (collision test) for two convex polygons.
    Here: both are rectangles (4 corners).

    Idea:
      Two convex polygons do NOT overlap if there exists at least one axis where
      their projected intervals do not overlap (a "separating axis").

    For rectangles, it is sufficient to test normals of both rectangles' edges
    (4 axes total: 2 from rect1, 2 from rect2).
    """
    def axes_from(poly: np.ndarray):
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            e = p1 - p0
            axis = np.array([-e[1], e[0]], dtype=float)
            if np.linalg.norm(axis) > 1e-12:
                yield axis

    for axis in list(axes_from(poly1)) + list(axes_from(poly2)):
        min1, max1 = _project_polygon(poly1, axis)
        min2, max2 = _project_polygon(poly2, axis)
        if max1 < min2 or max2 < min1:
            return False  # found a separating axis => no overlap
    return True  # no separating axis => overlap (collision)


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance from point p to segment a-b (Euclidean).
    Used to compute min distance between two non-overlapping rectangles:
      - check corner-to-edge distances both ways.
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))  # clamp projection to segment
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def obb_min_distance_and_collision(
    center1: np.ndarray, yaw1: float,
    center2: np.ndarray, yaw2: float,
    L: float, W: float,
) -> Tuple[float, bool]:
    """
    Compute OBB-OBB minimum distance (2D) and collision flag.

    Steps:
      1) Build rectangle corners for both vehicles using yaw.
      2) SAT overlap test:
         - if overlap => collision => min distance = 0
      3) If no overlap:
         - compute min of:
             corners(rect1) -> edges(rect2)
             corners(rect2) -> edges(rect1)
         - that gives true minimum distance between the two rectangles.

    Returns:
      (min_dist, collision)
    """
    c1 = _obb_corners(center1, yaw1, L, W)
    c2 = _obb_corners(center2, yaw2, L, W)

    coll = _sat_overlap(c1, c2)
    if coll:
        return 0.0, True

    best = float("inf")
    for p in c1:
        for j in range(4):
            a = c2[j]
            b = c2[(j + 1) % 4]
            best = min(best, _point_segment_dist(p, a, b))
    for p in c2:
        for j in range(4):
            a = c1[j]
            b = c1[(j + 1) % 4]
            best = min(best, _point_segment_dist(p, a, b))

    return float(best), False

# ============================================================
# 1b) Goal point vs ego OBB helpers
# ============================================================

def _point_in_obb(goal_xy: np.ndarray, center_xy: np.ndarray, yaw: float, L: float, W: float, tol: float = 0.0) -> bool:
    """
    Check if a world point is inside the vehicle OBB.
    Uses world->local transform (rotate by -yaw), then inside AABB check.
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])

    dx, dy = gx - cx, gy - cy
    c, s = float(np.cos(yaw)), float(np.sin(yaw))

    # world -> local (R(-yaw))
    local_x =  c * dx + s * dy
    local_y = -s * dx + c * dy

    hx, hy = 0.5 * L, 0.5 * W
    return (abs(local_x) <= (hx + tol)) and (abs(local_y) <= (hy + tol))


def _point_to_obb_distance(goal_xy: np.ndarray, center_xy: np.ndarray, yaw: float, L: float, W: float) -> float:
    """
    Minimum distance from a world point to the vehicle OBB.
    If inside => 0. Otherwise distance to the rectangle boundary (in local frame).
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])

    dx, dy = gx - cx, gy - cy
    c, s = float(np.cos(yaw)), float(np.sin(yaw))

    local_x =  c * dx + s * dy
    local_y = -s * dx + c * dy

    hx, hy = 0.5 * L, 0.5 * W

    ox = max(abs(local_x) - hx, 0.0)
    oy = max(abs(local_y) - hy, 0.0)
    return float(np.hypot(ox, oy))


# ============================================================
# 2) Collision record structure (optional, for simout.collisions)
# ============================================================

@dataclass
class CollisionEvent:
    """
    Lightweight record for the first time we detect collision
    between ego and an adversary.
    """
    step_idx: int
    time: Any
    ego_key: str
    adv_key: str
    ego_xy: Tuple[float, float]
    adv_xy: Tuple[float, float]
    ego_yaw: float
    adv_yaw: float
    min_dist: float  # should be 0.0 when colliding


# ============================================================
# 3) Fitness class using OBB distance + collision logging
#    + Goal point vs Ego OBB distance
# ============================================================

class FitnessObstacleDistanceGoalDistance(Fitness):
    """
    Fitness returns two objectives (both minimized):
      1) Min OBB distance between ego and any adversary over time (0 => collision)
      2) Min ego-center distance to goal center over time

    Additionally (optional):
      - stores collision events into simout.collisions as CollisionEvent objects

    Requires:
      simout.location[key] -> list/array of (x,y,...) per timestep
      simout.yaw[key]      -> list/array of yaw [rad] per timestep
      simout.times         -> list/array of time steps
      simout.otherParams   -> includes goal_center_x, goal_center_y
    """

    VEHICLE_LENGTH_M = 4.7
    VEHICLE_WIDTH_M = 1.8

    GOAL_INSIDE_TOL_M = 0.00  # tolerance for goal inside ego OBB

    @property
    def min_or_max(self):
        return "min", "min"

    @property
    def name(self):
        return "Min distance to obstacle", "Distance to goal"

    def eval(self, simout: "SimulationOutput", **kwargs):
        log.info("[FITNESS] ===== start =====")

        # ---------- Goal ----------
        gx = float(simout.otherParams.get("goal_center_x", 0.0))
        gy = float(simout.otherParams.get("goal_center_y", 0.0))
        goal_pos = np.array([gx, gy], dtype=float)
        log.info("[FITNESS] goal_pos=(%.3f, %.3f)", gx, gy)

        # ---------- Ego trajectories ----------
        ego_raw = np.asarray(simout.location.get("ego", []), dtype=float)
        ego_xy = ego_raw[:, :2]

        yaw_dict = getattr(simout, "yaw", None)
        if yaw_dict is None:
            raise ValueError("[FITNESS] simout.yaw is missing (None)")
        ego_yaw_full = np.asarray(yaw_dict.get("ego", []), dtype=float)
        if ego_yaw_full.size == 0:
            raise ValueError("[FITNESS] ego yaw is empty: simout.yaw['ego'] missing/empty")

        if ego_xy.size == 0:
            log.warning("[FITNESS] ego trajectory empty -> return defaults")

        log.info("[FITNESS] ego_xy_len=%d ego_yaw_len=%d", len(ego_xy), len(ego_yaw_full))

        # ---------- Goal distance: goal point vs ego OBB (NEW) ----------
        L = float(self.VEHICLE_LENGTH_M)
        W = float(self.VEHICLE_WIDTH_M)

        n_goal = min(len(ego_xy), len(ego_yaw_full))
        ego_yaw_goal = np.nan_to_num(ego_yaw_full[:n_goal], nan=0.0, posinf=0.0, neginf=0.0)

        goal_dists = np.empty((n_goal,), dtype=float)
        reached_idx = -1
        for i in range(n_goal):
            goal_dists[i] = _point_to_obb_distance(goal_pos, ego_xy[i], float(ego_yaw_goal[i]), L, W)
            if reached_idx < 0 and _point_in_obb(goal_pos, ego_xy[i], float(ego_yaw_goal[i]), L, W, tol=float(self.GOAL_INSIDE_TOL_M)):
                reached_idx = i

        if reached_idx >= 0:
            evaluation_end_idx = reached_idx
            log.info(f"[FITNESS] Goal reached at step {reached_idx}. Ignoring subsequent events.")
        else:
            evaluation_end_idx = n_goal - 1
            log.info("[FITNESS] Goal NOT reached. Evaluating entire simulation.")

        min_goal_distance = float(np.nanmin(goal_dists)) if goal_dists.size else None
        goal_min_idx = int(np.nanargmin(goal_dists)) if goal_dists.size else -1

        log.info(
            "[FITNESS] goal_min_dist(OBB)=%.6f at idx=%d ego_at=(%.3f, %.3f) ego_yaw=%.6f reached_idx=%d",
            min_goal_distance,
            goal_min_idx,
            ego_xy[goal_min_idx][0] if goal_min_idx >= 0 else float("nan"),
            ego_xy[goal_min_idx][1] if goal_min_idx >= 0 else float("nan"),
            float(ego_yaw_goal[goal_min_idx]) if goal_min_idx >= 0 else float("nan"),
            reached_idx,
        )

        # store optional debug into otherParams (safe)
        try:
            simout.otherParams["goal_reached"] = (reached_idx >= 0)
            simout.otherParams["goal_reached_idx"] = reached_idx
            simout.otherParams["goal_min_dist_obb"] = min_goal_distance
            simout.otherParams["goal_min_dist_obb_idx"] = goal_min_idx
        except Exception:
            pass

        # ---------- OBB distance to adversaries ----------
        if not hasattr(simout, "collisions") or simout.collisions is None:
            simout.collisions = []

        collided_adv = set()
        per_adv_best = []
        adv_seen = 0

        for key, val in simout.location.items():
            if key == "ego" or not (key == "adversary" or key.startswith("adversary_")):
                continue

            adv_seen += 1
            obs_raw = np.asarray(val, dtype=float)
            if obs_raw.size == 0:
                log.info("[FITNESS] ADV=%s skipped (empty location)", key)
                continue
            if obs_raw.ndim != 2 or obs_raw.shape[1] < 2:
                log.info("[FITNESS] ADV=%s skipped (bad shape %s)", key, str(obs_raw.shape))
                continue
            obs_xy = obs_raw[:, :2]

            obs_yaw_full = np.asarray(yaw_dict.get(key, []), dtype=float)
            if obs_yaw_full.size == 0:
                log.info("[FITNESS] ADV=%s skipped (empty yaw)", key)
                continue

            limit_idx = min(len(ego_xy), len(obs_xy), len(ego_yaw_full), len(obs_yaw_full))
            n_eval = min(limit_idx, evaluation_end_idx + 1)
            if n_eval <= 0:
                log.info("[FITNESS] ADV=%s skipped (n=%d)", key, n_eval)
                continue

            ego_yaw = np.nan_to_num(ego_yaw_full[:n_eval], nan=0.0, posinf=0.0, neginf=0.0)
            obs_yaw = np.nan_to_num(obs_yaw_full[:n_eval], nan=0.0, posinf=0.0, neginf=0.0)

            best_d = float("inf")
            best_i = -1
            best_coll = False

            for i in range(n_eval):
                d_i, coll_i = obb_min_distance_and_collision(
                    ego_xy[i], float(ego_yaw[i]),
                    obs_xy[i], float(obs_yaw[i]),
                    L, W
                )
                if not np.isfinite(d_i):
                    continue

                if coll_i and key not in collided_adv:
                    collided_adv.add(key)
                    t_val = simout.times[i] if hasattr(simout, "times") and len(simout.times) > i else i
                    simout.collisions.append(
                        CollisionEvent(
                            step_idx=i,
                            time=t_val,
                            ego_key="ego",
                            adv_key=key,
                            ego_xy=(float(ego_xy[i][0]), float(ego_xy[i][1])),
                            adv_xy=(float(obs_xy[i][0]), float(obs_xy[i][1])),
                            ego_yaw=float(ego_yaw[i]),
                            adv_yaw=float(obs_yaw[i]),
                            min_dist=float(d_i),
                        )
                    )
                    log.info("[FITNESS] COLLISION recorded: adv=%s idx=%d time=%s", key, i, str(t_val))

                if d_i < best_d:
                    best_d = float(d_i)
                    best_i = i
                    best_coll = bool(coll_i)
                    if best_d <= 0.0:
                        break

            if best_i < 0:
                log.info("[FITNESS] ADV=%s skipped (no finite distance)", key)
                continue

            log.info(
                "[FITNESS] ADV=%s best_min_dist=%.6f coll=%s at idx=%d "
                "ego=(%.3f, %.3f) obs=(%.3f, %.3f) ego_yaw=%.6f obs_yaw=%.6f",
                key, best_d, best_coll, best_i,
                ego_xy[best_i][0], ego_xy[best_i][1],
                obs_xy[best_i][0], obs_xy[best_i][1],
                float(ego_yaw[best_i]), float(obs_yaw[best_i]),
            )

            per_adv_best.append({"key": key, "dist": best_d, "idx": best_i, "coll": best_coll})

        log.info(
            "[FITNESS] adversaries_seen=%d valid=%d collisions_logged=%d",
            adv_seen, len(per_adv_best), len(simout.collisions)
        )

        if per_adv_best:
            best = min(per_adv_best, key=lambda d: d["dist"])
            min_obstacle_dist = float(best["dist"])
            best_idx = int(best["idx"])
            log.info(
                "[FITNESS] global_min_obstacle_dist=%.6f from %s at idx=%d",
                min_obstacle_dist, best["key"], int(best["idx"])
            )
            speeds = simout.speed["ego"]
            ego_speed_at_min_dist = float(speeds[best_idx])
            simout.otherParams["ego_speed_at_min_obstacle_dist"] = ego_speed_at_min_dist
            log.info(f"[FITNESS] Ego speed at min dist: {ego_speed_at_min_dist:.2f} m/s")
        else:
            log.info("[FITNESS] No adversaries within evaluation horizon.")

        log.info("[FITNESS] ===== end =====")
        return (min_obstacle_dist, min_goal_distance)
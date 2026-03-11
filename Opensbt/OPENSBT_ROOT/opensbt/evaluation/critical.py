from opensbt.simulation.simulator import SimulationOutput
import numpy as np
from typing import List

class Critical():
    """This class defines an interface for concrete oracle functions. 
       The functions name and eval have to be implemented.
    """
    @property
    def name(self):
        """The name of the criticality function. To be overriden for specific name. Otherwise class name is used.

        :return: Name of the criticality function.
        :rtype: str
        """
        return self.__class__.__name__

    def eval(self, vector_fitness: np.ndarray, simout: SimulationOutput) -> bool:
        """Evaluate whether a test has failed or passed. To be implemented based on specific conditions.
        
        :param vector_fitness: List of fitness values.
        :type vector_fitness: np.ndarray
        :param simout: SimulationOutput instance
        :type simout: SimulationOutput
        :return: Returns True if simulation is critical/failed or otherwise False.
        :rtype: bool
        """
        pass

class MockCritical():
    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, vector_fitness: np.ndarray, simout: SimulationOutput) -> bool:
        return True

class CriticalAdasExplicitRearCollision(Critical):

    ''' ADAS problems '''
    def eval(self, vector_fitness: List[float], simout: SimulationOutput = None):
        safety_distance = 0.50 # all dimensions, radial, in m

        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None
        # a) collision occurred (not from the rear) and velocity of ego > 0
        # b) safety distance to pedestrian violated
        if isCollision:
            loc_ego = simout.location["ego"]
            loc_ped = simout.location["other"]
            diff = [np.subtract(pos_e,pos_p) for pos_e, pos_p in zip(loc_ego, loc_ped)]
            distance = np.linalg.norm(diff)
            ind_col= np.argmin(distance)
            dist_x = abs(simout.location["ego"][ind_col][0] - simout.location["other"][ind_col][0])
            raw_dist_y = simout.location["ego"][ind_col][1] - simout.location["other"][ind_col][1]
            if dist_x < simout.otherParams["height"]/2 and raw_dist_y > 0 and vector_fitness[1] < 0:
                return True
            return False
        elif (vector_fitness[0]  - safety_distance <= 0 and vector_fitness[1] < 0):
            return True
        else:
            return False

class CriticalAdasFrontCollisions(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None

        if (isCollision == True) or (vector_fitness[0] < 0.5) and (vector_fitness[1] < 0):
            return True
        else:
            return False

class CriticalAdasTTCVelocity(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None

        if(isCollision == True) or (vector_fitness[0] < 5) and (vector_fitness[1] < -1):
            return True
        else:
            return False
        
class CriticalAdasTTC(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        if (vector_fitness[0] < 0.01):
            return True
        else:
            return False


    '''
        f[0] - min distance ego <-> pedestrian
        f[1] - velocity at time of minimal distance

        # Scenario critical <->

        # a) collision ocurred
        # b) minimal distance between ego and other vehicle < 0.3m
        # c) velcoty at time of minimal distance is > 1 m/s
    '''
class CriticalAdasDistanceVelocity(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None

        if(isCollision == True) or (vector_fitness[0] < 0.3) and (vector_fitness[1] < -1):
            return True
        else:
            return False
        
''' Test problems '''
class CriticalBnhDivided(Critical):
    def eval(self, vector_fitness: np.ndarray, simout=None):
        return  (vector_fitness[0] < 10 ) and \
                (vector_fitness[1] < 50) and  (vector_fitness[1] > 20) or \
                (vector_fitness[0] < 140 ) and (vector_fitness[0] > 40 )  and \
                (vector_fitness[1] < 7) and  (vector_fitness[1] > 0)  or \
                (vector_fitness[0] < 40 ) and (vector_fitness[0] > 20 )  and \
                (vector_fitness[1] < 20) and  (vector_fitness[1] > 15)

class CriticalBnh(Critical):
    def eval(self, vector_fitness):
        return (vector_fitness[0] < 60) and (vector_fitness[1] < 20)

class CriticalRastrigin(Critical):
    def eval(self, fitness):
        return fitness < 2 and fitness > -2

class CriticalObstacleGoalDistance(Critical):
    """
    Critical function that evaluates if scenario is critical based on:
    1. Distance to obstacle (too close = critical)
    2. Distance to goal (too far = critical, meaning goal not reached)
    """
    
    def __init__(self, min_obstacle_distance=0.5, max_goal_distance=2.0):
        """
        Args:
            min_obstacle_distance: Minimum safe distance to obstacle (meters)
            max_goal_distance: Maximum acceptable distance to goal (meters)
        """
        self.min_obstacle_distance = min_obstacle_distance
        self.max_goal_distance = max_goal_distance
        self.stop_speed_threshold = 0.0
    
    @property
    def name(self):
        return "CriticalObstacleGoalDistance"
    
    def eval(self, vector_fitness, simout: SimulationOutput = None) -> bool:
        """
        Evaluates criticality based on obstacle and goal distances.
        
        Args:
            vector_fitness: [min_obstacle_distance, goal_distance]
            simout: SimulationOutput instance (optional)
            
        Returns:
            True if scenario is critical, False otherwise
        """
        min_obstacle_dist = vector_fitness[0]
        goal_distance = vector_fitness[1]
        ego_speed = float("nan")
        if simout is not None and getattr(simout, "otherParams", None) is not None:
            try:
                ego_speed = float(simout.otherParams.get("ego_speed_at_min_obstacle_dist", float("nan")))
            except Exception:
                ego_speed = float("nan")
        
        # Log critical evaluation for debugging
        import logging as log
        log.info(f"[CRITICAL_DEBUG] ===== Starting Critical Evaluation =====")
        log.info(f"[CRITICAL_DEBUG] Input fitness vector: {vector_fitness}")
        log.info(f"[CRITICAL_DEBUG] Min obstacle distance: {min_obstacle_dist:.2f}m")
        log.info(f"[CRITICAL_DEBUG] Goal distance: {goal_distance:.2f}m")
        log.info(f"[CRITICAL_DEBUG] Thresholds - Min obstacle: {self.min_obstacle_distance}m, Max goal: {self.max_goal_distance}m")

        # Too close to obstacle
        if min_obstacle_dist < self.min_obstacle_distance:
            safety_margin = self.min_obstacle_distance - min_obstacle_dist
            if min_obstacle_dist <= 0.00:
                log.info(f"[CRITICAL_DEBUG] COLLISION DETECTED: Obstacle distance ({min_obstacle_dist:.2f}m) <= 0.0m")
                return True
            if abs(ego_speed) == self.stop_speed_threshold:
                log.info(f"SAFE: Safety Stop detected. Dist {min_obstacle_dist:.2f} < Threshold, but Speed {ego_speed:.2f} is 0.")
                return False
            log.info(f"[CRITICAL_DEBUG] Safety margin violated by: {safety_margin:.2f}m")
            log.info(f"[CRITICAL_DEBUG] RESULT: CRITICAL (Obstacle Distance)")
            return True
        else:
            safety_margin = min_obstacle_dist - self.min_obstacle_distance
            log.info(f"[CRITICAL_DEBUG] SAFETY OK: Obstacle distance ({min_obstacle_dist:.2f}m) >= threshold ({self.min_obstacle_distance}m)")
            log.info(f"[CRITICAL_DEBUG] Safety margin: +{safety_margin:.2f}m")
            
        # Failed to reach goal    
        if goal_distance > self.max_goal_distance:
            goal_margin = goal_distance - self.max_goal_distance
            log.info(f"[CRITICAL_DEBUG] MISSION FAILURE: Goal distance ({goal_distance:.2f}m) > threshold ({self.max_goal_distance}m)")
            log.info(f"[CRITICAL_DEBUG] Goal threshold exceeded by: {goal_margin:.2f}m")
            log.info(f"[CRITICAL_DEBUG] RESULT: CRITICAL (Goal Distance)")
            return True
        else:
            goal_margin = self.max_goal_distance - goal_distance
            log.info(f"[CRITICAL_DEBUG] MISSION OK: Goal distance ({goal_distance:.2f}m) <= threshold ({self.max_goal_distance}m)")
            log.info(f"[CRITICAL_DEBUG] Goal margin: +{goal_margin:.2f}m")
        
        log.info(f"[CRITICAL_DEBUG] All criteria passed - RESULT: SAFE")
        log.info(f"[CRITICAL_DEBUG] ===== Critical Evaluation Complete =====")
        return False

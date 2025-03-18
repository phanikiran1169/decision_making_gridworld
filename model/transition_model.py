import logging
import pomdp_py
import copy
import heapq
import random
from domain.action import *
from domain.state import MosOOState, RobotState, ObjectState
from domain.observation import ObjectObservation, MosOOObservation

class MosTransitionModel(pomdp_py.OOTransitionModel):
    """Object-oriented transition model; The transition model supports the
    multi-robot case, where each robot is equipped with a sensor; The
    multi-robot transition model should be used by the Environment, but
    not necessarily by each robot for planning.
    """

    def __init__(self, dim, sensors, object_ids, epsilon=1e-9):
        """

        """
        self._sensors = sensors
        transition_models = {
            objid: ObjectTransitionModel(objid, dim, epsilon=epsilon)
            for objid in object_ids
            if objid not in sensors
        }
        for robot_id in sensors:
            transition_models[robot_id] = RobotTransitionModel(
                sensors[robot_id], dim, epsilon=epsilon
            )
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        logging.debug(f"MosTransitionModel - sample")
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        logging.debug(f"MosTransitionModel - argmax")
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)


class ObjectTransitionModel(pomdp_py.TransitionModel):
    """
    """

    def __init__(self, objid, dim, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon
        self._dim = dim

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state["id"]]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def sample(self, state, action):
        """Returns next_object_state"""
        logging.debug(f"ObjectTransitionModel - sample")
        return self.argmax(state, action)

    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        
        logging.debug(f"ObjectTransitionModel - argmax")
        logging.debug(f"Object Class - {state.object_states[self._objid].objclass}")
        logging.debug(f"Object ID - {self._objid}")

        if state.object_states[self._objid].objclass == "avoid":
            return self.move(state, action)
        else:
            return copy.deepcopy(state.object_states[self._objid])
    
    def move(self, state, action):
        # logging.debug("ObjectTransitionModel - Move")
        # next_obj_state = copy.deepcopy(state.object_states[self._objid])
        
        # if state.object_states[self._objid].objclass != "avoid":
        #     return next_obj_state
        
        # next_obj_state["pose"] = (4, 1)   
        # return next_obj_state
    
        logging.debug("ObjectTransitionModel - Move")
        next_obj_state = copy.deepcopy(state.object_states[self._objid])
        
        if state.object_states[self._objid].objclass != "avoid":
            return next_obj_state
        
        # Current position of the avoid object
        current_pos = state.object_states[self._objid]["pose"]
        
        # Extract the robot's position from the state (assuming there is only one robot)
        robot_pos = None
        for objid, obj_state in state.object_states.items():
            if isinstance(obj_state, RobotState):
                robot_pos = obj_state.pose
                break  # Assuming only one robot
        
        if robot_pos is None:
            raise ValueError("Robot position not found in the state.")
        
        # Obstacles are fixed and obtained from the environment (all objects except the robot and avoid objects)
        obstacles = {
            state.object_states[objid]["pose"]
            for objid, obj_state in state.object_states.items()
            if isinstance(obj_state, ObjectState) and obj_state.objclass in ["obstacle"]
        }
        
        # Apply A* planning
        path = a_star_search(current_pos, robot_pos, self._dim[0], self._dim[1], obstacles)
        
        if path:
            # If a path is found, move to the next position in the path with some randomness
            if random.random() < 0.60:
                next_obj_state["pose"] = current_pos
            else:
                next_obj_state["pose"] = path[0]
        else:
            # No path found, stay in the current position
            next_obj_state["pose"] = current_pos
        
        return next_obj_state

class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self, sensor, dim, epsilon=1e-9):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self._sensor = sensor
        self._robot_id = sensor.robot_id
        self._dim = dim
        self._epsilon = epsilon

    @classmethod
    def if_move_by(cls, robot_id, state, action, dim, check_collision=True):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        if not isinstance(action, MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        robot_pose = state.pose(robot_id)
        x, y = robot_pose
        dx, dy = action.motion
        x += dx
        y += dy

        if cls.valid_pose(
            (x, y),
            dim[0],
            dim[1],
            state=state,
            check_collision=check_collision,
            pose_objid=robot_id,
        ):
            return (x, y)
        else:
            return robot_pose  # no change because change results in invalid pose

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        logging.debug(f"RobotTransitionModel - argmax")
        if isinstance(state, RobotState):
            robot_state = state
        else:
            robot_state = state.object_states[self._robot_id]

        next_robot_state = copy.deepcopy(robot_state)
        # camera direction is only not None when looking
        next_robot_state["camera_direction"] = None
        if isinstance(action, MotionAction):
            # motion action
            next_robot_state["pose"] = RobotTransitionModel.if_move_by(
                self._robot_id, state, action, self._dim
            )
        elif isinstance(action, LookAction):
            if hasattr(action, "motion") and action.motion is not None:
                # rotate the robot
                next_robot_state["pose"] = self._if_move_by(
                    self._robot_id, state, action, self._dim
                )
            next_robot_state["camera_direction"] = action.name
        elif isinstance(action, FindAction):
            robot_pose = state.pose(self._robot_id)
            observations = self._sensor.observe(robot_pose, state)
            # Update "objects_found" set for target and avoid objects
            observed_target_objects = {
                objid
                for objid in observations.objposes
                if (
                    state.object_states[objid].objclass != "obstacle"
                    and observations.objposes[objid] != ObjectObservation.NULL
                )
            }
            next_robot_state["objects_found"] = tuple(
                set(next_robot_state["objects_found"]) | set(observed_target_objects)
            )
        return next_robot_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        logging.debug(f"RobotTransitionModel - sample")
        return self.argmax(state, action)
    
    @classmethod
    def valid_pose(cls, pose, width, length, state=None, check_collision=True, pose_objid=None):
        """
        Returns True if the given `pose` (x,y) is a valid pose;
        If `check_collision` is True, then the pose is only valid
        if it is not overlapping with any object pose in the environment state.
        """
        logging.debug(f"TransitionModel - valid_pose")
        x, y = pose

        # Check collision with obstacles
        if check_collision and state is not None:
            object_poses = state.object_poses
            for objid in object_poses:
                logging.debug(f"Check obj id - {objid} and robot id - {pose_objid}")
                if state.object_states[objid].objclass.startswith("obstacle"):
                    if objid == pose_objid:
                        logging.debug(f"Impossible")
                        continue
                    logging.debug(f"pose - {pose}")
                    logging.debug(f"Obj {objid} pose - {object_poses[objid]}")
                    if pose == object_poses[objid]:
                        return False
        return cls.in_boundary(pose, width, length)

    @staticmethod
    def in_boundary(pose, width, length):
        """Check if pose is within the world boundaries."""
        x, y = pose
        return 0 <= x < width and 0 <= y < length
    
def a_star_search(start, goal, width, length, obstacles):
    """ A* search algorithm to find the path from start to goal. """
    
    # Directions: up, down, left, right (no diagonals)
    directions = [EAST, WEST, NORTH, SOUTH]
    
    # Priority queue for A* search
    open_list = []
    heapq.heappush(open_list, (0, start))  # (cost, position)
    
    # G, F, and parent dictionaries
    g_cost = {start: 0}
    f_cost = {start: heuristic(start, goal)}
    parent = {start: None}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        # If we reached the goal, reconstruct the path
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        
        # Explore neighbors
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(neighbor, width, length, obstacles):
                tentative_g = g_cost[current] + 1  # Assumed cost per move is 1
                
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_cost[neighbor] = tentative_g + heuristic(neighbor, goal)
                    parent[neighbor] = current
                    heapq.heappush(open_list, (f_cost[neighbor], neighbor))
    
    return []  # No path found


def heuristic(pos, goal):
    """ Heuristic function (Manhattan distance) for A* """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def is_valid(pos, width, length, obstacles):
    """ Check if the position is valid (within bounds and not an obstacle) """
    x, y = pos
    return 0 <= x < width and 0 <= y < length and pos not in obstacles
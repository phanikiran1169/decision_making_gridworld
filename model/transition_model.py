import logging
import pomdp_py
import copy
from domain.action import *
from domain.state import MosOOState, RobotState
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
            objid: StaticObjectTransitionModel(objid, epsilon=epsilon)
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


class StaticObjectTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""

    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state["id"]]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def sample(self, state, action):
        """Returns next_object_state"""
        logging.debug(f"StaticObjectTransitionModel - sample")
        return self.argmax(state, action)

    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        logging.debug(f"StaticObjectTransitionModel - argmax")
        return copy.deepcopy(state.object_states[self._objid])


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
            # Update "objects_found" set for target objects
            observed_target_objects = {
                objid
                for objid in observations.objposes
                if (
                    state.object_states[objid].objclass == "target"
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
        x, y = pose

        # Check collision with obstacles
        if check_collision and state is not None:
            object_poses = state.object_poses
            for objid in object_poses:
                if state.object_states[objid].objclass.startswith("obstacle"):
                    if objid == pose_objid:
                        continue
                    if (x, y) == object_poses[objid]:
                        return False
        return cls.in_boundary(pose, width, length)

    @staticmethod
    def in_boundary(pose, width, length):
        """Check if pose is within the world boundaries."""
        x, y = pose
        return 0 <= x < width and 0 <= y < length
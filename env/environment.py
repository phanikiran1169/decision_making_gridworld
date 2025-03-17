import logging
import copy
import pomdp_py
from domain.state import ObjectState
from model.reward_model import GoalRewardModel
from model.transition_model import MosTransitionModel

class MosEnvironment(pomdp_py.Environment):
    """"""

    def __init__(self, dim, init_state, sensors, obstacles=set({})):
        """
        Args:
            sensors (dict): Map from robot_id to sensor (Sensor);
                            Sensors equipped on robots; Used to determine
                            which objects should be marked as found.
            obstacles (set): set of object ids that represent objects in the environment 
                            that can either be targets (objects to be reached) or avoidable 
                            objects (objects to be avoided). The classification of an object 
                            as a target or avoidable is determined by its `objclass`.
        """

        self.width, self.length = dim
        self.sensors = sensors
        self.obstacles = obstacles
        transition_model = MosTransitionModel(
            dim, sensors, set(init_state.object_states.keys())
        )
        
        # Initialize target and avoid objects
        self.target_objects = {
            objid
            for objid, state in init_state.object_states.items()
            if isinstance(state, ObjectState) and state.objclass == "target"
        }
        
        self.avoid_objects = {
            objid
            for objid, state in init_state.object_states.items()
            if isinstance(state, ObjectState) and state.objclass == "avoid"
        }

        reward_model = GoalRewardModel(self.target_objects)
        super().__init__(init_state, transition_model, reward_model)

    @property
    def robot_ids(self):
        return set(self.sensors.keys())

    def state_transition(self, action, execute=True, robot_id=None):
        """state_transition(self, action, execute=True, **kwargs)

        Overriding parent class function.
        Simulates a state transition given `action`. If `execute` is set to True,
        then the resulting state will be the new current state of the environment.

        Args:
            action (Action): action that triggers the state transition
            execute (bool): If True, the resulting state of the transition will
                            become the current state.

        Returns:
            float or tuple: reward as a result of `action` and state
            transition, if `execute` is True (next_state, reward) if `execute`
            is False.

        """
        logging.debug("MosEnvironment - state_transition")
        assert (
            robot_id is not None
        ), "state transition should happen for a specific robot"

        next_state = copy.deepcopy(self.state)
        next_state.object_states[robot_id] = self.transition_model[robot_id].sample(
            self.state, action
        )

        reward = self.reward_model.sample(
            self.state, action, next_state, robot_id=robot_id
        )
        if execute:
            self.apply_transition(next_state)
            return reward
        else:
            return next_state, reward

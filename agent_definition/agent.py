import logging
import pomdp_py
from agent_definition.belief import initialize_belief
from model.transition_model import MosTransitionModel
from model.observation_model import MosObservationModel
from model.reward_model import GoalRewardModel
from model.policy_model import PolicyModel

class MosAgent(pomdp_py.Agent):
    """One agent is one robot."""

    def __init__(
        self,
        robot_id,
        init_robot_state,  # initial robot state (assuming robot state is observable perfectly)
        object_ids,  # target object ids
        dim,  # tuple (w,l) of the width (w) and length (l) of the gridworld search space.
        sensor,  # Sensor equipped on the robot
        sigma=0.01,  # parameter for observation modelv6
        epsilon=1,  # parameter for observation model
        belief_rep="histogram",  # belief representation, either "histogram" or "particles".
        prior={},  # prior belief, as defined in belief.py:initialize_belief
        num_particles=100,  # used if the belief representation is particles
        grid_map=None,
    ):  # GridMap used to avoid collision with obstacles (None if not provided)
        self.robot_id = robot_id
        self._object_ids = object_ids
        self.sensor = sensor

        # since the robot observes its own pose perfectly, it will have 100% prior
        # on this pose.
        prior[robot_id] = {init_robot_state.pose: 1.0}

        # initialize belief
        init_belief = initialize_belief(
            dim,
            self.robot_id,
            self._object_ids,
            prior=prior,
            representation=belief_rep,
            num_particles=num_particles,
        )
        transition_model = MosTransitionModel(
            dim, {self.robot_id: self.sensor}, self._object_ids
        )
        observation_model = MosObservationModel(
            dim, self.sensor, self._object_ids, sigma=sigma, epsilon=epsilon
        )
        reward_model = GoalRewardModel(self._object_ids, robot_id=self.robot_id)
        policy_model = PolicyModel(self.robot_id, grid_map=grid_map)
        super().__init__(
            init_belief,
            policy_model,
            transition_model=transition_model,
            observation_model=observation_model,
            reward_model=reward_model,
        )


    def clear_history(self):
        """Custom function to clear agent's action-observation history."""
        self._history = None
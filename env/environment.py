import logging
import copy
import pomdp_py
from description.state import GridWorldState, EvaderState, ObstacleState
from description.action import MotionAction
from model.observation_model import GridWorldObservationModel
from model.reward_model import GridWorldRewardModel
from model.transition_model import GridWorldTransitionModel
from model.policy_model import GridWorldPolicyModel

class GridWorldEnvironment(pomdp_py.Environment):
    """
    Deterministic Environment for GridWorld Evader scenario.
    Combines Transition, Reward, and Observation Models.
    """

    def __init__(self, grid_size, init_state):
        self.grid_width, self.grid_height = grid_size
        transition_model = GridWorldTransitionModel(grid_size)
        reward_model = GridWorldRewardModel(robot_id='evader')
        self.observation_model = GridWorldObservationModel(grid_size)
        super().__init__(init_state, transition_model, reward_model)
        logging.debug(f"[Environment initialized with state: {init_state}")
        # self.policy_model = GridWorldPolicyModel(grid_size)


    def state_transition(self, action, execute=True):
        """Execute action and optionally apply it to update state"""
        next_state = self.transition_model.sample(self.state, action)
        reward = self.reward_model.sample(self.state, action, next_state)
        logging.debug(f"[Transitioning from {self.state} using {action} to {next_state}")

        if execute:
            self.apply_transition(next_state)
            logging.debug(f"[Applied transition. New state: {self.state}")
            return next_state, reward
        else:
            return next_state, reward

    def in_terminal_state(self):
        """Returns True if the agent has reached the goal."""
        evader_pos = self.state.evader.pose
        if evader_pos == self.state.evader.goal_pose:
            logging.debug("Agent has reached the goal! Terminating simulation.")
            return True
        return False
    
    def provide_observation(self, observation_model, action):
        """Uses the observation model to generate an observation based on the current state."""
        logging.debug(f"Current state in observation model - {self.state}")
        return observation_model.sample(self.state, action)
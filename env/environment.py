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

        self.policy_model = GridWorldPolicyModel(grid_size)


    def state_transition(self, action, execute=True):
        """Execute action and optionally apply it to update state"""
        next_state = self.transition_model.sample(self.state, action)
        reward = self.reward_model.sample(self.state, action, next_state)

        if execute:
            self.state = next_state
            return next_state, reward
        else:
            return next_state, reward

    def is_goal_reached(self):
        """Check if the agent reached its goal"""
        return self.state.evader.pose == self.state.evader.goal_pose

    def in_terminal_state(self):
        """Define terminal condition clearly"""
        return self.is_goal_reached()
import logging
import pomdp_py
from description.belief import GridWorldBelief
from model.transition_model import GridWorldTransitionModel
from model.observation_model import GridWorldObservationModel
from model.reward_model import GridWorldRewardModel
from model.policy_model import GridWorldPolicyModel

class GridWorldAgent(pomdp_py.Agent):
    """
    Agent for the GridWorld Pursuit scenario.
    Maintains beliefs about obstacles and uses previously defined models.
    """

    def __init__(self,
                 grid_size,
                 init_evader_pose,
                 goal_pose,
                 obstacle_prior=None):
        """
        Initializes the GridWorldAgent.

        Args:
            grid_size (tuple): (width, height) of grid.
            init_evader_pose (tuple): (x,y) initial position of evader.
            goal_pose (tuple): (x,y) goal position.
            belief (GridWorldBelief): The belief representation for the agent.
        """
        self.grid_width, self.grid_height = grid_size
        self.robot_id = 'evader'
        self.init_evader_pose = init_evader_pose
        self.goal_pose = goal_pose

        # Initialize models
        transition_model = GridWorldTransitionModel(grid_size)
        observation_model = GridWorldObservationModel(grid_size)
        reward_model = GridWorldRewardModel(robot_id=self.robot_id)
        policy_model = GridWorldPolicyModel(grid_size)
        belief = GridWorldBelief(evader_id=self.robot_id, 
                                grid_size=grid_size, 
                                evader_pose=self.init_evader_pose, 
                                goal_pose=self.goal_pose, 
                                obstacle_prior=obstacle_prior)

        super().__init__(belief,
                         policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

    def clear_history(self):
        """Custom function to clear agent's action-observation history."""
        self._history = None

    @property
    def current_pose(self):
        """Returns current known pose of evader (fully observable)."""
        return self.init_evader_pose

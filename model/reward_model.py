import pomdp_py
from description.action import MotionAction


class GridWorldRewardModel(pomdp_py.RewardModel):
    """
    Reward model for the evader agent in a grid world.

    Rewards:
        -1 for regular movement
        -25 for collision with obstacle
        +100 for reaching goal
    """

    MOVE_PENALTY = -1
    OBSTACLE_PENALTY = -25
    GOAL_REWARD = 100

    def __init__(self, robot_id):
        self.robot_id = robot_id

    def probability(self, reward, state, action, next_state, normalized=False):
        """Deterministic reward model"""
        actual_reward = self._reward_func(state, action, next_state)
        return 1.0 if reward == actual_reward else 0.0

    def sample(self, state, action, next_state, normalized=False):
        """Returns deterministic reward"""
        return self._reward_func(state, action, next_state)

    def argmax(self, state, action, next_state, normalized=False, **kwargs):
        """Returns deterministic reward"""
        return self._reward_func(state, action, next_state)

    def _reward_func(self, state, action, next_state):
        # Check if the action is a MotionAction
        if not isinstance(action, MotionAction):
            return 0  # Non-motion actions don't affect reward

        evader_next_pos = next_state.evader.pose

        # Check if reached the goal
        if evader_next_pos == next_state.evader.goal_pose:
            return self.GOAL_REWARD

        # Check collision with obstacle
        if next_state.obstacle_at(evader_next_pos):
            return self.OBSTACLE_PENALTY

        # Regular movement penalty
        return self.MOVE_PENALTY
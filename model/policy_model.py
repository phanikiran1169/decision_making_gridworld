import pomdp_py
import random
from description.action import ALL_MOTION_ACTIONS

class GridWorldPolicyModel(pomdp_py.RolloutPolicy):
    """
    Simple rollout policy model for GridWorld pursuit scenario.
    Randomly selects a valid movement action.
    """

    def __init__(self, grid_size):
        self.grid_width, self.grid_height = grid_size

    def sample(self, state, **kwargs):
        valid_actions = self.get_all_actions(state)
        return random.choice(valid_actions)

    def probability(self, action, state, **kwargs):
        valid_actions = self.get_all_actions(state)
        return 1.0 / len(valid_actions) if action in valid_actions else 0.0

    def argmax(self, state, **kwargs):
        # No meaningful argmax; returns random action for simplicity
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        valid_actions = []
        evader_pos = state.evader.pose

        for action in ALL_MOTION_ACTIONS:
            dx, dy = action.motion
            next_pos = (evader_pos[0] + dx, evader_pos[1] + dy)

            # Include only actions leading to valid cells
            if state.within_bounds(next_pos, (self.grid_width, self.grid_height)) and not state.obstacle_at(next_pos):
                valid_actions.append(action)

        # If no valid actions (fully blocked), return empty list
        return valid_actions if valid_actions else ALL_MOTION_ACTIONS

    def rollout(self, state, history=None):
        return self.sample(state)
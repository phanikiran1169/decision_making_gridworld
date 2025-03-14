import logging
import pomdp_py
import random
from description.action import ALL_MOTION_ACTIONS, LookAction, FindAction, MotionAction

class GridWorldPolicyModel(pomdp_py.RolloutPolicy):
    """
    Simple rollout policy model for GridWorld pursuit scenario.
    Randomly selects a valid movement action.
    """

    BELIEF_THRESHOLD_FOR_FIND = 0.9

    def __init__(self, grid_size):
        self.grid_width, self.grid_height = grid_size

    def sample(self, state, **kwargs):
        """Selects the best action based on a heuristic approach."""
        logging.info("GridWorldPolicyModel - Sample")
        valid_actions = self.get_all_actions(state, kwargs.get("belief", None), kwargs.get("history", None))

        if not valid_actions:
            logging.info("No valid actions available. Defaulting to LookAction.")
            return LookAction()

        # Prioritize FindAction if belief about goal is high
        if FindAction() in valid_actions:
            return FindAction()

        # Prioritize motion actions that move closer to the goal
        best_action = self._choose_best_movement(state, valid_actions)
        if best_action:
            return best_action

        # Otherwise, take LookAction as a fallback
        return LookAction()

    def _choose_best_movement(self, state, valid_actions):
        """Chooses the best movement action based on distance to the goal."""
        evader_pos = state.evader.pose
        goal_x, goal_y = state.evader.goal_pose

        def distance_to_goal(action):
            dx, dy = action.motion
            next_x, next_y = evader_pos[0] + dx, evader_pos[1] + dy
            return abs(next_x - goal_x) + abs(next_y - goal_y)  # Manhattan Distance

        movement_actions = [a for a in valid_actions if isinstance(a, MotionAction)]
        if not movement_actions:
            return None  # No valid movement actions

        # Select the action that minimizes distance to the goal
        return min(movement_actions, key=distance_to_goal)


    def probability(self, action, state, **kwargs):
        valid_actions = self.get_all_actions(state)
        return 1.0 / len(valid_actions) if action in valid_actions else 0.0

    def argmax(self, state, **kwargs):
        # No meaningful argmax; returns random action for simplicity
        return self.sample(state)

    def get_all_actions(self, state=None, belief=None, history=None):
        """Ensures the agent always has actions, including Look & Find"""
        logging.info("GridWorldPolicyModel - get_all_actions")
        if state is None:
            return ALL_MOTION_ACTIONS + [LookAction()] + [FindAction()]

        valid_actions = []
        evader_pos = state.evader.pose

        logging.info(f"[Evaluating actions for state: {state}")
        logging.error(f"history - {history}")

        # Check valid motion actions
        for action in ALL_MOTION_ACTIONS:
            dx, dy = action.motion
            next_pos = (evader_pos[0] + dx, evader_pos[1] + dy)

            # logging.info(f"[Checking move {action}: {evader_pos} -> {next_pos}")

            # Skip out-of-bounds moves
            if not state.within_bounds(next_pos, (self.grid_width, self.grid_height)):
                logging.info(f"[Move {action} out of bounds.")
                continue

            # Skip if obstacle
            if state.obstacle_at(next_pos):
                logging.info(f"[Obstacle detected at {next_pos}, skipping move {action}.")
                continue

            valid_actions.append(action)

        # Always include LookAction
        valid_actions.append(LookAction())
        valid_actions.append(FindAction())

        logging.info(f"[Final Available Actions: {valid_actions}")
        return valid_actions
        
    def rollout(self, state, history=None):
        return self.sample(state, history=history)
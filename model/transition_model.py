import logging
import pomdp_py
import copy
from description.action import MotionAction, LookAction, FindAction
from description.state import GridWorldState, EvaderState

class GridWorldTransitionModel(pomdp_py.TransitionModel):
    """
    Deterministic transition model for GridWorld Pursuit scenario.
    Evader moves in cardinal directions unless blocked by an obstacle or boundary.
    """

    def __init__(self, grid_size):
        self.grid_width, self.grid_height = grid_size

    def probability(self, next_state, state, action):
        # Deterministic transition
        expected_state = self.argmax(state, action)
        return 1.0 if next_state == expected_state else 0.0

    def sample(self, state, action):
        # Deterministic, simply returns argmax
        logging.debug("GridWorldTransitionModel - Sample")
        return self.argmax(state, action)

    def argmax(self, state, action):
        next_state = copy.deepcopy(state)

        if state.evader.pose == state.evader.goal_pose:
            logging.debug("Agent reached the goal. No state updates needed.")
            return state

        if isinstance(action, MotionAction):
            current_pos = state.evader.pose
            dx, dy = action.motion
            intended_pos = (current_pos[0] + dx, current_pos[1] + dy)

            logging.debug(f"[Checking transition: {current_pos} -> {intended_pos}")

            if next_state.within_bounds(intended_pos, (self.grid_width, self.grid_height)):
                if next_state.obstacle_at(intended_pos):
                    logging.debug(f"[Obstacle at {intended_pos}, staying at {current_pos}")
                    return state  # Stay in place if blocked
                else:
                    logging.debug(f"[Moving to {intended_pos}")
                    next_state.object_states["evader"] = EvaderState(
                        agent_id="evader",
                        pose=intended_pos,
                        goal_pose=state.evader.goal_pose
                    )

        elif isinstance(action, LookAction):
            logging.debug(f"[LookAction at {state.evader.pose}. No movement occurs.")
            return state  # No movement, but observation will update belief

        elif isinstance(action, FindAction):
            if state.evader.pose == state.evader.goal_pose:
                logging.debug("FindAction successful! Goal reached.")
                return state  # Reward will be handled in the reward model
            else:
                logging.debug("FindAction failed. Staying in place.")
                return state  # Stay in place, but penalty applies

        return next_state
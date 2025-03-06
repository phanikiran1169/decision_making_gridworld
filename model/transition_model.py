import pomdp_py
import copy
from description.action import MotionAction
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
        return self.argmax(state, action)

    def argmax(self, state, action):
        next_state = copy.deepcopy(state)

        if isinstance(action, MotionAction):
            current_pos = state.evader.pose
            dx, dy = action.motion
            intended_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Check for collisions or boundary violations
            if next_state.within_bounds(intended_pos, (self.grid_width, self.grid_height)) \
                    and not next_state.obstacle_at(intended_pos):
                # Move to new position if valid
                next_state.object_states["evader"] = EvaderState(
                    agent_id="evader",
                    pose=intended_pos,
                    goal_pose=state.evader.goal_pose
                )
            # Else: No change, stays in current position (collision or boundary violation)

        # Other actions (if any) can be handled here, but you have only MotionAction.
        return next_state
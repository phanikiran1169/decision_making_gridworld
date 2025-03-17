import logging
import pomdp_py
import random
from domain.action import ALL_MOTION_ACTIONS, Look, Find, FindAction, LookAction
from model.transition_model import RobotTransitionModel

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, grid_size=None):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self.grid_size = grid_size

    def sample(self, state, **kwargs):
        logging.debug(f"PolicyModel - sample")
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        logging.debug(f"PolicyModel - get_all_actions")
        can_find = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
        find_action = [Find] if can_find else []
        if state is None:
            logging.debug(f"{ALL_MOTION_ACTIONS + [Look] + find_action}")
            return ALL_MOTION_ACTIONS + [Look] + find_action
        else:
            if state is None:
                logging.debug(f"{ALL_MOTION_ACTIONS + [Look] + find_action}")
                return ALL_MOTION_ACTIONS + [Look] + find_action
            else:
                robot_pose = state.pose(self.robot_id)  # Get robot's current position

                # Validate motion actions using transition model's valid_pose function
                valid_motions = [
                    action for action in ALL_MOTION_ACTIONS
                    if RobotTransitionModel.valid_pose(
                        (robot_pose[0] + action.motion[0], robot_pose[1] + action.motion[1]),
                        self.grid_size[0],
                        self.grid_size[1],
                        state=state,
                        check_collision=True,
                        pose_objid=self.robot_id,
                    )
                ]

                logging.debug(f"Valid Motions: {valid_motions}")
                logging.debug(f"{valid_motions + [Look] + find_action}")
                return valid_motions + [Look] + find_action
            
    def is_valid_motion(self, current_pose, action):
        """
        Checks if a motion action is valid (within bounds and not colliding with obstacles).
        """
        if not isinstance(action, pomdp_py.Action):
            return False
        
        new_x, new_y = current_pose[0] + action.motion[0], current_pose[1] + action.motion[1]

        # Check if new position is within grid bounds
        if not (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]):
            return False

        # Check if new position collides with an obstacle
        if (new_x, new_y) in self.obstacles:
            return False

        return True

    def rollout(self, state, history=None):
        logging.debug(f"PolicyModel - rollout")
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

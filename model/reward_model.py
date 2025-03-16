import logging
import pomdp_py
from domain.action import MotionAction, LookAction, FindAction


class MosRewardModel(pomdp_py.RewardModel):
    def __init__(self, target_objects, big=1000, small=1, robot_id=None):
        """
        robot_id (int): This model is the reward for one agent (i.e. robot),
                        If None, then this model could be for the environment.
        target_objects (set): a set of objids for target objects.
        """
        self._robot_id = robot_id
        self.big = big
        self.small = small
        self._target_objects = target_objects

    def probability(
        self, reward, state, action, next_state, normalized=False, **kwargs
    ):
        if reward == self._reward_func(state, action):
            return 1.0
        else:
            return 0.0

    def sample(self, state, action, next_state, normalized=False, robot_id=None):
        # deterministic
        return self._reward_func(state, action, next_state, robot_id=robot_id)

    def argmax(self, state, action, next_state, normalized=False, robot_id=None):
        """Returns the most likely reward"""
        return self._reward_func(state, action, next_state, robot_id=robot_id)


class GoalRewardModel(MosRewardModel):
    """
    This is a reward where the agent gets reward only for detect-related actions.
    """

    def _reward_func(self, state, action, next_state, robot_id=None):
        if robot_id is None:
            assert (
                self._robot_id is not None
            ), "Reward must be computed with respect to one robot."
            robot_id = self._robot_id

        reward = 0

        # If the robot has caught the object
        for objid in state.object_states:
            if state.object_states[objid].objclass == "obstacle":
                if state.object_states[objid]["pose"] == state.object_states[robot_id]["pose"]:
                    reward -= self.big
                else:
                    pass
            elif state.object_states[objid].objclass == "target":
                if state.object_states[objid]["pose"] == state.object_states[robot_id]["pose"]:
                    return 0  # no reward or penalty; the task is finished
                else:
                    pass
            else:
                pass

        if isinstance(action, MotionAction):
            reward = reward - self.small - action.distance_cost
        elif isinstance(action, LookAction):
            reward = reward - self.small
        elif isinstance(action, FindAction):
            if state.object_states[robot_id]["camera_direction"] is None:
                # The robot didn't look before detect. So nothing is in the field of view.
                reward -= self.big
            else:
                # transition function should've taken care of the detection.
                new_objects_count = len(
                    set(next_state.object_states[robot_id].objects_found)
                    - set(state.object_states[robot_id].objects_found)
                )
                if new_objects_count == 0:
                    # No new detection. "detect" is a bad action.
                    reward -= self.big
                else:
                    # New detection. Award.
                    reward += self.big
        return reward

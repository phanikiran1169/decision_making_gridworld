import logging
import random
import pomdp_py
from domain.observation import MosOOObservation, ObjectObservation
from domain.action import MotionAction, LookAction
from domain.state import ObjectState

class MosObservationModel(pomdp_py.OOObservationModel):
    """Object-oriented transition model"""

    def __init__(self, dim, sensor, object_ids, sigma=0.01, epsilon=1):
        self.sigma = sigma
        self.epsilon = epsilon
        observation_models = {
            objid: ObjectObservationModel(
                objid, sensor, dim, sigma=sigma, epsilon=epsilon
            )
            for objid in object_ids
        }
        pomdp_py.OOObservationModel.__init__(self, observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        if not isinstance(action, LookAction):
            return MosOOObservation({})
            # return MosOOObservation({objid: ObjectObservationModel.NULL
            #                          for objid in next_state.object_states
            #                          if objid != next_state.object_states[objid].objclass != "robot"})

        factored_observations = super().sample(next_state, action, argmax=argmax)
        return MosOOObservation.merge(factored_observations, next_state)


class ObjectObservationModel(pomdp_py.ObservationModel):
    def __init__(self, objid, sensor, dim, sigma=0, epsilon=1):
        """
        sigma and epsilon are parameters of the observation model (see paper),
        dim (tuple): a tuple (width, length) for the dimension of the world"""
        self._objid = objid
        self._sensor = sensor
        self._dim = dim
        self.sigma = sigma
        self.epsilon = epsilon

    def _compute_params(self, object_in_sensing_region):
        if object_in_sensing_region:
            # Object is in the sensing region
            alpha = self.epsilon
            beta = (1.0 - self.epsilon) / 2.0
            gamma = (1.0 - self.epsilon) / 2.0
        else:
            # Object is not in the sensing region.
            alpha = (1.0 - self.epsilon) / 2.0
            beta = (1.0 - self.epsilon) / 2.0
            gamma = self.epsilon
        return alpha, beta, gamma

    def probability(self, observation, next_state, action, **kwargs):
        """
        Returns the probability of Pr (observation | next_state, action).

        Args:
            observation (ObjectObservation)
            next_state (State)
            action (Action)
        """
        logging.debug("ObjectObservationModel - probability")
        if not isinstance(action, LookAction):
            # No observation should be received
            if observation.pose == ObjectObservation.NULL:
                return 1.0
            else:
                return 0.0

        if observation.objid != self._objid:
            raise ValueError("The observation is not about the same object")

        # histogram belief update using O(oi|si',sr',a).
        next_robot_state = kwargs.get("next_robot_state", None)
        if next_robot_state is not None:
            assert (
                next_robot_state["id"] == self._sensor.robot_id
            ), "Robot id of observation model mismatch with given state"
            robot_pose = next_robot_state.pose

            if isinstance(next_state, ObjectState):
                assert (
                    next_state["id"] == self._objid
                ), "Object id of observation model mismatch with given state"
                object_pose = next_state.pose
            else:
                object_pose = next_state.pose(self._objid)
        else:
            robot_pose = next_state.pose(self._sensor.robot_id)
            object_pose = next_state.pose(self._objid)

        # Compute the probability
        zi = observation.pose
        alpha, beta, gamma = self._compute_params(
            self._sensor.within_range(robot_pose, object_pose, next_state)
        )

        prob = 0.0
        # Event A:
        # object in sensing region and observation comes from object i
        if zi == ObjectObservation.NULL:
            # Even though event A occurred, the observation is NULL.
            # This has 0.0 probability.
            prob += 0.0 * alpha
        else:
            gaussian = pomdp_py.Gaussian(
                list(object_pose), [[self.sigma**2, 0], [0, self.sigma**2]]
            )
            prob += gaussian[zi] * alpha

        # Event B
        prob += (1.0 / self._sensor.sensing_region_size) * beta

        # Event C
        pr_c = 1.0 if zi == ObjectObservation.NULL else 0.0  # indicator zi == NULL
        prob += pr_c * gamma
        return prob

    def sample(self, next_state, action, **kwargs):
        """Returns observation"""
        logging.debug("ObjectObservationModel - sample")
        if not isinstance(action, LookAction):
            # Not a look action. So no observation
            return ObjectObservation(self._objid, ObjectObservation.NULL)

        robot_pose = next_state.pose(self._sensor.robot_id)
        object_pose = next_state.pose(self._objid)

        # Obtain observation according to distribution.
        alpha, beta, gamma = self._compute_params(
            self._sensor.within_range(robot_pose, object_pose, next_state)
        )

        event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        zi = self._sample_zi(event_occured, next_state)

        return ObjectObservation(self._objid, zi)

    def argmax(self, next_state, action, **kwargs):
        logging.debug("ObjectObservationModel - argmax")
        # Obtain observation according to distribution.
        alpha, beta, gamma = self._compute_params(
            self._sensor.within_range(robot_pose, object_pose, next_state)
        )

        event_probs = {"A": alpha, "B": beta, "C": gamma}
        event_occured = max(event_probs, key=lambda e: event_probs[e])
        zi = self._sample_zi(event_occured, next_state, argmax=True)
        return ObjectObservation(self._objid, zi)

    def _sample_zi(self, event, next_state, argmax=False):
        if event == "A":
            object_true_pose = next_state.object_pose(self._objid)
            gaussian = pomdp_py.Gaussian(
                list(object_true_pose), [[self.sigma**2, 0], [0, self.sigma**2]]
            )
            if not argmax:
                zi = gaussian.random()
            else:
                zi = gaussian.mpe()
            zi = (int(round(zi[0])), int(round(zi[1])))

        elif event == "B":

            width, height = self._dim
            zi = (
                random.randint(0, width),  # x axis
                random.randint(0, height),
            )  # y axis
        else:  # event == C
            zi = ObjectObservation.NULL
        return zi
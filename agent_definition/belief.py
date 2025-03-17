import logging
import pomdp_py
import random
import copy
from domain.state import RobotState, MosOOState, ObjectState

class MosOOBelief(pomdp_py.OOBelief):
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""

    def __init__(self, robot_id, object_beliefs):
        """
        robot_id (int): The id of the robot that has this belief.
        object_beliefs (objid -> GenerativeDistribution)
            (includes robot)
        """
        self.robot_id = robot_id
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return MosOOState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return MosOOState(pomdp_py.OOBelief.random(self, **kwargs).object_states)


def initialize_belief(
    dim,
    robot_id,
    object_ids,
    prior={},
    representation="histogram",
    num_particles=100,
):
    """
    Returns a GenerativeDistribution that is the belief representation for
    the multi-object search problem.

    Args:
        dim (tuple): a tuple (width, length) of the search space gridworld.
        robot_id (int): robot id that this belief is initialized for.
        object_ids (dict): a set of object ids that we want to model the belief distribution
                          over; They are `assumed` to be the target objects, not obstacles,
                          because the robot doesn't really care about obstacle locations and
                          modeling them just adds computation cost.
        prior (dict): A mapping {(objid|robot_id) -> {(x,y) -> [0,1]}}. If used, then
                      all locations not included in the prior will be treated to have 0 probability.
                      If unspecified for an object, then the belief over that object is assumed
                      to be a uniform distribution.
        num_particles (int): Maximum number of particles used to represent the belief

    Returns:
        GenerativeDistribution: the initial belief representation.
    """
    if representation == "histogram":
        return _initialize_histogram_belief(
            dim, robot_id, object_ids, prior
        )
    elif representation == "particles":
        return _initialize_particles_belief(
            dim, robot_id, object_ids, num_particles=num_particles
        )
    else:
        raise ValueError("Unsupported belief representation %s" % representation)


def _initialize_histogram_belief(dim, robot_id, object_ids, prior):
    """
    Returns the belief distribution represented as a histogram
    """
    oo_hists = {}  # objid -> Histogram
    width, length = dim
    for objid in object_ids:
        hist = {}  # pose -> prob
        total_prob = 0
        if objid in prior:
            # prior knowledge provided. Just use the prior knowledge
            for pose in prior[objid]:
                state = ObjectState(objid, "target", pose)
                hist[state] = prior[objid][pose]
                total_prob += hist[state]
        else:
            # no prior knowledge. So uniform.
            for x in range(width):
                for y in range(length):
                    state = ObjectState(objid, "target", (x, y))
                    hist[state] = 1.0
                    total_prob += hist[state]

        # Normalize
        for state in hist:
            hist[state] /= total_prob

        hist_belief = pomdp_py.Histogram(hist)
        oo_hists[objid] = hist_belief

    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    oo_hists[robot_id] = pomdp_py.Histogram(
        {RobotState(robot_id, init_robot_pose, (), None): 1.0}
    )

    return MosOOBelief(robot_id, oo_hists)


def _initialize_particles_belief(
    dim, robot_id, object_ids, prior, num_particles=100
):
    """This returns a single set of particles that represent the distribution over a
    joint state space of all objects.

    Since it is very difficult to provide a prior knowledge over the joint state
    space when the number of objects scales, the prior (which is
    object-oriented), is used to create particles separately for each object to
    satisfy the prior; That is, particles beliefs are generated for each object
    as if object_oriented=True. Then, `num_particles` number of particles with
    joint state is sampled randomly from these particle beliefs.

    """
    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]

    oo_particles = {}  # objid -> Particageles
    width, length = dim
    for objid in object_ids:
        particles = [
            RobotState(robot_id, init_robot_pose, (), None)
        ]  # list of states; Starting the observable robot state.
        if objid in prior:
            # prior knowledge provided. Just use the prior knowledge
            prior_sum = sum(prior[objid][pose] for pose in prior[objid])
            for pose in prior[objid]:
                state = ObjectState(objid, "target", pose)
                amount_to_add = (prior[objid][pose] / prior_sum) * num_particles
                for _ in range(amount_to_add):
                    particles.append(state)
        else:
            # no prior knowledge. So uniformly sample `num_particles` number of states.
            for _ in range(num_particles):
                x = random.randrange(0, width)
                y = random.randrange(0, length)
                state = ObjectState(objid, "target", (x, y))
                particles.append(state)

        particles_belief = pomdp_py.Particles(particles)
        oo_particles[objid] = particles_belief

    # Return Particles distribution which contains particles
    # that represent joint object states
    particles = []
    for _ in range(num_particles):
        object_states = {}
        for objid in oo_particles:
            random_particle = random.sample(oo_particles[objid], 1)[0]
            object_states[objid] = copy.deepcopy(random_particle)
        particles.append(MosOOState(object_states))
    return pomdp_py.Particles(particles)

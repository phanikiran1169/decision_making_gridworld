import pomdp_py
import random
from description.state import GridWorldState, EvaderState, ObstacleState

class GridWorldBelief(pomdp_py.GenerativeDistribution):
    """
    Histogram (tabular) belief distribution for GridWorld scenario
    Represents uncertainty over obstacle locations (free/obstacle)
    """

    def __init__(self, grid_size, evader_pose, goal_pose, obstacle_prior=None):
        """
        Args:
            grid_size: (width, height) of grid world
            evader_pose: Known initial pose of evader (fully observable)
            goal_pose: Known location of goal
            obstacle_prior: {(x,y): probability_of_obstacle}
                            If None, assumes uniform uncertainty
        """
        self.grid_width, self.grid_height = grid_size
        self.evader_pose = evader_pose
        self.goal_pose = goal_pose

        self.obstacle_prior = obstacle_prior or self._uniform_prior()
        self.histogram = self._initialize_histogram()

    def _uniform_prior(self):
        """Uniform probability (0.5 obstacle, 0.5 free) for unknown cells"""
        prior = {}
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if (x, y) != self.evader_pose and (x, y) != self.goal_pose:
                    prior[(x, y)] = 0.5
        return prior

    def _initialize_histogram(self):
        """
        Initialize histogram belief as {(x,y): obstacle_probability}
        """
        histogram = {}
        for pos, prob in self.obstacle_prior.items():
            histogram[pos] = prob
        return histogram

    def update(self, observation):
        """
        Update histogram belief based on the received observation
        Args:
            observation: EvaderObservation from observation_model
        """
        for pos, status in observation.observed_cells.items():
            if status == "free":
                self.histogram[pos] = 0.0
            elif status == "obstacle":
                self.histogram[pos] = 1.0
            # Unknown cells remain unchanged

    def mpe(self):
        """Returns most probable GridWorldState (MPE)"""
        obstacles = {}
        for pos, prob in self.histogram.items():
            if prob >= 0.5:
                obs_id = f"obs_{pos[0]}_{pos[1]}"
                obstacles[obs_id] = ObstacleState(obs_id, pos)
        evader_state = EvaderState('evader', self.evader_pose, self.goal_pose)
        return GridWorldState(evader_state, obstacles)

    def random(self):
        """Samples random GridWorldState based on current belief"""
        obstacles = {}
        for pos, prob in self.histogram.items():
            if random.random() < prob:
                obs_id = f"obs_{pos[0]}_{pos[1]}"
                obstacles[obs_id] = ObstacleState(obs_id, pos)
        evader_state = EvaderState('evader', self.evader_pose, self.goal_pose)
        return GridWorldState(evader_state, obstacles)
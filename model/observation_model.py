import pomdp_py
from description.observation import EvaderObservation, CellObservation
from description.action import MotionAction
from description.state import GridWorldState

class GridWorldObservationModel(pomdp_py.ObservationModel):
    """
    Deterministic observation model for Grid World Evader.

    Observations:
        - Observes cells as "free" or "obstacle" along cardinal directions,
          up to the first obstacle.
    """

    def __init__(self, grid_size):
        self.grid_width, self.grid_height = grid_size

    def probability(self, observation, next_state, action, normalized=False, **kwargs):
        # Deterministic observation
        expected_obs = self.sample(next_state, action)
        return 1.0 if observation == expected_obs else 0.0

    def sample(self, next_state, action):
        evader_pos = next_state.evader.pose
        observed_cells = {}

        # Check visibility in all cardinal directions
        directions = [MotionAction.NORTH, MotionAction.SOUTH, MotionAction.EAST, MotionAction.WEST]

        for dx, dy in directions:
            x, y = evader_pos
            while True:
                x += dx
                y += dy
                if not next_state.within_bounds((x, y), (self.grid_width, self.grid_height)):
                    break  # outside grid
                if next_state.obstacle_at((x, y)):
                    observed_cells[(x, y)] = CellObservation.OBSTACLE
                    break  # vision blocked by obstacle
                else:
                    observed_cells[(x, y)] = CellObservation.FREE

        return EvaderObservation(observed_cells)

    def argmax(self, next_state, action, **kwargs):
        # Deterministic, argmax is the same as sample
        return self.sample(next_state, action)
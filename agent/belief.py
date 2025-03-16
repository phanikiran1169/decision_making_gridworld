import logging
import pomdp_py
import random
from domain.state import GridWorldState, EvaderState, ObstacleState
from domain.action import LookAction
from domain.observation import CellObservation

class GridWorldBelief(pomdp_py.OOBelief):
    """
    Object-Oriented Belief (OOBelief) for a grid world with multiple objects.
    Stores separate beliefs for the evader and obstacles.
    Once an obstacle is observed, its belief remains fixed (no uncertainty).
    """
    
    def __init__(self, evader_id, grid_size, evader_pose, goal_pose, obstacle_prior=None):
        self.evader_id = evader_id
        self.grid_width, self.grid_height = grid_size
        self.evader_pose = evader_pose
        self.goal_pose = goal_pose
        
        object_beliefs = self._initialize_obstacle_beliefs(obstacle_prior)
        object_beliefs[self.evader_id] = pomdp_py.Histogram({evader_pose: 1.0})
        
        super().__init__(object_beliefs)

    def _initialize_obstacle_beliefs(self, obstacle_prior):
        """Initializes beliefs for obstacles, excluding evader and goal positions."""
        beliefs = {}
        logging.debug(f"obstacle prior in init belief method - {obstacle_prior}")
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                pos = (x, y)    
                if pos in {self.evader_pose, self.goal_pose}:
                    continue
                
                prior_prob = obstacle_prior.get(pos, 0.0) if obstacle_prior else 0.0
                obs_id = f"obstacle_{x}_{y}"
                beliefs[obs_id] = pomdp_py.Histogram({pos: prior_prob})

        logging.debug(f"Belief - {beliefs['obstacle_0_1'].histogram.get((0,1))}")
        return beliefs

    def update(self, action, observation):
        """Updates belief based on received observation."""
        if isinstance(action, LookAction):
            for pos, status in observation.observed_cells.items():
                obs_id = f"obstacle_{pos[0]}_{pos[1]}"
                if status == CellObservation.FREE:
                    self.object_beliefs.pop(obs_id, None)
                elif status == CellObservation.OBSTACLE:
                    self.object_beliefs[obs_id] = pomdp_py.Histogram({pos: 1.0})
    
    def mpe(self, **kwargs):
        """Returns the most probable GridWorldState (MPE)."""
        mpe_state = pomdp_py.OOBelief.mpe(self, **kwargs)
        evader = EvaderState(self.evader_id, mpe_state.object_states['evader'], self.goal_pose)
        obstacles = {
            obj_id: ObstacleState(obj_id, pos) 
            for obj_id, pos in mpe_state.object_states.items()
            if obj_id != 'evader' and obj_id.startswith("obstacle") and self.object_beliefs[obj_id].histogram.get(pos, 0.0) > 0.0
        }
        logging.debug(f"Most probable GridWorldState - {obstacles}")
        return GridWorldState(evader, obstacles)
    
    def random(self, **kwargs):
        """Samples a random GridWorldState."""
        logging.debug("GridWorldBelief - Random")
        random_state = {}
        for obj_id, belief in self.object_beliefs.items():
            if isinstance(belief, pomdp_py.Histogram):
                if sum(belief.histogram.values()) > 0:  # If there is at least one valid sample
                    random_state[obj_id] = belief.random()
                elif obj_id.startswith("obstacle"):  # If it is an obstacle with zero probability, remove it
                    # logging.debuging(f"Removing {obj_id} from belief due to zero probability mass.")
                    pass
                else:
                    random_state[obj_id] = belief.mpe()  # Default to MPE for evader

        evader = EvaderState(self.evader_id, random_state[self.evader_id], self.goal_pose)

        obstacles = {
            obj_id: ObstacleState(obj_id, pos)
            for obj_id, pos in random_state.items()
            if obj_id != 'evader' and obj_id.startswith("obstacle")
        }

        logging.debug(f"Random GridWorldState - {GridWorldState(evader, obstacles)}")
        return GridWorldState(evader, obstacles)    

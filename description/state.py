import logging
import pomdp_py

class ObstacleState(pomdp_py.ObjectState):
    def __init__(self, obj_id, pose):
        """Obstacle with unique ID and (x,y) position"""
        super().__init__("obstacle", {"pose": pose, "id": obj_id})

    def __str__(self):
        return f"ObstacleState({self.pose})"

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def obj_id(self):
        return self.attributes["id"]


class EvaderState(pomdp_py.ObjectState):
    def __init__(self, agent_id, pose, goal_pose):
        """
        Evader agent in gridworld.
        - pose: (x, y)
        - goal_pose: (x_goal, y_goal)
        """
        super().__init__(
            "evader",
            {
                "id": agent_id,
                "pose": pose,
                "goal_pose": goal_pose,
            },
        )

    def __str__(self):
        return f"EvaderState(Pos:{self.pose} | Goal:{self.goal_pose})"

    def __repr__(self):
        return self.__str__()

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def goal_pose(self):
        return self.attributes["goal_pose"]

    @property
    def set_goal_pose(self, pose):
        self.attributes["goal_pose"] = pose


class GridWorldState(pomdp_py.OOState):
    def __init__(self, evader_state, obstacles):
        """
        Environment containing evader and obstacles.
        - evader_state: EvaderState instance
        - obstacles: dict {id: ObstacleState}
        """
        object_states = {"evader": evader_state, **obstacles}
        super().__init__(object_states)

    def obstacle_at(self, pos):
        """Check if obstacle occupies position pos=(x,y)."""
        obstacles = self.obstacles
        logging.info(f"[Checking obstacle at {pos}. Obstacles present: {obstacles}")
        return any(obs.pose == pos for obs in self.obstacles.values())

    def within_bounds(self, pos, grid_size):
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < grid_size[0] and 0 <= y < grid_size[1]

    @property
    def evader(self):
        return self.object_states["evader"]

    @property
    def obstacles(self):
        return {obj_id: obj for obj_id, obj in self.object_states.items() if obj.objclass == "obstacle"}

    def __str__(self):
        obs_poses = {oid: obs.pose for oid, obs in self.obstacles.items()}
        return f"GridWorldState(Evader:{self.evader.pose}, Obstacles:{obs_poses})"

    def __repr__(self):
        return self.__str__()
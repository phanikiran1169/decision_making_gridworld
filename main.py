import pomdp_py

from description.agent import GridWorldAgent
from description.action import ALL_MOTION_ACTIONS
from env.environment import GridWorldEnvironment

class GridWorldPOMDP(pomdp_py.OOPOMDP):
    """
    Defines the GridWorld Pursuit POMDP by combining the agent and environment.
    """
    def __init__(self, grid_size, init_evader_pose, goal_pose, obstacle_prior=None):
        agent = GridWorldAgent(grid_size, init_evader_pose, goal_pose, obstacle_prior)
        env_init_state = agent.belief.mpe()  # Initialize environment to most probable state
        env = GridWorldEnvironment(grid_size, env_init_state)
        super().__init__(agent, env)


def simulate(problem, max_steps=100, planning_time=0.5, visualize=False):
    """
    Runs the simulation loop for the GridWorld POMDP.

    Args:
        problem: GridWorldPOMDP instance
        max_steps: Maximum number of steps for simulation
        planning_time: Time allowed for planning each step
        visualize: Whether to visualize each step (optional)
    """
    planner = pomdp_py.POUCT(
        max_depth=20,
        discount_factor=0.95,
        planning_time=planning_time,
        exploration_const=100,
        rollout_policy=pomdp_py.RandomRollout(action_space=ALL_MOTION_ACTIONS)
    )

    total_reward = 0
    for step in range(max_steps):
        action = planner.plan(problem.agent)
        next_state, reward = problem.env.state_transition(action, execute=True)
        observation = problem.env.provide_observation(action)
        problem.agent.update_belief(action, observation)

        total_reward += reward

        print(f"Step {step+1}:")
        print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}\n")

        if problem.env.in_terminal_state():
            print("Goal reached! Ending simulation.")
            break

if __name__ == '__main__':
    grid_size = (10, 10)
    init_evader_pose = (0, 0)
    goal_pose = (9, 9)

    # Optional prior knowledge about obstacles (uniformly uncertain if not provided)
    obstacle_prior = None  # or specify {(x, y): probability}

    problem = GridWorldPOMDP(grid_size, init_evader_pose, goal_pose, obstacle_prior)

    simulate(problem, max_steps=50, planning_time=0.5)

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
        max_depth=1,
        discount_factor=0.95,
        planning_time=planning_time,
        exploration_const=1,
        rollout_policy=problem.agent.policy_model
    )

    total_reward = 0

    for step in range(max_steps):
        print(f"\n[STEP {step+1}] ------------------------")

        # Get valid actions ONCE before planning
        belief = problem.agent.belief
        state = belief.mpe()
        valid_actions = problem.agent.policy_model.get_all_actions(state, belief)

        if not valid_actions:
            print("[ERROR] No valid actions available! Ending simulation.")
            break

        # Plan once using POUCT
        action = planner.plan(problem.agent)

        # Execute state transition ONCE
        next_state, reward = problem.env.state_transition(action, execute=True)

        # Get observation and update belief
        observation = problem.env.provide_observation(problem.agent.observation_model, action)
        problem.agent.update_belief(action, observation)

        print(f"Action Taken: {action}")
        print(f"Observation Received: {observation}")
        print(f"Reward Gained: {reward}")

        # Check terminal condition
        if problem.env.in_terminal_state():
            print("[INFO] Goal reached! Ending simulation.")
            break


if __name__ == '__main__':
    grid_size = (3, 3)
    init_evader_pose = (0, 0)
    goal_pose = (2, 1)

    # Optional prior knowledge about obstacles (uniformly uncertain if not provided)
    obstacle_prior = {(1, 0): 0.5}  # or specify {(x, y): probability}

    problem = GridWorldPOMDP(grid_size, init_evader_pose, goal_pose, obstacle_prior=obstacle_prior)

    simulate(problem, max_steps=1, planning_time=0.5)

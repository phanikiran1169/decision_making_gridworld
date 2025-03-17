import logging
import argparse
import colorlog

import pomdp_py
from domain.state import RobotState, ObjectState, MosOOState
from env.environment import MosEnvironment
from model.sensor_model import SimpleCamera
from agent_definition.agent import MosAgent

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run GridWorld POMDP simulation.")
parser.add_argument("--enable-logs", action="store_true", help="Enable logging output")
parser.add_argument(
    "--log-level",
    type=str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
    help="Set logging level (default: WARNING)",
)
args = parser.parse_args()

# Define color logging formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
)

# Configure logging based on flag
logger = logging.getLogger()
console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if args.enable_logs:
    # Enable logging with INFO level
    logger.setLevel(getattr(logging, args.log_level))
else:
    # Disable all logs by default
    logging.disable(logging.INFO)

class GridWorldPOMDP(pomdp_py.OOPOMDP):
    """
    Defines the GridWorld Pursuit POMDP by combining the agent and environment.
    """
    def __init__(self, 
                 grid_size, 
                 robot_id,
                 robots, 
                 sensors, 
                 objects, 
                 obstacles, 
                 prior=None,
                 sigma=0.01,
                 epsilon=1,
                 belief_rep="histogram",
                 num_particles=100):
        
        init_state = MosOOState({**objects, **robots})
        env = MosEnvironment(grid_size, init_state, sensors, obstacles=obstacles)

        prior = None
        if prior:
            prior = dict()
        else:
            prior = dict()
            for objid in env.target_objects:
                        groundtruth_pose = env.state.pose(objid)
                        prior[objid] = {groundtruth_pose: 1.0}
        
        agent = MosAgent(robot_id, 
                         env.state.object_states[robot_id], 
                         env.target_objects,
                         env.avoid_objects,
                         grid_size,
                         env.sensors[robot_id],
                         sigma=sigma,
                         epsilon=epsilon,
                         belief_rep=belief_rep,
                         prior=prior,
                         num_particles=num_particles,
                         grid_map=None)
        
        super().__init__(
            agent,
            env,
            name="MOS(%d,%d,%d)" % (env.width, env.length, len(env.target_objects)),
        )


def simulate(problem, max_steps=100, planning_time=0.5, max_time=120, visualize=False):
    """
    Runs the simulation loop for the GridWorld POMDP.

    Args:
        problem: GridWorldPOMDP instance
        max_steps: Maximum number of steps for simulation
        planning_time: Time allowed for planning each step
        visualize: Whether to visualize each step (optional)
    """
    planner = pomdp_py.POUCT(
        max_depth=10,
        discount_factor=0.99,
        planning_time=planning_time,
        exploration_const=100,
        rollout_policy=problem.agent.policy_model
    )

    total_reward = 0
    robot_id = problem.agent.robot_id

    for step in range(max_steps):
        logging.info(f"\n[STEP {step+1}] ------------------------")

        # Plan once using POUCT
        action = planner.plan(problem.agent)

        # Execute state transition ONCE
        reward = problem.env.state_transition(action, execute=True, robot_id=robot_id)
        total_reward += reward

        # Get observation and update belief
        logging.info(f"Get observation and update current belief")
        observation = problem.env.provide_observation(problem.agent.observation_model, action)
        problem.agent.clear_history()
        problem.agent.update_history(action, observation)
        planner.update(problem.agent, action, observation)

        logging.info(f"Action Taken: {action}")
        logging.info(f"Observation Received: {observation}")
        logging.info(f"Reward Gained: {reward}")
        logging.info(f"Total Reward: {total_reward}")
        logging.info(f"Current state - {problem.env.cur_state}")

        logging.info(f"Robot pose - {problem.env.state.object_states[robot_id].pose}")
        logging.info(f"Target pose - {problem.env.state.object_states[100].pose}")
        
        # Check terminal condition
        if (
            problem.env.state.object_states[robot_id].pose
            == problem.env.state.object_states[100].pose
        ):
            logging.info("Goal reached! Ending simulation.")
            break


if __name__ == '__main__':
    grid_size = (6, 6)
    
    evader_pose = (0, 0)
    evader_id = 0

    chaser_pose = (4, 0)
    chaser_id = 1

    robots = dict()
    robots[evader_id] = RobotState(evader_id, evader_pose, (), None)
    # robots[chaser_id] = RobotState(chaser_id, chaser_pose, (), None)
    
    sensors = dict()
    sensors[evader_id] = SimpleCamera(evader_id, grid_size=grid_size)
    # sensors[chaser_id] = SimpleCamera(chaser_id, grid_size=grid_size)
    
    evader_target_pose = (2, 5)

    objects = dict()
    objects = {
        1000: ObjectState(1000, "obstacle", (1, 0)),
        1001: ObjectState(1001, "obstacle", (2, 0)),
        1002: ObjectState(1002, "obstacle", (2, 2)),
        100: ObjectState(100, "target", evader_target_pose),
        chaser_id: ObjectState(chaser_id, "avoid", chaser_pose)
    }

    obstacles = dict()
    obstacles = {
        1000: ObjectState(1000, "obstacle", (1, 0)),
        1001: ObjectState(1001, "obstacle", (2, 0)),
        1002: ObjectState(1002, "obstacle", (2, 2)),
    }
    obstacles = set(obstacles)


    prior = None

    problem = GridWorldPOMDP(grid_size, evader_id, robots, sensors, objects, obstacles, prior)

    simulate(problem, max_steps=25, planning_time=0.5)

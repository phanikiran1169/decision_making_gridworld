import logging
import argparse
import colorlog
import json
import csv
import os
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
logger.addHandler(console_handler)

if args.enable_logs:
    logger.setLevel(getattr(logging, args.log_level))
else:
    logging.disable(logging.INFO)


def load_environment_from_json(json_file):
    """Loads environment details from a JSON file."""
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        # Check that all necessary keys are in the loaded JSON
        required_keys = ['grid_size', 'evader', 'pursuer', 'target', 'obstacles']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key: {key} in JSON file.")

        grid_size = (data['grid_size']['rows'], data['grid_size']['cols'])
        evader_pose = tuple(data['evader'])
        pursuer_pose = tuple(data['pursuer'])
        target_pose = tuple(data['target'])
        obstacles = {i + 1000: ObjectState(i + 1000, "obstacle", tuple(obs)) for i, obs in enumerate(data['obstacles'])}

        return grid_size, evader_pose, pursuer_pose, target_pose, obstacles

    except FileNotFoundError:
        logging.error(f"File {json_file} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the file: {json_file}. Please check the file format.")
        raise
    except KeyError as e:
        logging.error(f"Missing key in JSON file: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the environment: {e}")
        raise


class GridWorldPOMDP(pomdp_py.OOPOMDP):
    """
    Defines the GridWorld Pursuit POMDP by combining the agent and environment.
    """
    def __init__(self, grid_size, evader_pose, pursuer_pose, target_pose, obstacles):
        # Define robot and object IDs here
        self.evader_id = 1  # Evader ID
        self.pursuer_id = 2  # Pursuer ID
        self.target_id = 100  # Target ID

        # Define robots (evader is a robot)
        self.robots = {self.evader_id: RobotState(self.evader_id, evader_pose, (), None)}

        # Define pursuer (avoid type object)
        self.pursuer = {self.pursuer_id: ObjectState(self.pursuer_id, "avoid", pursuer_pose)}

        # Define the target object
        self.target = {self.target_id: ObjectState(self.target_id, "target", target_pose)}

        # Define the obstacles
        self.obstacles = obstacles

        # Initialize sensors
        self.sensors = {self.evader_id: SimpleCamera(self.evader_id, grid_size=grid_size)}

        # Initialize environment
        init_state = MosOOState({**self.robots, **self.pursuer, **self.target, **self.obstacles})
        env = MosEnvironment(grid_size, init_state, self.sensors, obstacles=set(obstacles.values()))

        # Create the prior belief model
        prior = None
        if prior:
            prior = dict()
        else:
            prior = dict()
            for objid in env.target_objects:
                groundtruth_pose = env.state.pose(objid)
                prior[objid] = {groundtruth_pose: 1.0}
        
        # Initialize the agent
        agent = MosAgent(self.evader_id, 
                         env.state.object_states[self.evader_id], 
                         env.target_objects,
                         env.avoid_objects,
                         grid_size,
                         env.sensors[self.evader_id],
                         sigma=0.01,
                         epsilon=1,
                         belief_rep="histogram",
                         prior=prior,
                         num_particles=100,
                         grid_map=None)
        
        super().__init__(
            agent,
            env,
            name="MOS(%d,%d,%d)" % (env.width, env.length, len(env.target_objects)),
        )


def simulate(problem, max_steps=100, planning_time=0.5, max_time=120, visualize=False, results_folder="results"):
    """
    Runs the simulation loop for the GridWorld POMDP.

    Args:
        problem: GridWorldPOMDP instance
        max_steps: Maximum number of steps for simulation
        planning_time: Time allowed for planning each step
        visualize: Whether to visualize each step (optional)
        results_folder: Folder to store the results
    """
    # Ensure the results folder exists
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Prepare the results list to store poses at each step
    results_file = os.path.join(results_folder, "simulation_results.csv")
    
    # Define CSV header
    header = [
        "Step", 
        "Evader X", "Evader Y", 
        "Pursuer X", "Pursuer Y", 
        "Target X", "Target Y"
    ]
    
    # Get the obstacles data (each as Xn, Yn)
    obstacles = []
    for idx, obs in enumerate(problem.env.state.object_states.values()):
        if isinstance(obs, ObjectState) and obs.objclass == "obstacle":
            obstacles.append(f"Obstacle{idx+1} X")
            obstacles.append(f"Obstacle{idx+1} Y")
    
    header.extend(obstacles)

    # Initialize CSV file and write header
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

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
            reward = problem.env.state_transition(action, execute=True, robot_id=robot_id, pursuer_id=problem.pursuer_id)
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

            evader_pose = problem.env.state.object_states[problem.evader_id].pose
            pursuer_pose = problem.env.state.object_states[problem.pursuer_id].pose
            target_pose = problem.env.state.object_states[problem.target_id].pose

            # Collect obstacle poses
            obstacle_poses = []
            for obs in problem.env.state.object_states.values():
                if isinstance(obs, ObjectState) and obs.objclass == "obstacle":
                    obstacle_poses.extend(obs.pose)

            # Write the data for this step
            row = [
                step + 1, 
                evader_pose[0], evader_pose[1], 
                pursuer_pose[0], pursuer_pose[1], 
                target_pose[0], target_pose[1]
            ]
            row.extend(obstacle_poses)  # Add obstacle coordinates
            writer.writerow(row)

            logging.info(f"Evader pose - {evader_pose}")
            logging.info(f"Pursuer pose - {pursuer_pose}")
            logging.info(f"Target pose - {target_pose}")

            # Check terminal condition
            if (evader_pose == target_pose):
                logging.info("Goal reached! Ending simulation.")
                break
            if (pursuer_pose == evader_pose):
                logging.info("Evader got caught! Ending simulation.")
                break

    logging.info(f"Simulation results saved to {results_file}")


if __name__ == '__main__':
    # Load the environment from the JSON file
    grid_size, evader_pose, pursuer_pose, target_pose, obstacles = load_environment_from_json('env/environment.json')
    logging.debug(f"grid_size - {grid_size}")
    logging.debug(f"evader_pose - {evader_pose}")
    logging.debug(f"pursuer_pose - {pursuer_pose}")
    logging.debug(f"target_pose - {target_pose}")
    logging.debug(f"obstacles - {obstacles}")
    
    problem = GridWorldPOMDP(grid_size, evader_pose, pursuer_pose, target_pose, obstacles)

    simulate(problem, max_steps=25, planning_time=0.5)

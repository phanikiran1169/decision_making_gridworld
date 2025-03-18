import logging
import argparse
import colorlog
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

def load_environment_from_csv(csv_file):
    """Loads environment details from a CSV file."""
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header row

            # Read the data row
            row = next(reader)

            # Extract grid size (width and length)
            grid_size = (int(row[0]), int(row[1]))

            # Extract positions for evader, pursuer, and target
            evader_pose = (int(row[2]), int(row[3]))
            pursuer_pose = (int(row[4]), int(row[5]))
            target_pose = (int(row[6]), int(row[7]))

            # Extract obstacles dynamically
            obstacles = {}
            obstacle_index = 1000  # Start at 1000 for "Obstacle1 X", "Obstacle1 Y", ...
            for i in range(8, len(row), 2):
                if row[i] != '' and row[i+1] != '':
                    obstacles[obstacle_index] = ObjectState(obstacle_index, "obstacle", (int(row[i]), int(row[i+1])))
                    obstacle_index += 1

            return grid_size, evader_pose, pursuer_pose, target_pose, obstacles

    except FileNotFoundError:
        logging.error(f"File {csv_file} not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the environment from CSV: {e}")
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

def simulate(problem, run_number, max_steps=100, planning_time=0.5, max_time=120, visualize=False, results_folder="results/gridworld_1"):
    """
    Runs the simulation loop for the GridWorld POMDP.

    Args:
        problem: GridWorldPOMDP instance
        run_number: The current run number (used for folder naming)
        max_steps: Maximum number of steps for simulation
        planning_time: Time allowed for planning each step
        visualize: Whether to visualize each step (optional)
        results_folder: Folder to store the results
    """
    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # Prepare the results file in this run's folder
    results_file = os.path.join(results_folder, f"simulation_results_run_{run_number}.csv")
    
    # Define CSV header
    header = [
        "Grid Width", "Grid Length",  # Grid size in the first two columns
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
    
    # Add pursuit success as a column to track each step
    header.append("Pursuit Success")

    # Initialize CSV file and write header
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # Get initial environment settings and save them
        grid_size, evader_pose, pursuer_pose, target_pose, obstacles = load_environment_from_csv('env/gridworld_1.csv')

        # Write grid size and initial positions in the first row
        initial_row = [
            grid_size[0], grid_size[1],
            evader_pose[0], evader_pose[1], 
            pursuer_pose[0], pursuer_pose[1], 
            target_pose[0], target_pose[1]
        ]
        
        # Add initial obstacle positions
        for obs in obstacles.values():
            initial_row.extend(obs.pose)
        initial_row.append(False)
        
        writer.writerow(initial_row)

        # Reinitialize the environment and agent for this simulation run
        problem = GridWorldPOMDP(grid_size, evader_pose, pursuer_pose, target_pose, obstacles)

        planner = pomdp_py.POUCT(
            max_depth=10,
            discount_factor=0.99,
            planning_time=planning_time,
            exploration_const=100,
            rollout_policy=problem.agent.policy_model
        )

        total_reward = 0
        robot_id = problem.agent.robot_id
        pursuit_success = False

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
            
            logging.info(f"Evader pose - {evader_pose}")
            logging.info(f"Pursuer pose - {pursuer_pose}")
            logging.info(f"Target pose - {target_pose}")

            # Check terminal condition
            if (evader_pose == target_pose):
                logging.info("Goal reached! Ending simulation.")
                pursuit_success = False
                break
            if (pursuer_pose == evader_pose):
                logging.info("Evader got caught! Ending simulation.")
                pursuit_success = True
                break

            # Write the data for this step
            row = [
                grid_size[0], grid_size[1],
                evader_pose[0], evader_pose[1], 
                pursuer_pose[0], pursuer_pose[1], 
                target_pose[0], target_pose[1]
            ]
            row.extend(obstacle_poses)  # Add obstacle coordinates
            row.append(pursuit_success)  # Add pursuit success status at this step
            
            writer.writerow(row)
        
        # After the loop, ensure the last step is saved if the loop breaks early
        if step < max_steps - 1:  # If the loop breaks early, we need to save the last state
            row = [
                grid_size[0], grid_size[1],
                evader_pose[0], evader_pose[1], 
                pursuer_pose[0], pursuer_pose[1], 
                target_pose[0], target_pose[1]
            ]
            row.extend(obstacle_poses)  # Add obstacle coordinates
            row.append(pursuit_success)  # Add pursuit success status at this step
            
            writer.writerow(row)

        logging.info(f"Simulation results saved to {results_file}")

if __name__ == '__main__':
    # Load the environment from the CSV file
    grid_size, evader_pose, pursuer_pose, target_pose, obstacles = load_environment_from_csv('env/gridworld_1.csv')
    logging.debug(f"grid_size - {grid_size}")
    logging.debug(f"evader_pose - {evader_pose}")
    logging.debug(f"pursuer_pose - {pursuer_pose}")
    logging.debug(f"target_pose - {target_pose}")
    logging.debug(f"obstacles - {obstacles}")
    
    problem = GridWorldPOMDP(grid_size, evader_pose, pursuer_pose, target_pose, obstacles)

    # Run the simulation 100 times and save results in the gridworld_1 folder
    for run_number in range(1, 101):
        simulate(problem, run_number)
import logging
import pygame
import csv
import numpy as np
import os
from colors import *  # Import your color definitions
from PIL import Image, ImageOps  # For saving and mirroring GIF

class GridGifRenderer:
    """
    Generates a GIF animation of the simulation directly from a CSV file 
    without displaying it in Pygame.
    """

    def __init__(self, env_file, output_gif="simulation.gif"):
        """
        Initializes the renderer.

        Args:
            env_file (str): CSV file containing simulation steps.
            output_gif (str): Name of the output GIF file.
        """
        if not env_file:
            raise FileNotFoundError("ERROR: No CSV file provided for simulation.")

        pygame.init()
        self.env_file = env_file
        self.output_gif = output_gif
        self.steps = []  # Store all simulation steps
        self.frames = []  # Store frames for GIF

        self.load_environment()  # Load the environment from CSV

        # Grid and display settings (initialized AFTER reading CSV)
        self.cell_size = 40  # Keep cell size fixed
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size

        # Create an off-screen Pygame surface (no display window)
        self.screen = pygame.Surface((self.width, self.height))

    def load_environment(self):
        """
        Loads the environment data from a CSV file.

        Raises:
            FileNotFoundError: If the CSV file is not found.
            ValueError: If the CSV format is incorrect.
        """
        try:
            with open(self.env_file, "r") as file:
                reader = csv.reader(file)
                header = next(reader)  # Read the header row

                # Validate header format
                if not header or len(header) < 8:  # We need at least 8 columns now
                    raise ValueError("ERROR: CSV file format is incorrect. Expected at least 8 columns.")

                self.steps = []  # Clear previous steps

                for row in reader:
                    if len(row) < 8:  # If row doesn't have enough columns, skip it
                        continue
                    
                    # Read the data from the row
                    step = {
                        "evader": (int(row[2]), int(row[3])),  # Evader position (updated index for new CSV format)
                        "pursuer": (int(row[4]), int(row[5])),  # Pursuer position
                        "target": (int(row[6]), int(row[7])),  # Target position
                        "obstacles": [(int(row[i]), int(row[i+1])) for i in range(8, len(row)-1, 2)],  # Obstacles
                        "pursuit_success": row[-1].strip() == "True"  # Read the Pursuit Success column
                    }
                    self.steps.append(step)

                if not self.steps:
                    raise ValueError("ERROR: CSV file does not contain valid simulation data.")

                # Infer grid size dynamically based on max coordinates
                all_x = [step["evader"][0] for step in self.steps] + \
                        [step["pursuer"][0] for step in self.steps] + \
                        [step["target"][0] for step in self.steps if step["target"]] + \
                        [obs[0] for step in self.steps for obs in step["obstacles"]]

                all_y = [step["evader"][1] for step in self.steps] + \
                        [step["pursuer"][1] for step in self.steps] + \
                        [step["target"][1] for step in self.steps if step["target"]] + \
                        [obs[1] for step in self.steps for obs in step["obstacles"]]

                self.rows = max(all_x) + 1 if all_x else 7
                self.cols = max(all_y) + 1 if all_y else 7

        except FileNotFoundError:
            raise FileNotFoundError(f"ERROR: CSV file '{self.env_file}' not found.")
        except ValueError as e:
            raise ValueError(str(e))

    def render_frame(self, step):
        """
        Renders a single frame and stores it in the frame buffer.

        Args:
            step (dict): Contains evader, pursuer, target, and obstacle positions.
        """
        self.screen.fill(WHITE)
        
        # Draw grid
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Draw obstacles (black squares)
        for obs in step["obstacles"]:
            obs_rect = pygame.Rect(obs[1] * self.cell_size, obs[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, BLACK, obs_rect)

        # Draw evader (blue circle)
        evader_x, evader_y = step["evader"][1] * self.cell_size, step["evader"][0] * self.cell_size
        pygame.draw.circle(self.screen, BLUE, (evader_x + self.cell_size // 2, evader_y + self.cell_size // 2), self.cell_size // 2 - 5)

        # Draw pursuer (red circle)
        pursuer_x, pursuer_y = step["pursuer"][1] * self.cell_size, step["pursuer"][0] * self.cell_size
        pygame.draw.circle(self.screen, RED, (pursuer_x + self.cell_size // 2, pursuer_y + self.cell_size // 2), self.cell_size // 2 - 5)

        # Draw target (light green circle)
        if step["target"]:
            target_x, target_y = step["target"][1] * self.cell_size, step["target"][0] * self.cell_size
            pygame.draw.circle(self.screen, LIGHTGREEN, (target_x + self.cell_size // 2, target_y + self.cell_size // 2), self.cell_size // 2 - 5)

        # Capture frame for GIF (FIX: Flip horizontally before saving)
        frame = pygame.surfarray.array3d(self.screen)
        flipped_frame = np.fliplr(frame)  # Flip the image horizontally
        self.frames.append(np.rot90(flipped_frame, k=3))  # Rotate correctly

    def save_gif(self):
        """
        Saves the generated frames as a GIF file.
        """
        if not self.frames:
            logging.warning("No frames to save.")
            return

        pil_images = [Image.fromarray(frame) for frame in self.frames]
        
        # Apply horizontal flip to all frames before saving
        pil_images = [ImageOps.mirror(img) for img in pil_images]
        

        pil_images[0].save(self.output_gif, save_all=True, append_images=pil_images[1:], optimize=True, duration=250, loop=0)
        logging.info(f"Simulation saved as {self.output_gif}")

    def generate_gif(self):
        """
        Generates a GIF of the simulation by rendering all frames off-screen.
        """
        logging.info("Generating GIF from simulation data...")

        for step in self.steps:
            self.render_frame(step)  # Render each step
        
        self.save_gif()  # Save the generated frames as a GIF

        logging.info("GIF generation complete.")

if __name__ == "__main__":
    # Folder containing the CSV files for all runs
    results_folder = "gridworld_3"

    # Loop through all CSV files in the folder and generate GIFs
    for filename in os.listdir(results_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(results_folder, filename)
            output_gif = os.path.join(results_folder, f"{filename.replace('.csv', '.gif')}")

            if os.path.exists(file_path):
                logging.info(f"Generating GIF for {file_path}...")
                renderer = GridGifRenderer(file_path, output_gif)
                renderer.generate_gif()
            else:
                logging.warning(f"File {file_path} does not exist, skipping.")
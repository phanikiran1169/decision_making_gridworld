import logging
import pygame
import sys
import csv
import numpy as np
from colors import *  # Import your color definitions
from PIL import Image  # For saving GIF

class GridEnv:
    """
    Grid-based environment simulation for two agents (Evader & Pursuer).
    
    The environment is loaded from a CSV file and visualized in Pygame.
    Clicking the "Save Environment" button saves the animation as a GIF.
    """

    def __init__(self, env_file):
        """
        Initializes the grid world environment.

        Args:
            env_file (str): CSV file containing environment steps.
        
        Raises:
            FileNotFoundError: If no CSV file is provided or the file does not exist.
        """
        if not env_file:
            raise FileNotFoundError("ERROR: No CSV file provided for environment initialization.")

        pygame.init()
        self.env_file = env_file
        self.steps = []  # Store all simulation steps
        self.frames = []  # Store frames for GIF creation

        self.load_environment()  # Load the environment from CSV

        # Grid and display settings (initialized AFTER reading CSV)
        self.cell_size = 40  # Keep cell size fixed
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size + 50  # Space for the Save button
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid World Simulation")

        # Save Button UI
        self.save_button_rect = pygame.Rect(self.width // 2 - 60, self.height - 40, 120, 30)

        # Frame rate controller
        self.clock = pygame.time.Clock()

    def load_environment(self):
        """
        Loads the environment from a CSV file.

        The CSV file contains:
        - First row: Column headers
        - Subsequent rows: Time steps with positions of evader, pursuer, target, and obstacles.

        Raises:
            FileNotFoundError: If the CSV file is not found.
            ValueError: If the CSV format is incorrect.
        """
        try:
            with open(self.env_file, "r") as file:
                reader = csv.reader(file)
                header = next(reader)  # Read the header row
                
                # Validate header format
                if not header or len(header) < 7:
                    raise ValueError("ERROR: CSV file format is incorrect. Expected at least 7 columns.")

                self.steps = []  # Clear previous steps

                for row in reader:
                    if len(row) < 7:
                        continue  # Ignore malformed rows
                    
                    step = {
                        "evader": (int(row[1]), int(row[2])),
                        "pursuer": (int(row[3]), int(row[4])),
                        "target": (int(row[5]), int(row[6])),
                        "obstacles": [(int(row[i]), int(row[i+1])) for i in range(7, len(row), 2)]
                    }
                    self.steps.append(step)

                # If no valid steps, raise an error
                if not self.steps:
                    raise ValueError("ERROR: CSV file does not contain valid environment data.")

                # Infer grid size dynamically based on max coordinates
                all_x = [step["evader"][0] for step in self.steps] + \
                        [step["pursuer"][0] for step in self.steps] + \
                        [step["target"][0] for step in self.steps if step["target"]] + \
                        [obs[0] for step in self.steps for obs in step["obstacles"]]

                all_y = [step["evader"][1] for step in self.steps] + \
                        [step["pursuer"][1] for step in self.steps] + \
                        [step["target"][1] for step in self.steps if step["target"]] + \
                        [obs[1] for step in self.steps for obs in step["obstacles"]]

                self.rows = max(all_x) + 1 if all_x else 7  # Default to 7x7 if empty
                self.cols = max(all_y) + 1 if all_y else 7

                # Initialize first step
                self.current_step = 0
                self.update_state()

        except FileNotFoundError:
            raise FileNotFoundError(f"ERROR: CSV file '{self.env_file}' not found.")
        except ValueError as e:
            raise ValueError(str(e))

    def update_state(self):
        """
        Updates the simulation state based on the current step.
        Ensures that evader, pursuer, target, and obstacles always exist.
        """
        step = self.steps[self.current_step]
        self.evader = step["evader"]
        self.pursuer = step["pursuer"]
        self.target = step["target"]
        self.obstacles = set(step["obstacles"])

    def render(self):
        """
        Renders the grid world including agents, obstacles, and the target.
        """
        self.screen.fill(WHITE)
        
        # Draw grid
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Draw obstacles (black squares)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs[1] * self.cell_size, obs[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, BLACK, obs_rect)

        # Draw evader (blue circle)
        evader_x, evader_y = self.evader[1] * self.cell_size, self.evader[0] * self.cell_size
        pygame.draw.circle(self.screen, BLUE, (evader_x + self.cell_size // 2, evader_y + self.cell_size // 2), self.cell_size // 2 - 5)

        # Draw pursuer (red circle)
        pursuer_x, pursuer_y = self.pursuer[1] * self.cell_size, self.pursuer[0] * self.cell_size
        pygame.draw.circle(self.screen, RED, (pursuer_x + self.cell_size // 2, pursuer_y + self.cell_size // 2), self.cell_size // 2 - 5)

        # Draw target (light green circle)
        if self.target:
            target_x, target_y = self.target[1] * self.cell_size, self.target[0] * self.cell_size
            pygame.draw.circle(self.screen, LIGHTGREEN, (target_x + self.cell_size // 2, target_y + self.cell_size // 2), self.cell_size // 2 - 5)

        # Draw Save Button
        pygame.draw.rect(self.screen, GREEN, self.save_button_rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, self.save_button_rect, width=2, border_radius=10)
        font = pygame.font.Font(None, 18)
        text = font.render("Save Environment", True, BLACK)
        text_rect = text.get_rect(center=self.save_button_rect.center)
        self.screen.blit(text, text_rect)

        # Capture frame for GIF
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(np.rot90(frame, k=3))

        pygame.display.flip()

    def save_gif(self):
        """
        Saves recorded frames as a GIF file.
        """
        if not self.frames:
            logging.warning("No frames to save.")
            return

        pil_images = [Image.fromarray(frame) for frame in self.frames]
        pil_images[0].save("simulation.gif", save_all=True, append_images=pil_images[1:], optimize=True, duration=100, loop=0)
        logging.info("Simulation saved as simulation.gif")

    def run(self):
        """
        Runs the step-based simulation and updates the display.

        Clicking "Save Environment" will save the simulation as a GIF.
        """
        running = True

        while running:
            self.clock.tick(1)  # 1 step per second

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and self.save_button_rect.collidepoint(event.pos):
                    self.save_gif()

            self.update_state()
            self.render()
            self.current_step = (self.current_step + 1) % len(self.steps)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    env = GridEnv(env_file="simulation_results.csv")
    env.run()

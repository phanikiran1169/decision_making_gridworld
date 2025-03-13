import logging
import pygame
import sys
import json
from colors import *

class GridEnv:
    """
    @brief Class representing a grid-based environment for two agents (Evader & Pursuer)
    
    This class allows agents to move within a grid, avoid obstacles, and interact with a path planner
    It supports placing/removing obstacles, saving/loading the environment, and running simulations
    """

    def __init__(self, rows=15, cols=15, cell_size=40, env_file="environment.json"):
        """
        @brief Initializes the environment with customizable grid size
        
        @param rows Number of rows in the grid (default: 15)
        @param cols Number of columns in the grid (default: 15)
        @param cell_size Size of each cell in pixels (default: 40)
        @param env_file File to save/load environment settings
        """
        pygame.init()
        
        # Grid parameters
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size + 50  # Extra space for Save button
        
        # Create game window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid World")

        # File to save/load environment
        self.env_file = env_file

        # Load environment settings
        self.load_environment()

        # Save Environment Button
        self.save_button_rect = pygame.Rect(self.width // 2 - 60, self.height - 40, 120, 30)

        # Frame rate controller
        self.clock = pygame.time.Clock()

    def load_environment(self):
        """
        @brief Loads the environment (obstacles, agent positions) from a JSON file.
        """
        try:
            with open(self.env_file, "r") as file:
                data = json.load(file)
                self.obstacles = set(tuple(obstacle) for obstacle in data["obstacles"])
                self.evader = data["evader"]
                self.pursuer = data["pursuer"]
        except FileNotFoundError:
            self.obstacles = set()
            self.evader = [0, 0]  # Default evader position
            self.pursuer = [self.rows - 1, self.cols - 1]  # Default pursuer position

    def save_environment(self):
        """
        @brief Saves the current environment (obstacles, agent positions) to a JSON file
        """
        data = {
            "obstacles": list(self.obstacles),
            "evader": self.evader,
            "pursuer": self.pursuer
        }
        with open(self.env_file, "w") as file:
            json.dump(data, file)

    def is_valid_move(self, new_pos):
        """
        @brief Checks if the move is valid
        
        @param new_pos The (row, col) position to check
        
        @return True if the move is within bounds and not an obstacle, False otherwise
        """
        row, col = new_pos
        return (0 <= row < self.rows and 0 <= col < self.cols and new_pos not in self.obstacles)

    def move_agent(self, agent, direction):
        """
        @brief Moves the specified agent in the given direction

        @param agent The agent to move ('evader' or 'pursuer')
        @param direction The direction to move ('north', 'south', 'west', 'east')
        """
        if agent == "evader":
            current_pos = self.evader
        elif agent == "pursuer":
            current_pos = self.pursuer
        else:
            return  # Invalid agent name

        new_pos = list(current_pos)

        if direction == "north":
            new_pos[0] -= 1
        elif direction == "south":
            new_pos[0] += 1
        elif direction == "west":
            new_pos[1] -= 1
        elif direction == "east":
            new_pos[1] += 1

        if self.is_valid_move(tuple(new_pos)):
            if agent == "evader":
                self.evader = new_pos
            else:
                self.pursuer = new_pos

    def toggle_obstacle(self, pos):
        """
        @brief Adds or removes an obstacle at the clicked grid position
        
        @param pos (x, y) screen coordinates of the mouse click
        """
        col = pos[0] // self.cell_size
        row = pos[1] // self.cell_size

        if 0 <= row < self.rows and 0 <= col < self.cols:
            if (row, col) in self.obstacles:
                self.obstacles.remove((row, col))  # Remove obstacle
            else:
                self.obstacles.add((row, col))  # Add obstacle

    def render(self):
        """
        @brief Renders the grid environment, including agents, obstacles, and the goal
        """
        self.screen.fill(WHITE)
        
        # Draw the grid
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Draw obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs[1] * self.cell_size, obs[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, BLACK, obs_rect)

        # Draw the evader
        evader_x = self.evader[1] * self.cell_size + self.cell_size // 2
        evader_y = self.evader[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, BLUE, (evader_x, evader_y), self.cell_size // 2 - 5)

        # Draw the pursuer
        pursuer_x = self.pursuer[1] * self.cell_size + self.cell_size // 2
        pursuer_y = self.pursuer[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, RED, (pursuer_x, pursuer_y), self.cell_size // 2 - 5)

        # Draw the Save Environment button
        pygame.draw.rect(self.screen, GREEN, self.save_button_rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, self.save_button_rect, width=2, border_radius=10) 

        # Button text
        font = pygame.font.Font(None, 18)
        text = font.render("Save Environment", True, BLACK)
        text_rect = text.get_rect(center=self.save_button_rect.center)
        self.screen.blit(text, text_rect)

        # Update display
        pygame.display.flip()


    def run(self):
        """
        @brief Runs the environment with two modes:
            - Obstacle Editing Mode (click to place/remove obstacles)
            - Agent Movement Mode (move agents)
            - Press "M" to switch between modes.
        """
        running = True
        edit_mode = True  # Start in obstacle editing mode

        while running:
            self.clock.tick(10)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:  # Press "M" to switch modes
                        edit_mode = not edit_mode
                        logging.info("Switched Mode:", "Obstacle Editing" if edit_mode else "Agent Movement")

                if edit_mode:  # Obstacle Editing Mode
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if self.save_button_rect.collidepoint(event.pos):
                            self.save_environment()
                            logging.info("Environment saved successfully!")
                        else:
                            self.toggle_obstacle(event.pos)
                else:  # Agent Movement Mode
                    if event.type == pygame.KEYDOWN:
                        key_map = {
                            pygame.K_w: "north", pygame.K_s: "south",
                            pygame.K_a: "west", pygame.K_d: "east",
                            pygame.K_UP: "north", pygame.K_DOWN: "south",
                            pygame.K_LEFT: "west", pygame.K_RIGHT: "east"
                        }
                        if event.key in {pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d}:
                            self.move_agent("evader", key_map[event.key])
                        elif event.key in {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT}:
                            self.move_agent("pursuer", key_map[event.key])

            self.render()

        pygame.quit()
        sys.exit()


# main
if __name__ == "__main__":
    env = GridEnv()
    env.run()

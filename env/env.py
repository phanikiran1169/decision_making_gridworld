import pygame
import sys
from colors import WHITE, BLACK, BLUE, RED, GREEN, GRAY  # Import colors

class RobotGridEnv:
    """
    @brief Class representing a grid-based environment for two robots (Evader & Pursuer).
    
    This class allows the Evader and Pursuer to move within a grid, avoiding obstacles.
    A path planner can interact with the environment via method calls.
    """

    def __init__(self, rows=15, cols=15, cell_size=40):
        """
        @brief Initializes the environment with customizable grid size.
        
        @param rows Number of rows in the grid (default: 15).
        @param cols Number of columns in the grid (default: 15).
        @param cell_size Size of each cell in pixels (default: 40).
        """
        pygame.init()
        
        # Grid parameters
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size
        
        # Create game window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Robot Grid Environment")

        # Obstacles
        self.obstacles = {
            (3, 3), (3, 4), (3, 5),
            (7, 1), (7, 2), (7, 3),
            (10, 10), (10, 11), (11, 10),
            (5, 8), (6, 8), (7, 8)
        }

        # Robots and goal positions
        self.evader = [0, 0]        # Evader (Blue) starts at top-left
        self.pursuer = [self.rows - 1, self.cols - 1]  # Pursuer (Red) starts at bottom-right
        self.goal = (self.rows // 2, self.cols // 2)  # Goal placed in the center

        # Frame rate controller
        self.clock = pygame.time.Clock()

    def is_valid_move(self, new_pos):
        """
        @brief Checks if the move is valid.
        
        @param new_pos The (row, col) position to check.
        
        @return True if the move is within bounds and not an obstacle, False otherwise.
        """
        row, col = new_pos
        return (0 <= row < self.rows and 0 <= col < self.cols and new_pos not in self.obstacles)

    def move_robot(self, robot, direction):
        """
        @brief Moves the specified robot in the given direction.

        @param robot The robot to move ('evader' or 'pursuer').
        @param direction The direction to move ('up', 'down', 'left', 'right').
        """
        if robot == "evader":
            current_pos = self.evader
        elif robot == "pursuer":
            current_pos = self.pursuer
        else:
            return  # Invalid robot name

        new_pos = list(current_pos)

        if direction == "up":
            new_pos[0] -= 1
        elif direction == "down":
            new_pos[0] += 1
        elif direction == "left":
            new_pos[1] -= 1
        elif direction == "right":
            new_pos[1] += 1

        if self.is_valid_move(tuple(new_pos)):
            if robot == "evader":
                self.evader = new_pos
            else:
                self.pursuer = new_pos

    def render(self):
        """
        @brief Renders the grid environment, including robots, obstacles, and the goal.
        """
        self.screen.fill(WHITE)
        
        # Draw the grid lines
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Draw obstacles
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs[1] * self.cell_size, obs[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, BLACK, obs_rect)

        # Draw the goal
        goal_rect = pygame.Rect(self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, GREEN, goal_rect)

        # Draw the evader (blue)
        evader_x = self.evader[1] * self.cell_size + self.cell_size // 2
        evader_y = self.evader[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, BLUE, (evader_x, evader_y), self.cell_size // 2 - 5)

        # Draw the pursuer (red)
        pursuer_x = self.pursuer[1] * self.cell_size + self.cell_size // 2
        pursuer_y = self.pursuer[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, RED, (pursuer_x, pursuer_y), self.cell_size // 2 - 5)

        # Update display
        pygame.display.flip()

    def reset(self):
        """
        @brief Resets the environment by repositioning robots to their initial locations.
        """
        self.evader = [0, 0]
        self.pursuer = [self.rows - 1, self.cols - 1]

    def run_manual_control(self):
        """
        @brief Runs the environment in manual mode where users can control robots using the keyboard.
        
        Arrow keys control the Evader.
        WASD controls the Pursuer.
        """
        running = True
        while running:
            self.clock.tick(10)  # Limit frame rate to 10 FPS

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.move_robot("evader", "up")
                    elif event.key == pygame.K_DOWN:
                        self.move_robot("evader", "down")
                    elif event.key == pygame.K_LEFT:
                        self.move_robot("evader", "left")
                    elif event.key == pygame.K_RIGHT:
                        self.move_robot("evader", "right")

                    if event.key == pygame.K_w:
                        self.move_robot("pursuer", "up")
                    elif event.key == pygame.K_s:
                        self.move_robot("pursuer", "down")
                    elif event.key == pygame.K_a:
                        self.move_robot("pursuer", "left")
                    elif event.key == pygame.K_d:
                        self.move_robot("pursuer", "right")

            self.render()

        pygame.quit()
        sys.exit()


# Example usage
if __name__ == "__main__":
    env = RobotGridEnv(rows=20, cols=20, cell_size=30)  # Example: custom grid size
    env.run_manual_control()

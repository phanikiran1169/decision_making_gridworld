import pygame
import sys

# Initialize Pygame
pygame.init()

# Grid parameters
rows = 15
cols = 15
cell_size = 40
width = cols * cell_size
height = rows * cell_size

# Create the game window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Grid Simulation")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Define obstacles as a list of (row, col) tuples
obstacles = [
    (3, 3), (3, 4), (3, 5),
    (7, 1), (7, 2), (7, 3),
    (10, 10), (10, 11), (11, 10),
    (5, 8), (6, 8), (7, 8)
]

# Define initial positions of robots and goal
blue_robot = [0, 0]
red_robot = [14, 14]
goal = (7, 7)

# Set up the frame rate
clock = pygame.time.Clock()


def is_valid_move(new_row, new_col):
    """
    @brief Checks whether the new position is a valid move.
    
    @param new_row The row index of the new position.
    @param new_col The column index of the new position.
    
    @return True if the move is valid (within grid bounds and not an obstacle), False otherwise.
    """
    if 0 <= new_row < rows and 0 <= new_col < cols:
        if (new_row, new_col) not in obstacles:
            return True
    return False


# Main loop control variable
running = True

# Main game loop
while running:
    # Limit the frame rate to 10 frames per second
    clock.tick(10)
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            
            # Handle movement for the blue robot (Arrow Keys)
            new_row = blue_robot[0]
            new_col = blue_robot[1]

            if event.key == pygame.K_UP:
                new_row -= 1
            elif event.key == pygame.K_DOWN:
                new_row += 1
            elif event.key == pygame.K_LEFT:
                new_col -= 1
            elif event.key == pygame.K_RIGHT:
                new_col += 1
            
            if is_valid_move(new_row, new_col):
                blue_robot = [new_row, new_col]

            # Handle movement for the red robot (WASD Keys)
            new_row = red_robot[0]
            new_col = red_robot[1]

            if event.key == pygame.K_w:
                new_row -= 1
            elif event.key == pygame.K_s:
                new_row += 1
            elif event.key == pygame.K_a:
                new_col -= 1
            elif event.key == pygame.K_d:
                new_col += 1
            
            if is_valid_move(new_row, new_col):
                red_robot = [new_row, new_col]

    # Clear the screen
    screen.fill(WHITE)
    
    # Draw the grid lines
    for row in range(rows):
        for col in range(cols):
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, GRAY, rect, 1)
    
    # Draw obstacles
    for obs in obstacles:
        obs_rect = pygame.Rect(obs[1] * cell_size, obs[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, BLACK, obs_rect)
    
    # Draw the goal
    goal_rect = pygame.Rect(goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, GREEN, goal_rect)
    
    # Draw the blue robot
    blue_center_x = blue_robot[1] * cell_size + cell_size // 2
    blue_center_y = blue_robot[0] * cell_size + cell_size // 2
    pygame.draw.circle(screen, BLUE, (blue_center_x, blue_center_y), cell_size // 2 - 5)
    
    # Draw the red robot
    red_center_x = red_robot[1] * cell_size + cell_size // 2
    red_center_y = red_robot[0] * cell_size + cell_size // 2
    pygame.draw.circle(screen, RED, (red_center_x, red_center_y), cell_size // 2 - 5)
    
    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()

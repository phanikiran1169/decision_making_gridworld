import math
import numpy as np
from domain import MosOOObservation, ObjectObservation
from domain.action import *

class SimpleCamera:
    def __init__(self, robot_id, grid_size, min_range=0, max_range=100):
        self.robot_id = robot_id
        self.min_range = min_range
        self.max_range = max_range
        self.grid_size = grid_size

    def observe(self, robot_pose, env_state):
        objposes = {}

        # Check visibility in all directions
        directions = [NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST]

        # Initialize all objects as unseen
        for objid in env_state.object_states:
            objposes[objid] = ObjectObservation.NULL

        # Check visibility in all 8 directions
        for dx, dy in directions:
            x, y = robot_pose  # Start from the robot's position

            while True:
                x += dx  # Move step-by-step in the direction
                y += dy

                # Stop if out of bounds
                if not self.in_boundary((x, y), self.grid_size[0], self.grid_size[1]):
                    break

                # Check if (x, y) contains any object (obstacle, or target)
                for objid, obj_state in env_state.object_states.items():
                    if obj_state["pose"] == (x, y):  # Ensure correct pose access
                        objposes[objid] = (x, y)  # Record visibility
                        break  # Stop checking further in this direction (vision is blocked)

        return MosOOObservation(objposes)  # Return observations
    
    def within_range(self, robot_pose, point, env_state):
        """Checks if the point is within the sensor range and visible (not blocked by obstacles)."""

        # Extract robot and target point coordinates
        rx, ry = robot_pose
        px, py = point

        # Compute the Euclidean distance
        dist = math.sqrt((px - rx) ** 2 + (py - ry) ** 2)

        # Check if the point is within the valid sensing range
        if not (self.min_range <= dist <= self.max_range):
            return False

        # Get the list of grid points along the line (Bresenham's Line Algorithm)
        line_points = self.get_line(rx, ry, px, py)

        # Check if any obstacle blocks the vision
        for x, y in line_points:
            if (x, y) == (px, py):
                return True  # We reached the target point without being blocked
            for objid, obj_state in env_state.object_states.items():
                if obj_state["pose"] == (x, y) and obj_state.objclass.startswith("obstacle"):
                    return False  # Vision is blocked

        return True  # The point is within range and not blocked

    def get_line(self, x0, y0, x1, y1):
        """Bresenham's Line Algorithm to get all points between two coordinates."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def in_boundary(self, pose, width, length):
        """Check if pose is within the world boundaries."""
        x, y = pose
        return 0 <= x < width and 0 <= y < length
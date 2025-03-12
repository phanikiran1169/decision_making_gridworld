import pomdp_py

# Grid step size
STEP_SIZE = 1

class Action(pomdp_py.Action):
    """Base Action class."""
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Action({self.name})"
    
    def __hash__(self):
        return hash(self.name)


class MotionAction(Action):
    EAST = (STEP_SIZE, 0)
    WEST = (-STEP_SIZE, 0)
    NORTH = (0, STEP_SIZE)
    SOUTH = (0, -STEP_SIZE)

    VALID_MOTIONS = {
        "east": EAST,
        "west": WEST,
        "north": NORTH,
        "south": SOUTH,
    }

    def __init__(self, motion, distance_cost=1):
        """
        motion: tuple representing the movement in (x, y).
        """
        if motion not in MotionAction.VALID_MOTIONS.values():
            raise ValueError(f"Invalid motion: {motion}")

        motion_name = next(direction for direction, motion_val in MotionAction.VALID_MOTIONS.items() if motion_val == motion)

        super().__init__(f"move-{motion_name}")
        self.motion = motion
        self.distance_cost = distance_cost

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"MotionAction({self.name})"
    
    def __hash__(self):
        return hash(self.motion)


# Explicit instances for easier usage
MoveEast = MotionAction(MotionAction.EAST)
MoveWest = MotionAction(MotionAction.WEST)
MoveNorth = MotionAction(MotionAction.NORTH)
MoveSouth = MotionAction(MotionAction.SOUTH)

# List of all movement actions
ALL_MOTION_ACTIONS = [MoveEast, MoveWest, MoveNorth, MoveSouth]
class LookAction(Action):
    """Action to look around before moving."""
    def __init__(self):
        super().__init__("look")

class FindAction(Action):
    """Action to declare that an object has been found."""
    def __init__(self):
        super().__init__("find")
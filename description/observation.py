import pomdp_py

class CellObservation(pomdp_py.Observation):
    """Observation of a single cell: either obstacle or free."""

    FREE = "free"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"

    def __init__(self, position, status):
        """
        Args:
            position (tuple): (x,y) grid cell position.
            status (str): FREE, OBSTACLE, or UNKNOWN
        """
        if status not in {self.FREE, self.OBSTACLE, self.UNKNOWN}:
            raise ValueError("Invalid status: must be 'free', 'obstacle', or None.")
        self.position = position
        self.status = status

    def __str__(self):
        return f"CellObservation({self.position}: {self.status})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (isinstance(other, CellObservation) and
                self.position == other.position and
                self.status == other.status)

    def __hash__(self):
        return hash((self.position, self.status))


class EvaderObservation(pomdp_py.Observation):
    """Observation from the evader's viewpoint; observes cells within line-of-sight."""

    def __init__(self, observed_cells):
        """
        observed_cells: dict of {(x,y): "free"/"obstacle"}
        representing visibility results.
        """
        self.observed_cells = observed_cells

    def cell_observation(self, position):
        if position in self.observed_cells:
            return CellObservation(position, self.observed_cells[position])
        else:
            return CellObservation(position, CellObservation.UNKNOWN)

    def __str__(self):
        return f"EvaderObservation({self.observed_cells})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, EvaderObservation) and \
               self.observed_cells == other.observed_cells

    def __hash__(self):
        return hash(frozenset(self.observed_cells.items()))

    def factor(self, state):
        """Factor observation by cells for possible OO-POMDP updates."""
        return {
            pos: CellObservation(pos, status)
            for pos, status in self.observed_cells.items()
        }
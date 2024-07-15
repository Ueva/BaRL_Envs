import numpy as np

from simpleenvs.envs.discrete_rooms import DiscreteRoomEnvironment
from simpleenvs.envs.discrete_rooms.rooms import CELL_TYPES_DICT


class ExplorableWrapper(DiscreteRoomEnvironment):
    def __init__(self, *args, **kwargs):
        super(ExplorableWrapper, self).__init__(*args, **kwargs)

    def _initialise_rooms(self, room_template_file_path):
        """
        Initialises an explorable version of a gridworld envionment according to a
        given template file. Goal states are ignored, and replaced by empty cells.

        Arguments:
            room_template_file_path {string or Path} -- Path to a gridworld template file.
        """

        # Load gridworld template file.
        self.gridworld = np.loadtxt(room_template_file_path, comments="//", dtype=str)

        # Discover start and goal states.
        self.initial_states = []
        self.terminal_states = []
        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):
                if CELL_TYPES_DICT[self.gridworld[y, x]] == "start":
                    self.initial_states.append((y, x))

    def _initialise_state_space(self):
        # Create set of all valid states.
        self.state_space = set()
        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):
                if CELL_TYPES_DICT[self.gridworld[y, x]] != "wall":
                    self.state_space.add((y, x))

    def is_state_terminal(self, state=None):
        return False

import numpy as np

from simpleenvs.envs.discrete_rooms import DiscreteRoomEnvironment
from simpleenvs.envs.discrete_rooms.rooms import CELL_TYPES_DICT


class DiscreteExplorableRoomEnvironment(DiscreteRoomEnvironment):
    def __init__(self, *args, **kwargs):
        super(DiscreteExplorableRoomEnvironment, self).__init__(*args, **kwargs)

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


# Import room template files.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from . import data

with pkg_resources.path(data, "two_rooms.txt") as path:
    default_two_room = path

with pkg_resources.path(data, "six_rooms.txt") as path:
    default_six_room = path

with pkg_resources.path(data, "nine_rooms.txt") as path:
    default_nine_room = path

with pkg_resources.path(data, "xu_four_rooms.txt") as path:
    xu_four_room = path

with pkg_resources.path(data, "bridge_room.txt") as path:
    bridge_room = path

with pkg_resources.path(data, "cage_room.txt") as path:
    cage_room = path

with pkg_resources.path(data, "empty_room.txt") as path:
    empty_room = path

with pkg_resources.path(data, "small_rooms.txt") as path:
    small_rooms = path

with pkg_resources.path(data, "four_rooms.txt") as path:
    four_rooms = path

with pkg_resources.path(data, "four_rooms_holes.txt") as path:
    four_rooms_holes = path

with pkg_resources.path(data, "maze_rooms.txt") as path:
    maze_rooms = path

with pkg_resources.path(data, "spiral_room.txt") as path:
    spiral_rooms = path

with pkg_resources.path(data, "parr_maze.txt") as path:
    parr_maze = path

with pkg_resources.path(data, "parr_mini_maze.txt") as path:
    parr_mini_maze = path

with pkg_resources.path(data, "ramesh_maze.txt") as path:
    ramesh_maze = path

with pkg_resources.path(data, "wide_path.txt") as path:
    wide_path = path


class ExplorableTwoRooms(DiscreteExplorableRoomEnvironment):
    """
    A default two-rooms environment, as is commonly featured in the HRL literature.
    Goal Reward: +1.0
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(default_two_room, movement_penalty, goal_reward)


class ExplorableSixRooms(DiscreteExplorableRoomEnvironment):
    """
    A default six-rooms environment, as is commonly featured in the HRL literature.
    Goal Reward: +1.0
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(default_six_room, movement_penalty, goal_reward)


class ExplorableNineRooms(DiscreteExplorableRoomEnvironment):
    """
    A default nine-rooms environment, as is commonly featured in the HRL literature.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(default_nine_room, movement_penalty, goal_reward)


class ExplorableXuFourRooms(DiscreteExplorableRoomEnvironment):
    """
    The four-room environment used in Xu X., Yang M. & Li G. 2019
    "Constructing Temporally Extended Actions through Incremental
    Community Detection". Contains four offset rooms.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(xu_four_room, movement_penalty, goal_reward)


class ExplorableBridgeRoom(DiscreteExplorableRoomEnvironment):
    """
    A single-room environment feating two routes from the starting state
    to the goal state --- a longer, wider path, and a shorter, thinner path.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(bridge_room, movement_penalty, goal_reward)


class ExplorableCageRoom(DiscreteExplorableRoomEnvironment):
    """
    A single-room environment, with a small "cage" room within the larger room.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(cage_room, movement_penalty, goal_reward)


class ExplorableEmptyRoom(DiscreteExplorableRoomEnvironment):
    """
    A single, empty room environment.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(empty_room, movement_penalty, goal_reward)


class ExplorableSmallRooms(DiscreteExplorableRoomEnvironment):
    """
    A four-room environment comprised of a single large room with
    three smaller rooms above.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(small_rooms, movement_penalty, goal_reward)


class ExplorableFourRooms(DiscreteExplorableRoomEnvironment):
    """
    A default four-rooms environment, as commonly seen in the HRL literature.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(four_rooms, movement_penalty, goal_reward)


class ExplorableFourRoomsHoles(DiscreteExplorableRoomEnvironment):
    """
    A four-room environment, with a number of pillars blocking the way in one of the rooms.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(four_rooms_holes, movement_penalty, goal_reward)


class ExplorableMazeRooms(DiscreteExplorableRoomEnvironment):
    """
    A maze-style environment made up of a number of inter-connected corridors.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(maze_rooms, movement_penalty, goal_reward)


class ExplorableSpiralRoom(DiscreteExplorableRoomEnvironment):
    """
    An environment comprised of a single, spiral-shaped room.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(spiral_rooms, movement_penalty, goal_reward)


class ExplorableParrMaze(DiscreteExplorableRoomEnvironment):
    """
    The Maze gridworld introduced by Parr and Russell, 1998, to test HAMs.
    Contains ~3600 states.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(parr_maze, movement_penalty, goal_reward)


class ExplorableParrMiniMaze(DiscreteExplorableRoomEnvironment):
    """
    A smaller section of the Maze gridworld introduced by Parr and Russell, 1998, to test HAMs.
    Contains ~1800 states.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(parr_mini_maze, movement_penalty, goal_reward)


class ExplorableRameshMaze(DiscreteExplorableRoomEnvironment):
    """
    One of the maze gridworlds used by Ramesh et al. 2019 to test Successor Options.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(ramesh_maze, movement_penalty, goal_reward)


class ExplorableWidePath(DiscreteExplorableRoomEnvironment):
    """
    A single-room environment featuring a wide path from the starting state to the goal state.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1.0):
        super().__init__(wide_path, movement_penalty, goal_reward)

import copy
import random
import numpy as np

from simpleoptions import TransitionMatrixBaseEnvironment

from simpleenvs.renderers import RoomRenderer

CELL_TYPES_DICT = {".": "floor", "#": "wall", "S": "start", "G": "goal", "A": "agent"}

ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class DiscreteRoomEnvironment(TransitionMatrixBaseEnvironment):
    """
    Class representing a discrete "rooms-like" gridworld, as is commonly seen in the HRL literature.
    """

    def __init__(self, room_template_file_path, movement_penalty=-0.001, goal_reward=1.0):
        """
        Initialises a new DiscreteRoomEnvironment object.

        Arguments:
            room_template_file_path {string or Path} -- Path to a gridworld template file.

        Keyword Arguments:
            movement_penalty {float} -- Penalty applied each time step for taking an action. (default: {-1.0})
            goal_reward {float} -- Reward given to the agent upon reaching a goal state. (default: {10.0})
        """
        self._initialise_rooms(room_template_file_path)
        self._initialise_state_space()
        self.movement_penalty = movement_penalty
        self.goal_reward = goal_reward
        self.is_reset = False
        self.renderer = None
        self.current_state = None

        super().__init__(deterministic=True)

    def _initialise_rooms(self, room_template_file_path):
        """
        Initialises the envionment according to a given template file.

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
                if self.gridworld[y, x] not in CELL_TYPES_DICT:
                    if not self.gridworld[y, x].replace("-", "", 1).isnumeric():
                        raise ValueError(f"Invalid cell type '{self.gridworld[y, x]}' in room template file.")
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "start":
                    self.initial_states.append((y, x))
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "goal":
                    self.terminal_states.append((y, x))

    def _initialise_state_space(self):
        # Create set of all valid states.
        self.state_space = set()
        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):
                if self.gridworld[y, x] in CELL_TYPES_DICT and CELL_TYPES_DICT[self.gridworld[y, x]] != "wall":
                    self.state_space.add((y, x))
                elif self.gridworld[y, x].replace("-", "", 1).isnumeric():
                    self.state_space.add((y, x))

    def reset(self, state=None):
        """
        Resets the environment, setting the agent's position to a random starting state
        and selecting a random goal state.

        Arguments:
           state {(int, int)} -- The initial state to use. Defaults to None, in which case an state is chosen according to the environment's initial state distribution.

        Returns:
            [(int, int)] -- The agent's initial position.
        """

        if state is None:
            self.current_state = random.choice(self.initial_states)
        else:
            self.current_state = copy.deepcopy(state)
        self.current_initial_state = self.current_state

        self.is_reset = True

        return self.current_state

    def step(self, action, state=None):
        if state is None:
            next_state, reward, terminal, info = super().step(action, state=self.current_state)
        else:
            next_state, reward, terminal, info = super().step(action, state=state)

        self.current_state = next_state

        return next_state, reward, terminal, info

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return {0, 1, 2, 3}

    def get_available_actions(self, state=None):
        """
        Returns the set of available actions for the given state (by default, the current state).

        Keyword Arguments:
            state {(int, int)} -- The state to return available actions for. Defaults to None (uses current state).

        Returns:
            [list(int)] -- The list of actions available in the given state.
        """
        if state is None:
            state = self.current_state

        if self.is_state_terminal(state):
            return []
        else:
            return [0, 1, 2, 3]

    def get_action_mask(self, state=None):
        """
        Returns a boolean mask indicating which actions are available in the given state (by default, the current state).

        A value of True at index i indicates that this action is available.
        A value of False at index i indicates that the corresponding action is not available.

        Keyword Args:
            state {(int, int)} -- The state to return an action mask for. Defaults to None (uses current state).

        Returns:
            [list(bool)] -- A boolean mask indicating action availability in the current state.
        """
        if state is None:
            state = self.current_state

        if self.is_state_terminal(state):
            return [False for i in range(4)]
        else:
            return [True for i in range(4)]

    def render(self):
        """
        Renders the current environmental state.
        """

        if self.renderer is None:
            self.renderer = RoomRenderer(
                self.gridworld,
                start_state=self.current_initial_state,
                goal_states=self.terminal_states,
            )

        self.renderer.update(
            self.current_state,
            self.gridworld,
            start_state=self.current_initial_state,
            goal_states=self.terminal_states,
        )

    def close(self):
        """
        Terminates all environment-related processes.
        """
        if self.renderer is not None:
            self.renderer.close()

    def is_state_terminal(self, state=None):
        """
        Returns whether the given state is terminal or not.
        If no state is specified, whether the current state is terminal will be returned.

        Args:
            state (tuple, optional): Whether or not the given state is terminal. Defaults to None (i.e. current state).

        Returns:
            bool: Whether or not the given state is terminal.
        """
        if state is None:
            state = self.current_state

        # return CELL_TYPES_DICT[self.gridworld[state[0]][state[1]]] == "goal"
        return state in self.terminal_states

    def get_initial_states(self):
        """
        Returns the initial state(s) for this environment.

        Returns:
            List[Tuple[int]]: The initial state(s) in this environment.
        """
        return copy.deepcopy(self.initial_states)

    def get_successors(self, state=None, actions=None):
        if state is None:
            state = self.current_state

        if actions is None:
            actions = self.get_available_actions(state=state)

        successor_states = []
        for action in actions:
            next_state = copy.deepcopy(state)
            if ACTIONS_DICT[action] == "DOWN":
                next_state = (state[0] + 1, state[1])
            elif ACTIONS_DICT[action] == "UP":
                next_state = (state[0] - 1, state[1])
            elif ACTIONS_DICT[action] == "RIGHT":
                next_state = (state[0], state[1] + 1)
            elif ACTIONS_DICT[action] == "LEFT":
                next_state = (state[0], state[1] - 1)
            # if next state is a wall return to the current state
            if (
                self.gridworld[next_state[0]][next_state[1]] in CELL_TYPES_DICT
                and CELL_TYPES_DICT[self.gridworld[next_state[0]][next_state[1]]] == "wall"
            ):
                next_state = (state[0], state[1])

            if self.is_state_terminal(state=(next_state[0], next_state[1])):
                reward = self.goal_reward + self.movement_penalty
            else:
                if (
                    self.gridworld[next_state[0]][next_state[1]] not in CELL_TYPES_DICT
                    and self.gridworld[next_state[0]][next_state[1]].replace("-", "", 1).isnumeric()
                ):
                    reward = float(self.gridworld[next_state[0]][next_state[1]]) + self.movement_penalty
                else:
                    reward = self.movement_penalty

            successor_states.append(((next_state, reward), 1.0 / len(actions)))

        return successor_states


# Import room template files.
from importlib.resources import files

from . import data

default_two_room = files(data).joinpath("two_rooms.txt")
default_six_room = files(data).joinpath("six_rooms.txt")
default_nine_room = files(data).joinpath("nine_rooms.txt")
xu_four_room = files(data).joinpath("xu_four_rooms.txt")
bridge_room = files(data).joinpath("bridge_room.txt")
cage_room = files(data).joinpath("cage_room.txt")
empty_room = files(data).joinpath("empty_room.txt")
small_rooms = files(data).joinpath("small_rooms.txt")
four_rooms = files(data).joinpath("four_rooms.txt")
four_rooms_holes = files(data).joinpath("four_rooms_holes.txt")
maze_rooms = files(data).joinpath("maze_rooms.txt")
spiral_rooms = files(data).joinpath("spiral_room.txt")
parr_maze = files(data).joinpath("parr_maze.txt")
parr_mini_maze = files(data).joinpath("parr_mini_maze.txt")
ramesh_maze = files(data).joinpath("ramesh_maze.txt")
wide_path = files(data).joinpath("wide_path.txt")


class TwoRooms(DiscreteRoomEnvironment):
    """
    A default two-rooms environment, as is commonly featured in the HRL literature.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(default_two_room, movement_penalty, goal_reward)


class SixRooms(DiscreteRoomEnvironment):
    """
    A default six-rooms environment, as is commonly featured in the HRL literature.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(default_six_room, movement_penalty, goal_reward)


class NineRooms(DiscreteRoomEnvironment):
    """
    A default nine-rooms environment, as is commonly featured in the HRL literature.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(default_nine_room, movement_penalty, goal_reward)


class XuFourRooms(DiscreteRoomEnvironment):
    """
    The four-room environment used in Xu X., Yang M. & Li G. 2019
    "Constructing Temporally Extended Actions through Incremental
    Community Detection". Contains four offset rooms.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(xu_four_room, movement_penalty, goal_reward)


class BridgeRoom(DiscreteRoomEnvironment):
    """
    A single-room environment feating two routes from the starting state
    to the goal state --- a longer, wider path, and a shorter, thinner path.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(bridge_room, movement_penalty, goal_reward)


class CageRoom(DiscreteRoomEnvironment):
    """
    A single-room environment, with a small "cage" room within the larger room.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(cage_room, movement_penalty, goal_reward)


class EmptyRoom(DiscreteRoomEnvironment):
    """
    A single, empty room environment.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(empty_room, movement_penalty, goal_reward)


class SmallRooms(DiscreteRoomEnvironment):
    """
    A four-room environment comprised of a single large room with
    three smaller rooms above.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(small_rooms, movement_penalty, goal_reward)


class FourRooms(DiscreteRoomEnvironment):
    """
    A default four-rooms environment, as commonly seen in the HRL literature.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(four_rooms, movement_penalty, goal_reward)


class FourRoomsHoles(DiscreteRoomEnvironment):
    """
    A four-room environment, with a number of pillars blocking the way in one of the rooms.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(four_rooms_holes, movement_penalty, goal_reward)


class MazeRooms(DiscreteRoomEnvironment):
    """
    A maze-style environment made up of a number of inter-connected corridors.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(maze_rooms, movement_penalty, goal_reward)


class SpiralRoom(DiscreteRoomEnvironment):
    """
    An environment comprised of a single, spiral-shaped room.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(spiral_rooms, movement_penalty, goal_reward)


class ParrMaze(DiscreteRoomEnvironment):
    """
    The Maze gridworld introduced by Parr and Russell, 1998, to test HAMs.
    Contains ~3600 states.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(parr_maze, movement_penalty, goal_reward)


class ParrMiniMaze(DiscreteRoomEnvironment):
    """
    A smaller section of the Maze gridworld introduced by Parr and Russell, 1998, to test HAMs.
    Contains ~1800 states.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(parr_mini_maze, movement_penalty, goal_reward)


class RameshMaze(DiscreteRoomEnvironment):
    """
    One of the maze gridworlds used by Ramesh et al. 2019 to test Successor Options.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(ramesh_maze, movement_penalty, goal_reward)


class WidePath(DiscreteRoomEnvironment):
    """
    A single-room environment featuring a wide path from the starting state to the goal state.
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(wide_path, movement_penalty, goal_reward)

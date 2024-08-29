import copy
from importlib.resources import files
import math

import numpy as np
import networkx as nx

from simpleoptions import TransitionMatrixBaseEnvironment
from simpleenvs.renderers import RoomRenderer

# Import room template files.
from . import data

CELL_TYPES_DICT = {".": "floor", "#": "wall", "S": "start", "G": "goal"}

ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "TERMINATE"}


class ProximityRoomEnvironment(TransitionMatrixBaseEnvironment):
    """Represents a discrete rooms-like gridworld with distance-to-goal based reward on termination.

    Reward function is a small penalty per step. On termination an additional penalty proportional to the distance-to-goal is given.
    """

    def __init__(
        self,
        room_template_file_path,
        movement_penalty=-1.0,
        distance_penalty_factor=1.0,
        initial_state=None,
        goal_state=None,
    ):

        self.is_reset = True
        self.renderer = None
        self.current_state = None
        self.terminal_state = (-math.inf, -math.inf)
        self._initialise_rooms(room_template_file_path, initial_state, goal_state)
        self._initialise_state_space()
        self.movement_penalty = movement_penalty
        self.distance_penalty_factor = distance_penalty_factor
        self.stg = self.generate_interaction_graph()

        super().__init__(deterministic=True)
        self.is_reset = False

    def _initialise_rooms(self, room_template_file_path, initial_state=None, goal_state=None):
        """
        Initialises the envionment according to a given template file.

        Arguments:
            room_template_file_path {string or Path} -- Path to a gridworld template file.
            initial_state
            goal_state
        """

        # Load gridworld template file.
        self.gridworld = np.loadtxt(room_template_file_path, comments="//", dtype=str)

        # Discover start and goal states.
        initial_states = []
        goal_states = []
        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):
                if self.gridworld[y, x] not in CELL_TYPES_DICT:
                    if not self.gridworld[y, x].replace("-", "", 1).isnumeric():
                        raise ValueError(f"Invalid cell type '{self.gridworld[y, x]}' in room template file.")
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "start":
                    initial_states.append((y, x))
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "goal":
                    goal_states.append((y, x))

        if initial_state:
            self.initial_state = initial_state
        elif len(initial_states) != 1:
            raise ValueError(f"{len(initial_states)} initial states defined, only one allowed.")
        else:
            self.initial_state = initial_states[0]

        if goal_state:
            self.goal_state = goal_state
        elif len(goal_states) != 1:
            raise ValueError(f"{len(goal_states)} goal states defined, only one allowed.")
        else:
            self.goal_state = goal_states[0]

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
            self.current_state = self.initial_state
        else:
            self.current_state = copy.deepcopy(state)

        self.current_initial_state = self.current_state

        self.is_reset = True

        return self.current_state

    def step(self, action, state=None):

        if state is None:
            state = self.current_state
        next_state, reward, terminal, info = super().step(action, state)

        self.current_state = next_state

        if terminal:
            self.is_reset = False
            reward -= self.distance_penalty_factor * self.distance_to_goal(state)

        return next_state, reward, terminal, info

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return set(ACTIONS_DICT.keys())

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

        if self.is_state_terminal(state) or not self.is_reset:
            return []
        else:
            return ACTIONS_DICT.keys()

    def encode(self, state) -> int:
        y, x = state
        idx = (self.gridworld.shape[1] * y) + x
        idx -= np.count_nonzero(self.gridworld.flatten()[:idx] == "#")
        return int(idx)

    def decode(self, idx):
        n = 0
        for state, value in np.ndenumerate(self.gridworld):
            if CELL_TYPES_DICT[value] == "wall":
                idx += 1
            if n == idx:
                return state
            n += 1

    def render(self):
        """
        Renders the current environmental state.
        """

        if self.renderer is None:
            self.renderer = RoomRenderer(
                self.gridworld,
                start_state=self.current_initial_state,
                goal_states=self.goal_state,
            )

        self.renderer.update(
            self.current_state,
            self.gridworld,
            start_state=self.current_initial_state,
            goal_states=self.goal_state,
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

        return state == self.terminal_state

    def get_initial_states(self):
        """
        Returns the initial state(s) for this environment.

        Returns:
            List[Tuple[int]]: The initial state(s) in this environment.
        """
        return [self.initial_state]

    def get_successors(self, state=None, actions=None):
        if state is None:
            state = self.current_state

        if actions is None:
            actions = self.get_available_actions(state=state)

        successor_states = []
        for action in actions:
            next_state = copy.deepcopy(state)
            if ACTIONS_DICT[action] == "TERMINATE":
                next_state = self.terminal_state
            else:
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
                    next_state = state
            reward = self.movement_penalty

            successor_states.append(((next_state, reward), 1.0 / len(actions)))

        return successor_states

    def distance_to_goal(self, state):
        return nx.dijkstra_path_length(self.stg, state, self.goal_state)


class FourRoomsProximity(ProximityRoomEnvironment):
    def __init__(self, movement_penalty=-1, distance_penalty_factor=1, initial_state=None, goal_state=None):
        four_rooms = files(data).joinpath("four_rooms.txt")
        super().__init__(four_rooms, movement_penalty, distance_penalty_factor, initial_state, goal_state)


class RameshMazeProximity(ProximityRoomEnvironment):
    def __init__(self, movement_penalty=-1, distance_penalty_factor=1, initial_state=None, goal_state=None):
        maze = files(data).joinpath("ramesh_maze.txt")
        super().__init__(maze, movement_penalty, distance_penalty_factor, initial_state, goal_state)

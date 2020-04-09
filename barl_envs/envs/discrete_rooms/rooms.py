import random
import operator
import numpy as np

from barl_envs.renderers import RoomRenderer

# Import room template files.
try :
    import importlib.resources as pkg_resources
except ImportError :
    import importlib_resources as pkg_resources

from . import data

with pkg_resources.path(data, "two_rooms.txt") as path :
    default_two_room = path

with pkg_resources.path(data, "six_rooms.txt") as path :
    default_six_room = path

CELL_TYPES_DICT = {
    "." : "floor",
    "#" : "wall",
    "S" : "start",
    "G" : "goal",
    "A" : "agent"
}

ACTIONS_DICT = {
    0 : "UP",
    1 : "DOWN",
    2 : "LEFT",
    3 : "RIGHT"
}

class DiscreteRoomEnvironment(object) :

    def __init__(self, room_template_file_path, movement_penalty = -1, goal_reward = 10) :
        self._initialise_rooms(room_template_file_path)
        self.movement_penalty = movement_penalty
        self.goal_reward = goal_reward
        self.is_reset = False
        self.renderer = None

    def _initialise_rooms(self, room_template_file_path) :
        # Load gridworld template file.
        self.gridworld = np.loadtxt(room_template_file_path, comments = "//", dtype = str)

        print(self.gridworld.shape)

        # Discover start and goal states.
        self.initial_states = []
        self.terminal_states = []
        for y in range(self.gridworld.shape[0]) :
            for x in range(self.gridworld.shape[1]) :
                if (CELL_TYPES_DICT[self.gridworld[y, x]] == "start") :
                    self.initial_states.append((y, x))
                elif (CELL_TYPES_DICT[self.gridworld[y, x]] == "goal") :
                    self.terminal_states.append((y, x))

    def reset(self) :

        self.position = random.choice(self.initial_states)
        self.goal = random.choice(self.terminal_states)

        self.current_initial_state = self.position
        self.current_goal_state = self.goal

        self.is_reset = True

        return (self.position[0], self.position[1])

    def step(self, action) :

        current_state = self.position
        next_state = self.position

        if (ACTIONS_DICT[action] == "DOWN") :
            next_state = (next_state[0] + 1, next_state[1])
        elif (ACTIONS_DICT[action] == "UP") :
            next_state = (next_state[0] - 1, next_state[1])
        elif (ACTIONS_DICT[action] == "RIGHT") :
            next_state = (next_state[0], next_state[1] + 1)
        elif (ACTIONS_DICT[action] == "LEFT") :
            next_state = (next_state[0], next_state[1] - 1)

        reward = 0
        terminal = False

        if (CELL_TYPES_DICT[self.gridworld[next_state[0]][next_state[1]]] == "wall") :
            next_state = current_state
        elif (CELL_TYPES_DICT[self.gridworld[next_state[0]][next_state[1]]] == "goal") :
            reward += self.goal_reward
            terminal = True

        if (terminal) :
            self.is_reset = False

        reward += self.movement_penalty

        self.position = next_state

        return next_state, reward, terminal

    def get_available_actions(self) :
        return list(range(4))

    def render(self) :
        if (self.renderer is None) :
            self.renderer = RoomRenderer(self.gridworld, start_state = self.current_initial_state, goal_state = self.current_goal_state)

        self.renderer.update(self.position, self.gridworld, start_state = self.current_initial_state, goal_state = self.current_goal_state)

    def close(self) :
        if (self.renderer is not None) :
            self.renderer.close()


class DiscreteDefaultTwoRooms(DiscreteRoomEnvironment) :
    def __init__(self, movement_penalty = -1, goal_reward = 10) :
        super().__init__(default_two_room, movement_penalty, goal_reward)

class DiscreteDefaultSixRooms(DiscreteRoomEnvironment) :
    def __init__(self, movement_penalty = -1, goal_reward = 10) :
        super().__init__(default_six_room, movement_penalty, goal_reward)

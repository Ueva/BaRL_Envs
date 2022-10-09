import copy
import random
import numpy as np

from itertools import cycle

from barl_envs.envs.discrete_rooms import DiscreteRoomEnvironment

from . import data

# Import room template files.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

with pkg_resources.path(data, "four_rooms_transfer.txt") as path:
    transfer_four_room = path

with pkg_resources.path(data, "xu_four_rooms_transfer.txt") as path:
    transfer_xu_four_room = path

with pkg_resources.path(data, "nine_rooms_transfer.txt") as path:
    transfer_nine_room = path

CELL_TYPES_DICT = {".": "floor", "#": "wall", "S": "start", "G": "goal", "A": "agent"}

ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class TransferRoomEnvironment(DiscreteRoomEnvironment):
    def __init__(
        self,
        room_template_file_path,
        movement_penalty=-0.001,
        goal_reward=1.0,
        options=[],
        initial_states_order=None,
        goal_states_order=None,
    ):
        super().__init__(room_template_file_path, movement_penalty, goal_reward, options)

        if initial_states_order is None:
            self.initial_states_order = None
        else:
            self.initial_states_order = cycle(initial_states_order)

        if goal_states_order is None:
            self.goal_states_order = None
        else:
            self.goal_states_order = cycle(goal_states_order)

    def reset(self, state=None, goal_state=None):
        # If an initial state is specified, use it.
        if state is not None:
            self.position = copy.deepcopy(state)
        # Else, if we have a defined initial state order, use the next initial state.
        elif self.initial_states_order is not None:
            self.position = copy.deepcopy(next(self.initial_states_order))
        # Else, randomly sample an initial state.
        else:
            self.position = random.choice(self.initial_states)

        # If a goal state is specified, use it.
        if goal_state is not None:
            self.goals = [copy.deepcopy(goal_state)]
        # Else, if we have a defined goal state order, use the next goal state.
        elif self.goal_states_order is not None:
            self.goals = [copy.deepcopy(next(self.goal_states_order))]
        # Else, randomly sample a goal state.
        else:
            self.goals = [random.choice(self.terminal_states)]

        self.current_initial_state = self.position

        self.is_reset = True

        # Make sure that the initial and goal states are legal.
        assert self.position in self.initial_states
        assert all([goal in self.terminal_states for goal in self.goals])

        return (self.position[0], self.position[1])


class DiscreteTransferFourRooms(TransferRoomEnvironment):
    """
    A four-room gridworld environment suitable for transfer learning experiments.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(transfer_four_room, movement_penalty, goal_reward)


class DiscreteTransferXuFourRooms(TransferRoomEnvironment):
    """
    A version of the four-room environment used in Xu X., Yang M. & Li G. 2019
    "Constructing Temporally Extended Actions through Incremental Community Detection"
    made suitable for transfer learning experiments.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(transfer_xu_four_room, movement_penalty, goal_reward)


class DiscreteTransferNineRooms(TransferRoomEnvironment):
    """
    A nine-room gridworld environment suitable for transfer learning experiments.
    Goal Reward: +1
    Movement Penalty: -0.001
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(transfer_nine_room, movement_penalty, goal_reward)

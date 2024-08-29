import copy

from itertools import product

from simpleenvs.envs.discrete_rooms import (
    DiscreteRoomEnvironment,
    CELL_TYPES_DICT,
    ACTIONS_DICT,
)


class ExtraItemsDiscreteRoomEnvironment(DiscreteRoomEnvironment):
    def __init__(
        self,
        room_template_file_path,
        movement_penalty=-0.001,
        goal_reward=1.0,
        persistent_items=False,  # Whether the additional items are persistent or consumed upon collection
    ):
        self.basic_init = True
        super().__init__(room_template_file_path, movement_penalty, goal_reward)
        self.basic_init = False
        self.positions = self.state_space
        self.item_locations = self.get_item_locations()
        self.persistent_items = persistent_items
        if not self.persistent_items:
            self.state_space, self.terminal_states = self.adjust_for_item(
                self.state_space, self.terminal_states, self.item_locations
            )
        self.transition_matrix = self._compute_transition_matrix()

    def get_item_locations(self):
        return [
            (x, y)
            for x in range(self.gridworld.shape[0])
            for y in range(self.gridworld.shape[1])
            if self.gridworld[x, y].replace("-", "", 1).isnumeric()
        ]

    def adjust_for_item(self, state_space, terminal_states, item_locations):
        """
        Given the number of item positions, adjusts the state space and terminal states.
        Increases the state space to include all possible combinations of item collection states.
        Removes the original item positions from the state space.
        Adds all combinations of the terminal states with different amounts of item.
        """
        # Create a list of all possible combinations of item collection states
        num_items = len(item_locations)
        item_combinations = list(product([0, 1], repeat=num_items))

        # Create the modified states array
        modified_states = []
        for state in state_space:
            for combination in item_combinations:
                modified_state = list(state) + list(combination)
                while modified_state and modified_state[-1] == 0:
                    modified_state.pop()
                trimmed_state = tuple(modified_state)
                modified_states.append(trimmed_state)
                if state in terminal_states and trimmed_state not in terminal_states:
                    terminal_states.append(trimmed_state)

        # remove the original item positions from the state space
        for item in item_locations:
            modified_states.remove(item)

        return modified_states, terminal_states

    def has_picked_up_item(self, state):
        if self.persistent_items:
            return False
        position = (state[0], state[1])
        if len(state) <= 2 or not hasattr(self, "item_locations"):
            return False
        # get the index of the x,y position in the item locations
        item_index = self.item_locations.index(position)
        if len(state) <= 2 + item_index:
            # print(f"ERR:\tstate: {state}\titem_index: {item_index}")
            return False
        return state[2 + item_index] == 1

    def get_successors(self, state=None, actions=None):
        if self.basic_init:  # if initial setup is being established, use the basic setup
            return super().get_successors(state=state, actions=actions)
        if state is None:
            state = self.current_state

        if actions is None:
            actions = self.get_available_actions(state=state)

        successor_states = []
        for action in actions:
            next_state = copy.deepcopy(state)
            if ACTIONS_DICT[action] == "DOWN":
                next_state = (state[0] + 1, state[1], *state[2:])
            elif ACTIONS_DICT[action] == "UP":
                next_state = (state[0] - 1, state[1], *state[2:])
            elif ACTIONS_DICT[action] == "RIGHT":
                next_state = (state[0], state[1] + 1, *state[2:])
            elif ACTIONS_DICT[action] == "LEFT":
                next_state = (state[0], state[1] - 1, *state[2:])
            # if next state is a wall return to the current state
            if (
                self.gridworld[next_state[0]][next_state[1]] in CELL_TYPES_DICT
                and CELL_TYPES_DICT[self.gridworld[next_state[0]][next_state[1]]] == "wall"
            ):
                next_state = (state[0], state[1], *state[2:])

            if self.is_state_terminal(state=(next_state[0], next_state[1])):
                reward = self.goal_reward
            else:  # state is either a floor or a item position
                # if ns is a item position and the item has not been picked up
                if (
                    self.gridworld[next_state[0]][next_state[1]] not in CELL_TYPES_DICT
                    and self.gridworld[next_state[0]][next_state[1]].replace("-", "", 1).isnumeric()
                    and not self.has_picked_up_item(next_state)
                ):
                    # get the reward at that position
                    reward = float(self.gridworld[next_state[0]][next_state[1]]) + self.movement_penalty
                    if not self.persistent_items:
                        # get the id of which item it is
                        item_index = self.item_locations.index(next_state[:2])
                        next_state = list(next_state)
                        # fill out the remainder of the state item flags
                        while len(next_state) < 2 + item_index + 1:
                            next_state.append(0)
                        next_state[2 + item_index] = 1
                        next_state = tuple(next_state)
                else:
                    reward = self.movement_penalty

            successor_states.append(((next_state, reward), 1.0 / len(actions)))

        return successor_states


# Import room template files.
from importlib.resources import files

from . import data


basic_reward_room = files(data).joinpath("basic_reward_room.txt")
double_reward_room = files(data).joinpath("double_reward_room.txt")
basic_penalty_room = files(data).joinpath("basic_penalty_room.txt")
double_penalty_room = files(data).joinpath("double_penalty_room.txt")
four_rooms_firewall = files(data).joinpath("four_rooms_firewall.txt")
four_rooms_penalty = files(data).joinpath("four_rooms_penalty.txt")


class BasicRewardRoom(ExtraItemsDiscreteRoomEnvironment):
    """
    An 11x11 grid with start in the top left and goal in the bottom right.
    The environment has an additional reward of 10 in the bottom left.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(basic_reward_room, movement_penalty, goal_reward)


class DoubleRewardRoom(ExtraItemsDiscreteRoomEnvironment):
    """
    An 11x11 grid with start in the top left and goal in the bottom right.
    The environment has two additional rewards of 10 in the bottom left and top right.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(double_reward_room, movement_penalty, goal_reward)


class BasicPenaltyRoom(ExtraItemsDiscreteRoomEnvironment):
    """
    An 11x11 grid with start in the top left and goal in the bottom right.
    The environment has an additional penalties of -10 in the bottom left that can be picked up.
    Once a penalty state has been visited, returning to this position will not result in an additional penalty.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(basic_penalty_room, movement_penalty, goal_reward)


class DoublePenaltyRoom(ExtraItemsDiscreteRoomEnvironment):
    """
    An 11x11 grid with start in the top left and goal in the bottom right.
    The environment has two additional penalties of -10 in the bottom left and top right that can be picked up.
    Once a penalty state has been visited, returning to this position will not result in an additional penalty.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(double_penalty_room, movement_penalty, goal_reward)


class FourRoomsFireWall(ExtraItemsDiscreteRoomEnvironment):
    """
    An instance of four-rooms except there are no walls, only states that give a
    high negative reward when visited. This high negative reward is received every
    time the agent visits that state.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(four_rooms_firewall, movement_penalty, goal_reward, persistent_items=True)


class FourRoomsPenalty(ExtraItemsDiscreteRoomEnvironment):
    """
    An instance of four-rooms with an additional penalty of -10 in the bottom left.
    Goal Reward: +1
    Movement Penalty: -0.01
    """

    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(four_rooms_penalty, movement_penalty, goal_reward)

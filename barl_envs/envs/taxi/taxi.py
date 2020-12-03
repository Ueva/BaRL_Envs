from os import terminal_size
from types import TracebackType

from gym.core import ActionWrapper
import gym
import math
import copy
import random
import itertools

from barl_envs.renderers import TaxiRenderer


ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PICKUP", 5: "PUTDOWN"}
TAXI_RANKS = [0, 3, 20, 24, -1]


class TaxiEnvironment(object):
    def __init__(self, movement_penalty=-1.0, goal_reward=20.0, invalid_penalty=-10):

        # Define action-space and state-space.
        self.action_space = gym.spaces.Discrete(6)
        self.state_space = gym.spaces.Tuple((gym.spaces.Discrete(25), gym.spaces.Discrete(5), gym.spaces.Discrete(4)))

        self.movement_penalty = movement_penalty
        self.goal_reward = goal_reward
        self.invalid_penalty = invalid_penalty

        self.current_state = None
        self.source_state = None
        self.destination_state = None
        self.terminal = True

        self.renderer = None

    def reset(self, state=None):
        # If no state is specified, randomly sample an initial state.
        if state is None:
            self.current_state = copy.deepcopy(random.sample(self.get_initial_states(), 1)[0])
        # Else use the specified state as the initial state.
        else:
            self.current_state = copy.deepcopy(state)

        self.terminal = False
        return self.current_state

    def step(self, action):

        right_wall = [0, 5, 16, 21, 2, 7, 4, 9, 14, 19, 24]
        left_wall = [0, 5, 10, 15, 20, 1, 6, 17, 22, 3, 8]
        up_wall = [20, 21, 22, 23, 24]
        down_wall = [0, 1, 2, 3, 4]

        taxi_pos, passenger_pos, goal_pos = self.current_state
        taxi_x, taxi_y = self._number_to_coords(taxi_pos)

        reward = self.movement_penalty

        ## Movement actions.
        # Tries to move right when not blocked.
        if ACTIONS_DICT[action] == "RIGHT" and (taxi_pos not in right_wall):
            taxi_x += 1
        # Tries to move left when not blocked.
        elif ACTIONS_DICT[action] == "LEFT" and (taxi_pos not in left_wall):
            taxi_x -= 1
        # Tries to move up when not blocked.
        elif ACTIONS_DICT[action] == "UP" and (taxi_pos not in up_wall):
            taxi_y += 1
        # Tries to move down when not blocked.
        elif ACTIONS_DICT[action] == "DOWN" and (taxi_pos not in down_wall):
            taxi_y -= 1

        ## Pickup action.
        if ACTIONS_DICT[action] == "PICKUP":
            # Tries to pickup when able to pick up the passenger.
            if taxi_pos == TAXI_RANKS[passenger_pos]:
                passenger_pos = 4
            # Tries to pickup when unable to pick up the passenger.
            else:
                reward += self.invalid_penalty

        ## Putdown action.
        if ACTIONS_DICT[action] == "PUTDOWN":
            # Tries to putdown correctly.
            if taxi_pos == TAXI_RANKS[goal_pos] and TAXI_RANKS[passenger_pos] == -1:
                passenger_pos = goal_pos
                reward += self.goal_reward
                self.terminal = True
            # Tries to putdown incorrectly.
            else:
                reward += -10

        taxi_pos = self._coords_to_number(taxi_x, taxi_y)

        self.current_state = (taxi_pos, passenger_pos, goal_pos)
        return copy.deepcopy(self.current_state), reward, self.terminal, {}

    def render(self, mode="human"):
        pass

    def close(self):
        """
        Cleanly stops the environment, closing any associated renderer.
        """
        # Close renderer, if one exists.
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_available_actions(self, state=None):
        """
        Returns the list of actions available in the given state.
        If no state is specified, the actions available in the current state will be returned.

        Args:
            state (tuple, optional): The state. Defaults to None (i.e. the current state).

        Returns:
            List[int]: List of actions available in the given state.
        """
        return [0, 1, 2, 3, 4, 5]

    def get_action_mask(self, state=None):
        """
        Returns a boolean mask indicating which actions are available in the given state.
        If no state is specified, an action mask for the current state will be returned.

        A value of True at index i indicates that this action is available.
        A value of False at index i indicates that the corresponding action is not available.

        Keyword Arguments:
            state (tuple, optional) -- The state to return an action mask for. Defaults to None (i.e. current state).
        Returns:
            list[int]: The list of actions available in the given state.
        """
        if state is None:
            state = self.current_state

        # Get legal actions in given state.
        legal_actions = self.get_available_actions(state=state)

        # Get list of all actions.
        all_actions = list(range(len(self.action_list)))

        # True is action is in legal actions, false otherwise.
        legal_action_mask = map(lambda action: action in legal_actions, all_actions)

        return list(legal_action_mask)

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

        taxi_pos, passenger_pos, goal_pos = state

        # A state is only terminal if it is the goal state.
        return TAXI_RANKS[passenger_pos] == TAXI_RANKS[goal_pos]

    def get_initial_states(self):
        """
        Returns the initial state(s) for this environment.

        Returns:
            List[Tuple[int]]: The initial state(s) in this environment.
        """
        # An initial state can be any of the states where the passenger is not in the taxi,
        # i.e. where second value in the tuple is not 5.
        initial_states = []
        for start_square in range(25):
            for source_square in range(4):
                for destination_square in range(4):
                    initial_states.append((start_square, source_square, destination_square))

        return initial_states

    def get_successors(self, state=None):
        """
        Returns a list of states which can be reached by taking an action in the given state.
        If no state is specified, a list of successor states for the current state will be returned.

        Args:
            state (tuple, optional): The state to return successors for. Defaults to None (i.e. current state).

        Returns:
            list[tuple]: A list of states reachable by taking an action in the given state.
        """
        if state is None:
            state = self.current_state

        legal_actions = self.get_available_actions(state=state)

        # Creates a list of all states which can be reached by
        # taking the legal actions available in the given state.
        successor_states = []
        for action in legal_actions:

            right_wall = [0, 5, 16, 21, 2, 7, 4, 9, 14, 19, 24]
            left_wall = [0, 5, 10, 15, 20, 1, 6, 17, 22, 3, 8]
            up_wall = [20, 21, 22, 23, 24]
            down_wall = [0, 1, 2, 3, 4]

            taxi_pos, passenger_pos, goal_pos = state
            taxi_x, taxi_y = self._number_to_coords(taxi_pos)

            ## Movement actions.
            # Tries to move right when not blocked.
            if ACTIONS_DICT[action] == "RIGHT" and taxi_pos not in right_wall:
                taxi_x += 1
            # Tries to move left when not blocked.
            elif ACTIONS_DICT[action] == "LEFT" and taxi_pos not in left_wall:
                taxi_x -= 1
            # Tries to move up when not blocked.
            elif ACTIONS_DICT[action] == "UP" and taxi_pos not in up_wall:
                taxi_y += 1
            # Tries to move down when not blocked.
            elif ACTIONS_DICT[action] == "DOWN" and taxi_pos not in down_wall:
                taxi_y -= 1

            ## Pickup action.
            if ACTIONS_DICT[action] == "PICKUP":
                # Tries to pickup when able to pick up the passenger.
                if taxi_pos == TAXI_RANKS[passenger_pos]:
                    passenger_pos = 4

            ## Putdown action.
            if ACTIONS_DICT[action] == "PUTDOWN":
                # Tries to putdown correctly.
                if taxi_pos == TAXI_RANKS[goal_pos] and TAXI_RANKS[passenger_pos] == -1:
                    passenger_pos = goal_pos

            taxi_pos = self._coords_to_number(taxi_x, taxi_y)
            successor_states.append(copy.deepcopy((taxi_pos, passenger_pos, goal_pos)))

        return successor_states

    def _number_to_coords(self, square_number):
        taxi_y, taxi_x = divmod(square_number, 5)
        return taxi_x, taxi_y

    def _coords_to_number(self, x, y):
        return 5 * y + x

import copy
import random

from itertools import cycle

from simpleoptions import BaseEnvironment, TransitionMatrixBaseEnvironment

from simpleenvs.renderers import TaxiRenderer


ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PICKUP", 5: "PUTDOWN"}
TAXI_RANKS = [0, 3, 20, 24, -1]  # -1 means that the passenger is inside the Taxi.


class TaxiEnvironment(TransitionMatrixBaseEnvironment):
    def __init__(self, movement_penalty=-0.001, goal_reward=1.0, invalid_penalty=-0.001, initial_states_order=None):
        self.movement_penalty = movement_penalty
        self.goal_reward = goal_reward
        self.invalid_penalty = invalid_penalty

        self.current_state = None
        self.source_state = None
        self.destination_state = None
        self.terminal = True

        self.state_space = set(self.generate_interaction_graph(directed=True).nodes)

        self.renderer = None

        if initial_states_order is None:
            self.initial_states_order = None
        else:
            self.initial_states_order = cycle(initial_states_order)

        super().__init__()

    def reset(self, state=None):
        # If an initial state is specified, use it.
        if state is not None:
            self.current_state = copy.deepcopy(state)
        # Else, if we have a defined initial state order, use the next initial state.
        elif self.initial_states_order is not None:
            self.current_state = copy.deepcopy(next(self.initial_states_order))
        # Else, randomly sample an initial state.
        else:
            self.current_state = copy.deepcopy(random.sample(self.get_initial_states(), 1)[0])

        self.terminal = False
        return self.current_state

    def step(self, action, state=None):
        if state is None:
            next_state, reward, terminal, info = super().step(action, state=self.current_state)
        else:
            next_state, reward, terminal, info = super().step(action, state=state)

        self.current_state = next_state

        return next_state, reward, terminal, info

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

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return {0, 1, 2, 3, 4, 5}

    def get_available_actions(self, state=None):
        """
        Returns the list of actions available in the given state.
        If no state is specified, the actions available in the current state will be returned.

        Args:
            state (tuple, optional): The state. Defaults to None (i.e. the current state).

        Returns:
            List[int]: List of actions available in the given state.
        """
        if state is None:
            state = self.current_state

        if self.is_state_terminal(state):
            return []
        else:
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
        all_actions = self.get_action_space()

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
                    if source_square != destination_square:
                        initial_states.append((start_square, source_square, destination_square))

        # Only return initial states where the passenger does not start on its destination square.
        return initial_states

    def get_successors(self, state=None, actions=None):
        """
        Returns a list of states which can be reached by taking an action in the given state.
        If no state is specified, a list of successor states for the current state will be returned.

        Args:
            state (tuple, optional): The state to return successors for. Defaults to None (i.e. current state).
            actions (List[Hashable], optional): The actions to test in the given state when searching for successors. Defaults to None (i.e. tests all available actions).


        Returns:
            list[tuple]: A list of states reachable by taking an action in the given state.
        """
        if state is None:
            state = self.current_state

        if actions is None:
            actions = self.get_available_actions(state=state)

        # Creates a list of all states which can be reached by
        # taking the legal actions available in the given state.
        successor_states = []
        for action in actions:
            right_wall = [0, 5, 16, 21, 2, 7, 4, 9, 14, 19, 24]
            left_wall = [0, 5, 10, 15, 20, 1, 6, 17, 22, 3, 8]
            up_wall = [20, 21, 22, 23, 24]
            down_wall = [0, 1, 2, 3, 4]

            taxi_pos, passenger_pos, goal_pos = state
            taxi_x, taxi_y = self._number_to_coords(taxi_pos)

            reward = self.movement_penalty

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
                else:
                    reward += self.invalid_penalty

            ## Putdown action.
            if ACTIONS_DICT[action] == "PUTDOWN":
                # Tries to putdown correctly.
                if taxi_pos == TAXI_RANKS[goal_pos] and TAXI_RANKS[passenger_pos] == -1:
                    passenger_pos = goal_pos
                else:
                    reward += self.invalid_penalty

            taxi_pos = self._coords_to_number(taxi_x, taxi_y)
            successor_states.append((((taxi_pos, passenger_pos, goal_pos), reward), 1.0 / len(actions)))

        return successor_states

    def _number_to_coords(self, square_number):
        taxi_y, taxi_x = divmod(square_number, 5)
        return taxi_x, taxi_y

    def _coords_to_number(self, x, y):
        return 5 * y + x

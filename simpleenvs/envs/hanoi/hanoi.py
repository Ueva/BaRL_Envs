# Much of this code is based on: https://github.com/RobertTLange/gym-hanoi/blob/master/gym_hanoi/envs/hanoi_env.py
import copy
import itertools

from simpleoptions.environment import TransititonMatrixBaseEnvironment

from simpleenvs.renderers import HanoiRenderer


class HanoiEnvironment(TransititonMatrixBaseEnvironment):
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_disks=4, num_poles=3, start_state=None, goal_state=None):
        """
        Instantiates a new HanoiEnvironment object with a specified number
        of disks and poles.

        Args:
            num_disks (int, optional): Number of poles in the environment. Defaults to 4.
            num_poles (int, optional): Number of disks in the environment. Defaults to 3.
        """
        self.num_disks = num_disks
        self.num_poles = num_poles

        # Define action-space and state-space.
        # self.action_space = gym.spaces.Discrete(math.factorial(self.num_poles) / math.factorial(self.num_poles - 2))
        # self.state_space = gym.spaces.Tuple(self.num_disks * (gym.spaces.Discrete(self.num_poles),))

        # Initialise state and action mappings.
        self.action_list = list(itertools.permutations(list(range(self.num_poles)), 2))
        self.state_list = list(itertools.product(list(range(self.num_poles)), repeat=self.num_disks))

        # Set start state.
        if start_state is not None:
            assert len(start_state) == self.num_disks
            self.start_state = start_state
        else:
            self.start_state = self.num_disks * (0,)

        # Set goal state.
        if goal_state is not None:
            assert len(goal_state) == self.num_disks
            self.goal_state = goal_state
        else:
            self.goal_state = self.num_disks * (self.num_poles - 1,)

        # Initialise environment state variables.
        self.current_state = None
        self.terminal = True

        self.renderer = None

        super().__init__(deterministic=True)

    def reset(self, state=None):
        """
        Resets the environment to an initial state, with all disks stacked
        on the leftmost pole (i.e. pole with index zero).

        Arguments:
           state (tuple) -- The initial state to use. Defaults to None, in which case an state is chosen according to the environment's initial state distribution.

        Returns:
            tuple: Initial environmental state.
        """

        if state is None:
            self.current_state = copy.deepcopy(self.start_state)
        else:
            self.current_state = copy.deepcopy(state)

        self.terminal = False
        return copy.deepcopy(self.current_state)

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
        return self.state_list

    def get_action_space(self):
        return set(range(len(self.action_list)))

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
            legal_actions = [
                i for i, action in enumerate(self.action_list) if self._is_action_legal(action, state=state)
            ]
            return legal_actions

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

    def get_initial_states(self):
        """
        Returns the initial state(s) for this environment.

        Returns:
            List[Tuple[int]]: The initial state(s) in this environment.
        """
        return [self.start_state]

    def get_successors(self, state=None, actions=None):
        if state is None:
            state = self.current_state

        if actions is None:
            actions = self.get_available_actions(state=state)

        # Creates a list of all states which can be reached by
        # taking the legal actions available in the given state.
        successor_states = []
        for action in actions:
            successor_state = list(state)
            source_pole, dest_pole = self.action_list[action]
            disk_to_move = min(self._disks_on_pole(source_pole, state=state))
            successor_state[disk_to_move] = dest_pole
            successor_state = tuple(successor_state)

            reward = 1.0 if successor_state == self.goal_state else -0.001

            successor_states.append(((successor_state, reward), 1.0 / len(actions)))

        return successor_states

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

        # A state is only terminal if it is the goal state.
        return state == self.goal_state

    def _is_action_legal(self, action, state=None):
        if state is None:
            state = self.current_state

        source_pole, dest_pole = action
        source_disks = self._disks_on_pole(source_pole, state=state)
        dest_disks = self._disks_on_pole(dest_pole, state=state)

        if source_disks == []:
            # Cannot move a disk from an empty pole!
            return False
        else:
            if dest_disks == []:
                # Can always move a disk to an empty pole!
                return True
            else:
                # Otherwise, only allow the move if the smallest disk on the
                # source pole is smaller than the smallest disk on destination pole.
                return min(source_disks) < min(dest_disks)

    def _disks_on_pole(self, pole, state=None):
        if state is None:
            state = self.current_state
        return [disk for disk in range(self.num_disks) if state[disk] == pole]

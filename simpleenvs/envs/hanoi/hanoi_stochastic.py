# Much of this code is based on: https://github.com/RobertTLange/gym-hanoi/blob/master/gym_hanoi/envs/hanoi_env.py
import random
import copy
import itertools

import networkx as nx

from simpleoptions.environment import BaseEnvironment

from simpleenvs.renderers import HanoiRenderer


class StochasticHanoiEnvironment(BaseEnvironment):
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_disks=3, num_poles=3, start_state=None, goal_state=None):
        """
        Instantiates a new HanoiEnvironment object with a specified number
        of disks and poles.

        Args:
            num_disks (int, optional): Number of poles in the environment. Defaults to 3.
            num_poles (int, optional): Number of disks in the environment. Defaults to 3.
        """
        super().__init__()

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

    def step(self, action):
        """
        Executes the given action in the environment, causing it to transition
        to a new state and yield some reward.

        Args:
            action (int): Index of the action to execute.

        Raises:
            RuntimeError: Raised when the environment has not been reset at the start of a new episode.

        Returns:
            next_state (tuple), reward (float), terminal (bool), info (dict)
        """
        if self.terminal:
            raise RuntimeError("Please call env.reset() before starting a new episode.")

        # Initialise transition info.
        info = {"invalid_action": False}
        new_state = list(self.current_state)

        # With probability 0.8, the agent's action is executed as normal.
        # With probability 0.2, a random action is executed.
        if random.random() > 0.8:
            action = random.choice(self.get_available_actions(new_state))

        source_pole, dest_pole = self.action_list[action]

        # If the chosen action is legal, determine the next state.
        if self._is_action_legal((source_pole, dest_pole)):
            disk_to_move = min(self._disks_on_pole(source_pole))
            new_state[disk_to_move] = dest_pole
            new_state = tuple(new_state)
        # If the chosen action is illegal, state doesn't change.
        else:
            info["invalid_action"] = True

        # Reward is 1 for reaching the goal state, -0.001 otherwise.
        reward = 1 if new_state == self.goal_state else -0.001

        # Only the goal state is terminal.
        self.done = True if new_state == self.goal_state else False

        # Update current state.
        self.current_state = new_state

        return self.current_state, reward, self.done, info

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

    def get_successors(self, state=None, actions=None, return_probs=False):
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
        probs = []
        for action in actions:
            # With probability 0.8, the action succeeds as intended.
            successor_state = list(copy.deepcopy(state))
            source_pole, dest_pole = self.action_list[action]
            disk_to_move = min(self._disks_on_pole(source_pole, state=state))
            successor_state[disk_to_move] = dest_pole
            successor_states.append(copy.deepcopy(tuple(successor_state)))
            probs.append(0.8)

            # With probability 0.2, the action fails and a random
            # action is executed instead.
            for failure_action in self.get_available_actions(state=state):
                successor_state = list(copy.deepcopy(state))
                source_pole, dest_pole = self.action_list[failure_action]
                disk_to_move = min(self._disks_on_pole(source_pole, state=state))
                successor_state[disk_to_move] = dest_pole
                successor_states.append(copy.deepcopy(tuple(successor_state)))
                probs.append(0.2 / len(actions))

        if return_probs:
            return successor_states, probs
        else:
            return successor_states

    def generate_interaction_graph(self, directed=False, weighted=False):
        if not weighted:
            return super().generate_interaction_graph(directed)
        else:
            # Generates a list of all reachable states, starting the search from the environment's initial states.
            states = []
            current_successor_states = self.get_initial_states()

            # Brute force construction of the state-transition graph. Starts with initial
            # states, then tries to add possible successor states until no new successor states
            # can be added. This can take quite a while for environments with a large state-space.
            while not len(current_successor_states) == 0:
                next_successor_states = []
                for successor_state in current_successor_states:
                    if not successor_state in states:
                        states.append(successor_state)

                        if not self.is_state_terminal(successor_state):
                            for new_successor_state in self.get_successors(successor_state):
                                next_successor_states.append(new_successor_state)

                current_successor_states = copy.deepcopy(next_successor_states)

            # Build state-transition graph with multiple edges between each pair of nodes.
            multi_stg = nx.MultiDiGraph()
            for state in states:
                # Add node for state.
                multi_stg.add_node(state)

                # Add directed edge between node and its successors.
                successors, transition_probs = self.get_successors(state, return_probs=True)
                for i in range(len(successors)):
                    multi_stg.add_node(successors[i])
                    multi_stg.add_edge(state, successors[i], weight=transition_probs[i])

            # Convert MultiDiGraph into DiGraph with at most one edge between any two nodes.
            stg = nx.DiGraph()
            for u, v, data in multi_stg.edges(data=True):
                w = data["weight"] if "weight" in data else 1.0
                if stg.has_edge(u, v):
                    stg[u][v]["weight"] += w
                else:
                    stg.add_edge(u, v, weight=w)

            return stg

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

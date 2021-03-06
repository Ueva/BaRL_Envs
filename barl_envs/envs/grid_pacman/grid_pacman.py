import gym
import copy
import random
import numpy as np
import networkx as nx

from barl_envs.renderers import GridPacManRenderer

# Import room template files.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from . import data

with pkg_resources.path(data, "four_room.txt") as path:
    four_room_layout = path

with pkg_resources.path(data, "classic.txt") as path:
    classic_layout = path

CELL_TYPES_DICT = {".": "floor", "#": "wall", "S": "start", "G": "goal", "A": "agent", "X": "ghost"}

ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class GridPacManEnvironment(object):
    def __init__(self, pacman_template_file, movement_penalty=-1.0, goal_reward=10.0, caught_penalty=-20.0):

        # Initialise environment variables.
        self._initialise_pacman(pacman_template_file)
        self.movement_penalty = movement_penalty
        self.goal_reward = goal_reward
        self.caught_penalty = caught_penalty
        self.num_ghosts = len(self.ghost_starts)
        self.ghost_positions = [[0, 0]] * self.num_ghosts
        self.position = (0, 0)

        # Define action-space and state-space shapes.
        env_shape = gym.spaces.Discrete(self.gridworld.shape[0]), gym.spaces.Discrete(self.gridworld.shape[1])
        self.action_space = gym.spaces.Discrete(4)
        self.state_space = gym.spaces.Tuple(env_shape + env_shape * self.num_ghosts)

        self.terminal = True
        self.renderer = None

    def reset(self):
        self.ghost_positions = copy.deepcopy(self.ghost_starts)
        self.position = random.choice(self.initial_states)
        self.goal_position = random.choice(self.goal_states)

        self.current_initial_state = self.position
        self.current_goal_state = self.goal_position

        # Build the state tuple.
        state = self.position
        for ghost_position in self.ghost_positions:
            state = state + tuple(ghost_position)

        self.terminal = False

        return state

    def step(self, action):

        current_position = copy.deepcopy(self.position)
        next_position = copy.deepcopy(self.position)
        current_ghost_positions = copy.deepcopy(self.ghost_positions)
        next_ghost_positions = copy.deepcopy(self.ghost_positions)

        # Move agent.
        reward = self.movement_penalty
        if ACTIONS_DICT[action] == "DOWN":
            next_position = (next_position[0] + 1, next_position[1])
        elif ACTIONS_DICT[action] == "UP":
            next_position = (next_position[0] - 1, next_position[1])
        elif ACTIONS_DICT[action] == "RIGHT":
            next_position = (next_position[0], next_position[1] + 1)
        elif ACTIONS_DICT[action] == "LEFT":
            next_position = (next_position[0], next_position[1] - 1)

        # If agent tries to move into a wall.
        if CELL_TYPES_DICT[self.gridworld[next_position[0]][next_position[1]]] == "wall":
            next_position = current_position

        self.position = next_position

        # If the agent has reached the goal.
        if next_position == self.goal_position:
            reward += self.goal_reward
            self.terminal = True

            # Build the state tuple.
            state = next_position
            for ghost_position in current_ghost_positions:
                state = state + tuple(ghost_position)

            return state, reward, self.terminal, {}

        # Move ghosts.
        for i in range(self.num_ghosts):
            # Find shortest path from ghost position to agent position.
            ghost_position = current_ghost_positions[i]
            shortest_path = nx.shortest_path(self.level_graph, source=tuple(ghost_position), target=next_position)

            # Move ghost along path.
            if len(shortest_path) > 1:
                ghost_next_step = shortest_path[1]
                next_ghost_positions[i] = [ghost_next_step[0], ghost_next_step[1]]

        # If a ghost has caught the agent.
        if any([next_position == tuple(ghost_position) for ghost_position in next_ghost_positions]):
            self.terminal = True
            reward += self.caught_penalty

        # Build the state tuple.
        state = next_position
        for ghost_position in next_ghost_positions:
            state = state + tuple(ghost_position)

        self.ghost_positions = next_ghost_positions

        return state, reward, self.terminal, {}

    def get_available_actions(self, state=None):
        """
        Returns the set of available actions for the given state (by default, the current state).

        Keyword Arguments:
            state {(int, int)} -- The state to return available actions for. Defaults to None (uses current state).

        Returns:
            [list(int)] -- The list of actions available in the given state.
        """
        return list(range(4))

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
        return [True for i in range(4)]

    def render(self):
        if self.renderer is None:
            self.renderer = GridPacManRenderer(
                self.gridworld,
                start_state=self.current_initial_state,
                goal_state=self.current_goal_state,
            )

        self.renderer.update(
            self.position,
            self.ghost_positions,
            self.gridworld,
            start_state=self.current_initial_state,
            goal_state=self.current_goal_state,
        )

    def close(self):
        """
        Terminates all environment-related processes.
        """
        if self.renderer is not None:
            self.renderer.close()

    def is_state_terminal(self, state=None):
        if state is None:
            position = self.position
            ghost_positions = self.ghost_positions
        else:
            position = (state[0], state[1])
            ghost_positions = [state[i : i + 2] for i in range(2, len(state), 2)]

        # Is the agent at the goal state?
        is_at_goal = position == self.goal_position

        # Has the agent been caught by a ghost?
        is_caught = any([position == tuple(ghost_position) for ghost_position in ghost_positions])

        return is_at_goal or is_caught

    def get_initial_states(self):
        """
        Returns the initial state(s) for this environment.

        Returns:
            List[Tuple[int]]: The initial state(s) in this environment.
        """
        initial_states = []
        for agent_start in self.initial_states:
            state = agent_start
            for ghost_start in self.ghost_starts:
                state = state + tuple(ghost_start)
            initial_states.append(copy.deepcopy(state))

        return initial_states

    def get_successors(self, state=None):
        if state is None:
            position = self.position
            ghost_positions = self.ghost_positions
        else:
            position = (state[0], state[1])
            ghost_positions = [state[i : i + 2] for i in range(2, len(state), 2)]

        legal_actions = self.get_available_actions(state=state)

        successor_states = []
        for action in legal_actions:
            # Move agent.
            next_position = copy.deepcopy(position)
            if ACTIONS_DICT[action] == "DOWN":
                next_position = (position[0] + 1, position[1])
            elif ACTIONS_DICT[action] == "UP":
                next_position = (position[0] - 1, position[1])
            elif ACTIONS_DICT[action] == "RIGHT":
                next_position = (position[0], position[1] + 1)
            elif ACTIONS_DICT[action] == "LEFT":
                next_position = (position[0], position[1] - 1)

            # No movement if agent has moved into a wall.
            if CELL_TYPES_DICT[self.gridworld[next_position[0]][next_position[1]]] == "wall":
                next_position = copy.deepcopy(position)

            # If the agent has not reached the goal, process ghost movement.
            next_ghost_positions = copy.deepcopy(ghost_positions)

            if not position == self.goal_position:
                for i in range(self.num_ghosts):
                    # Find shortest path from ghost position to agent position.
                    ghost_position = ghost_positions[i]
                    shortest_path = nx.shortest_path(
                        self.level_graph, source=tuple(ghost_position), target=next_position
                    )

                    # Move ghost along path.
                    if len(shortest_path) > 1:
                        ghost_next_step = shortest_path[1]
                        next_ghost_positions[i] = [ghost_next_step[0], ghost_next_step[1]]

            # Build the state tuple.
            next_state = next_position
            for ghost_position in next_ghost_positions:
                next_state = next_state + tuple(ghost_position)

            successor_states.append(next_state)

        return successor_states

    def _initialise_pacman(self, pacman_template_file):
        """
        Initialises the envionment according to a given template file.

        Arguments:
            pacman_template_file {string or Path} -- Path to a gridworld template file.
        """

        # Load gridworld template file.
        self.gridworld = np.loadtxt(pacman_template_file, comments="//", dtype=str)

        # Discover initial states, goal states, and ghost starting locations.
        self.initial_states = []
        self.goal_states = []
        self.ghost_starts = []
        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):
                if CELL_TYPES_DICT[self.gridworld[y, x]] == "start":
                    self.initial_states.append((y, x))
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "goal":
                    self.goal_states.append((y, x))
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "ghost":
                    self.ghost_starts.append([y, x])

        self.level_graph = self._generate_level_graph()

    def _generate_level_graph(self):
        level_graph = nx.Graph()

        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):

                if not CELL_TYPES_DICT[self.gridworld[y, x]] == "wall":
                    # Add each non-wall node to the graph.
                    level_graph.add_node((y, x))

                    # Add an edge between each node and its non-wall neighbours.
                    for move in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
                        if not CELL_TYPES_DICT[self.gridworld[move[0], move[1]]] == "wall":
                            level_graph.add_edge((y, x), move)

        return level_graph


class PacManFourRoom(GridPacManEnvironment):
    def __init__(self, movement_penalty=-1.0, goal_reward=10.0, caught_penalty=-20.0):
        super().__init__(four_room_layout, movement_penalty, goal_reward, caught_penalty)


class PacManClassic(GridPacManEnvironment):
    pass
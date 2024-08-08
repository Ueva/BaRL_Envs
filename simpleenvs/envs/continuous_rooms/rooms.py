import math
import random
import pygame
import numpy as np
import gymnasium as gym

from simpleoptions.function_approximation import GymWrapper

from typing import Tuple

CELL_TYPES_DICT = {".": "floor", "#": "wall", "S": "start", "G": "goal", "A": "agent"}

ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class ContinuousRoomsEnvironment(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        room_template_file_path: str,
        x_lims: Tuple[float, float] = (-10.0, 10.0),
        y_lims: Tuple[float, float] = (-10.0, 10.0),
        on_dir_noise_lims: Tuple[float, float] = (-0.5, 0.5),
        off_dir_noise_lims: Tuple[float, float] = (-0.1, 0.1),
        movement_penalty: float = -0.01,
        goal_reward: float = 1.0,
        noisy_starts: bool = False,
        explorable: bool = False,
        render_mode: str = "human",
    ):
        """
        A continuous gridworld environment with continuous state and discrete action spaces.

        Args:
            room_template_file_path (str): The path to a room template file. Examples can be found in the /data/ directory.
            x_lims (Tuple[float, float], optional): The limits of the x-axis in the observation space. Defaults to (-10.0, 10.0).
            y_lims (Tuple[float, float], optional): The limits of the y-axis in the observation space. Defaults to (-10.0, 10.0).
            on_dir_noise_lims (Tuple[float, float], optional): The limits of the noise in the direction of movement. Defaults to (-0.5, 0.5).
            off_dir_noise_lims (Tuple[float, float], optional): The limits of the noise orthogonal to the direction of movement. Defaults to (-0.1, 0.1).
            movement_penalty (float, optional): The penalty for each action taken. Defaults to -0.01.
            goal_reward (float, optional): The reward for reaching a goal state. Defaults to 1.0.
            noisy_starts (bool, optional): Whether the agent starts in a random position within an initial state. Defaults to False, meaning the agent starts in the centre of an initial state.
            explorable (bool, optional): Whether the environment is explorable (i.e., whether goal states are ignored). Defaults to False.
            render_mode (str, optional): Whether to render states to the screen or return an RGB array. Defaults to "human".
        """
        super().__init__()

        # Set observation bounds.
        self.y_lims = y_lims
        self.x_lims = x_lims

        # Set noise bounds.
        self.on_dir_noise_lims = on_dir_noise_lims
        self.off_dir_noise_lims = off_dir_noise_lims

        # Set reward function.
        self.movement_penalty = movement_penalty
        self.goal_reward = goal_reward

        # Initialise gridworld based on template file.
        self._initialise_rooms(room_template_file_path, explorable)

        # Define observation and action spaces.
        self.observation_space = gym.spaces.Box(
            low=np.array([self.y_lims[0], self.x_lims[0]]),
            high=np.array([self.y_lims[1], self.x_lims[1]]),
            dtype=np.float32,
        )  # 2D continuous state space.
        self.action_space = gym.spaces.Discrete(4)  # 4 discrete actions.

        # Set initial state variables.
        self.noisy_starts = noisy_starts

        # Rendering variables.
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _initialise_rooms(self, room_template_file_path, explorable):
        # Load gridworld template file.
        self.gridworld = np.loadtxt(room_template_file_path, comments="//", dtype=str)

        self.y_cells = self.gridworld.shape[0]
        self.x_cells = self.gridworld.shape[1]

        # Define mapping from cell-space to observation-space.
        self.y_interp = lambda y: (y - 0) * (self.y_lims[1] - self.y_lims[0]) / (self.y_cells - 0) + self.y_lims[0]
        self.x_interp = lambda x: (x - 0) * (self.x_lims[1] - self.x_lims[0]) / (self.x_cells - 0) + self.x_lims[0]

        # Define mapping from observation-space to cell-space.
        self.y_interp_inv = lambda y: (y - self.y_lims[0]) * (self.y_cells - 0) / (self.y_lims[1] - self.y_lims[0]) + 0
        self.x_interp_inv = lambda x: (x - self.x_lims[0]) * (self.x_cells - 0) / (self.x_lims[1] - self.x_lims[0]) + 0

        # Discover start and goal states.
        self.initial_states = []
        self.terminal_states = []
        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):
                if CELL_TYPES_DICT[self.gridworld[y, x]] == "start":
                    self.initial_states.append((y, x))
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "goal" and not explorable:
                    self.terminal_states.append((y, x))

    def reset(self, state=None):
        # It no initial state is specified, randomly select one.
        if state is None:
            # Randomly select an initial state.
            initial_grid_square = random.choice(self.initial_states)

            # If noisy starts are enabled, start in a random position within the initial state.
            if self.noisy_starts:
                initial_y = random.uniform(initial_grid_square[0], initial_grid_square[0] + 1)
                initial_x = random.uniform(initial_grid_square[1], initial_grid_square[1] + 1)
                self.current_state = (initial_y, initial_x)
            # Otherwise, start in the centre of the initial state.
            else:
                self.current_state = (initial_grid_square[0] + 0.5, initial_grid_square[1] + 0.5)
        # Otherwise, use the specified initial state.
        else:
            self.current_state = state

        return self._get_observation(), {}

    def step(self, action, state=None):

        if state is None:
            state = self.current_state

        current_state = self.current_state
        next_state = self.current_state

        noise_on_dir = random.uniform(self.on_dir_noise_lims[0], self.on_dir_noise_lims[1])
        noise_off_dir = random.uniform(self.off_dir_noise_lims[0], self.off_dir_noise_lims[1])

        # Move the agent.
        if action == 0:  # UP
            next_state = (current_state[0] - 1 - noise_on_dir, current_state[1] + noise_off_dir)
        elif action == 1:  # DOWN
            next_state = (current_state[0] + 1 + noise_on_dir, current_state[1] + noise_off_dir)
        elif action == 2:  # LEFT
            next_state = (current_state[0] + noise_off_dir, current_state[1] - 1 - noise_on_dir)
        elif action == 3:  # RIGHT
            next_state = (current_state[0] + noise_off_dir, current_state[1] + 1 + noise_on_dir)

        reward = self.movement_penalty
        terminal = False

        # Ensure the agent stays within the gridworld. If they have moved outside, reset to the current state.
        if next_state[0] < 0 or next_state[0] >= self.y_cells or next_state[1] < 0 or next_state[1] >= self.x_cells:
            next_state = current_state
        # Ensure that the agent does not move into a wall. If it does, reset to the current state.
        elif CELL_TYPES_DICT[self.gridworld[math.floor(next_state[0]), math.floor(next_state[1])]] == "wall":
            next_state = current_state
        # If the agent moves more than one cell in a single step, we need to check whether it has "jumped" over a wall.
        else:
            # Calculate the midpoint between the current and next state.
            midpoint = (
                (current_state[0] + next_state[0]) / 2,
                (current_state[1] + next_state[1]) / 2,
            )
            # If the midpoint is a wall, it means the agent has jumped over a wall. So, reset to the current state.
            if CELL_TYPES_DICT[self.gridworld[math.floor(midpoint[0]), math.floor(midpoint[1])]] == "wall":
                next_state = current_state

        # If the agent has reached the goal, give a reward and set terminal to True.
        if CELL_TYPES_DICT[self.gridworld[math.floor(next_state[0]), math.floor(next_state[1])]] == "goal":
            reward += self.goal_reward
            terminal = True

        self.current_state = next_state

        return self._get_observation(), reward, terminal, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            self._render_frame()

    def _render_frame(self):
        tile_size = 32
        agent_size = 8

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.x_cells * tile_size, self.y_cells * tile_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.x_cells * tile_size, self.y_cells * tile_size))
        canvas.fill((255, 255, 255))

        # Draw Gridworld.
        for x in range(self.x_cells):
            for y in range(self.y_cells):
                if CELL_TYPES_DICT[self.gridworld[y, x]] == "wall":
                    pygame.draw.rect(canvas, (0, 0, 0), (x * tile_size, y * tile_size, tile_size, tile_size))
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "start":
                    pygame.draw.rect(canvas, (0, 255, 0), (x * tile_size, y * tile_size, tile_size, tile_size))
                elif CELL_TYPES_DICT[self.gridworld[y, x]] == "goal":
                    pygame.draw.rect(canvas, (255, 0, 0), (x * tile_size, y * tile_size, tile_size, tile_size))

        # Draw Agent.
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int(self.current_state[1] * tile_size), int(self.current_state[0] * tile_size)),
            agent_size,
        )

        # If rendering for a human, render to screen.
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            pygame.time.wait(10)
        # Else, return an rgb array
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_observation(self, state=None):
        """
        Converts from a state (coordinates in the cell-space) to an observation (coordinates in the range -10 to +10).
        """
        if state is None:
            state = self.current_state

        return np.array([self.y_interp(state[0]), self.x_interp(state[1])], dtype=np.float32)

    def _get_cell(self, observation):
        """
        Converts from an observation (coordinates in the range -10 to +10) to a state (coordinates in the cell-space).
        """
        return (self.y_interp_inv(observation[0]), self.x_interp_inv(observation[1]))


# Import room template files.
from importlib.resources import files

from . import data

xu_four_rooms = files(data).joinpath("xu_four_rooms.txt")
empty_rooms = files(data).joinpath("empty_rooms.txt")


class ContinuousFourRooms(ContinuousRoomsEnvironment):
    def __init__(
        self,
        x_lims: Tuple[float, float] = (-10.0, 10.0),
        y_lims: Tuple[float, float] = (-10.0, 10.0),
        on_dir_noise_lims: Tuple[float, float] = (-0.5, 0.5),
        off_dir_noise_lims: Tuple[float, float] = (-0.1, 0.1),
        movement_penalty: float = -0.01,
        goal_reward: float = 1.0,
        noisy_starts: bool = False,
        explorable: bool = False,
        render_mode: str = "human",
    ):
        super().__init__(
            room_template_file_path=xu_four_rooms,
            x_lims=x_lims,
            y_lims=y_lims,
            on_dir_noise_lims=on_dir_noise_lims,
            off_dir_noise_lims=off_dir_noise_lims,
            movement_penalty=movement_penalty,
            goal_reward=goal_reward,
            noisy_starts=noisy_starts,
            explorable=explorable,
            render_mode=render_mode,
        )


class ContinuousEmptyRooms(ContinuousRoomsEnvironment):
    def __init__(
        self,
        x_lims: Tuple[float, float] = (-10.0, 10.0),
        y_lims: Tuple[float, float] = (-10.0, 10.0),
        on_dir_noise_lims: Tuple[float, float] = (-0.5, 0.5),
        off_dir_noise_lims: Tuple[float, float] = (-0.1, 0.1),
        movement_penalty: float = -0.01,
        goal_reward: float = 1.0,
        noisy_starts: bool = False,
        explorable: bool = False,
        render_mode: str = "human",
    ):
        super().__init__(
            room_template_file_path=empty_rooms,
            x_lims=x_lims,
            y_lims=y_lims,
            on_dir_noise_lims=on_dir_noise_lims,
            off_dir_noise_lims=off_dir_noise_lims,
            movement_penalty=movement_penalty,
            goal_reward=goal_reward,
            noisy_starts=noisy_starts,
            explorable=explorable,
            render_mode=render_mode,
        )

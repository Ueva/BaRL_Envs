import math
import random
import pygame
import numpy as np
import gymnasium as gym

from simpleoptions.function_approximation import GymWrapper


CELL_TYPES_DICT = {".": "floor", "#": "wall", "S": "start", "G": "goal", "A": "agent"}

ACTIONS_DICT = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class ContinuousRoomsEnvironment(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, room_template_file_path, explorable=False, render_mode="human"):
        super().__init__()

        # Initialise gridworld based on template file.
        self._initialise_rooms(room_template_file_path, explorable)

        # Define observation and action spaces.
        self.observation_space = gym.spaces.Box(
            low=np.array([-10.0, -10.0]), high=np.array([10.0, 10.0]), dtype=np.float32
        )  # 2D continuous state space.
        self.action_space = gym.spaces.Discrete(4)  # 4 discrete actions.

        # Rendering variables.
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _initialise_rooms(self, room_template_file_path, explorable):
        # Load gridworld template file.
        self.gridworld = np.loadtxt(room_template_file_path, comments="//", dtype=str)

        self.y_cells = self.gridworld.shape[0]
        self.x_cells = self.gridworld.shape[1]

        # Base Python version (fastest):
        self.y_interp = lambda y: (y - 0) * (10.0 - -10.0) / (self.y_cells + 1 - 0) + -10.0
        self.x_interp = lambda x: (x - 0) * (10.0 - -10.0) / (self.x_cells + 1 - 0) + -10.0

        # NumPy version (slower):
        # self.y_interp = lambda y: np.interp(y, [0, self.y_cells + 1], [-10.0, 10.0])
        # self.x_interp = lambda x: np.interp(x, [0, self.x_cells + 1], [-10.0, 10.0])

        # SciPy version (slowest):
        # self.y_interp = scipy.interpolate.interp1d([0, self.y_cells + 1], [-10.0, 10.0])
        # self.x_interp = scipy.interpolate.interp1d([0, self.x_cells + 1], [-10.0, 10.0])

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
        if state is None:
            initial_grid_square = random.choice(self.initial_states)
            initial_y = random.uniform(initial_grid_square[0], initial_grid_square[0] + 1)
            initial_x = random.uniform(initial_grid_square[1], initial_grid_square[1] + 1)
            self.current_state = (initial_y, initial_x)
        else:
            self.current_state = state

        return self._get_obs(), {}

    def step(self, action):

        current_state = self.current_state
        next_state = self.current_state
        noise_on_dir = random.uniform(-0.3, 0.0)
        noise_off_dir = random.uniform(-0.1, 0.1)

        # Move the agent.
        if action == 0:  # UP
            next_state = (current_state[0] - 1 - noise_on_dir, current_state[1] + noise_off_dir)
        elif action == 1:  # DOWN
            next_state = (current_state[0] + 1 + noise_on_dir, current_state[1] + noise_off_dir)
        elif action == 2:  # LEFT
            next_state = (current_state[0] + noise_off_dir, current_state[1] - 1 - noise_on_dir)
        elif action == 3:  # RIGHT
            next_state = (current_state[0] + noise_off_dir, current_state[1] + 1 + noise_on_dir)

        reward = 0.0
        terminal = False

        # Determine whether next-state is valid and/or a terminal state.
        if CELL_TYPES_DICT[self.gridworld[math.floor(next_state[0]), math.floor(next_state[1])]] == "wall":
            next_state = current_state
        elif CELL_TYPES_DICT[self.gridworld[math.floor(next_state[0]), math.floor(next_state[1])]] == "goal":
            reward = 1.0
            terminal = True

        reward += -0.01

        self.current_state = next_state

        return self._get_obs(), reward, terminal, False, {}

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

    def _get_obs(self):
        return np.array([self.y_interp(self.current_state[0]), self.x_interp(self.current_state[1])], dtype=np.float32)

    def _get_cell(self):
        return np.array([self.current_state[0], self.current_state[1]], dtype=np.int32)


try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from . import data

with pkg_resources.path(data, "xu_four_rooms.txt") as path:
    xu_four_rooms = path

with pkg_resources.path(data, "empty_rooms.txt") as path:
    empty_rooms = path


class ContinuousFourRooms(ContinuousRoomsEnvironment):
    def __init__(self, explorable=False, render_mode="human"):
        super().__init__(xu_four_rooms, explorable, render_mode)


class ContinuousEmptyRooms(ContinuousRoomsEnvironment):
    def __init__(self, explorable=False, render_mode="human"):
        super().__init__(empty_rooms, explorable, render_mode)


if __name__ == "__main__":

    env = ContinuousRoomsEnvironment("simpleenvs/envs/continuous_rooms/data/xu_four_rooms.txt", render_mode="human")
    env = GymWrapper(env)

    # i = 0

    # while i < 1_000_000:

    #     state, _ = env.reset()
    #     terminal = False

    #     while not terminal:
    #         action = env.action_space.sample()
    #         next_state, reward, terminal, _, _ = env.step(action)
    #         # print("{}:\t{}, {} --> {}".format(i, state, ACTIONS_DICT[action], next_state))
    #         state = next_state
    #         i += 1

    #         if i > 1_000_000:
    #             break

    # env.close()

    for _ in range(1000):

        state, _ = env.reset()
        terminal = False

        i = 0
        while not terminal:
            # action = env.action_space.sample()
            action = 1
            next_state, reward, terminal, _, _ = env.step(action)
            # print("{}:\t{}, {} --> {}".format(i, state, ACTIONS_DICT[action], next_state))
            env.render()
            state = next_state
            i += 1
    env.close()

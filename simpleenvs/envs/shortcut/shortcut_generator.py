import random
import pygame

import numpy as np
import numpy.typing as npt
import networkx as nx

from typing import List, Tuple

# This class generates a "shortcut" world.
# The world consists of a single grid. Each grid cell is connected to its four neighbours (up, down, left, right), unless it is blocked.
# The agent must navigate from some start cell to some target cell on the grid.
# Some positions on the grid are designated as "shortcuts". Each shortcut is connected to other shortcuts on the grid, allowing for immediate (but
# more costly) movement between them. There can be multiple levels of shortcut actions, with each level allowing movement between more distant shortcuts
# than the last (but costing more to use).
# Moving aroudn the grid using shortcuts can be thought of as moving around a city using different modes of transport:
#    - Moving around the grid using primitive actions can be thought of as walking,
#    - moving around the grid using the level shortcut actions can be thought of as taking the bus, and
#    - moving around the grid using the second level shortcut actions can be thought of as taking a plane.
# The grid may not always be regular, and may contain obstacles that block cells. A parameter controls the probability of a cell being blocked.
# A grid (i.e., a binary matrix) can be passed in, or a random grid of a specific height and width can be generated.


class ShortcutGenerator:
    def __init__(self):
        """
        Initializes the SingleLevelShortcutGenerator.
        """
        self.grid_height: int = None
        self.grid_width: int = None
        self.blocker_prob: float = None
        self.desired_walkability: float = None
        self.num_shortcut_hubs: int = None
        self.shortcut_hubs: List[Tuple[Tuple[int, int], List[int]]] = None
        self.shortcut_connections: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = None
        self.shortcut_hub_radii: List[int] = None
        self.shortcut_hub_costs: List[float] = None

    def generate_grid(
        self, grid_width: int, grid_height: int, blocker_prob: float, desired_walkability: float
    ) -> npt.NDArray[np.bool_]:
        # Store the parameters for later use.
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.blocker_prob = blocker_prob
        self.desired_walkability = desired_walkability

        found_valid_grid = False
        while not found_valid_grid:
            grid = np.random.rand(self.grid_height, self.grid_width) > self.blocker_prob

            # Dilate the cells to form larger, smoother connected regions.
            grid = self._ca_grid(grid)

            # Isolate the largest connected region.
            grid, largest_region_size = self._isolate_largest_component(grid)

            if largest_region_size / (self.grid_height * self.grid_width) >= desired_walkability:
                found_valid_grid = True

        self.grid = grid

        return grid

    def regenerate_grid(self):
        """
        Convenience method to regenerate the grid using the currently stored parameters.

        Raises:
            ValueError: Raised if `.generate_grid()` has not been called yet (i.e., there are no stored parameters to re-generate the grid from).

        See also:
            `.generate_grid()`
        """
        # Can only be called after a .generate_grid() has already been called.
        if (
            self.grid_height is None
            or self.grid_width is None
            or self.blocker_prob is None
            or self.desired_walkability is None
        ):
            raise ValueError("Please call .generate_grid() before calling .regenerate_grid().")

        # Regenerate the grid using the currently_stored parameters.
        self.grid = self.generate_grid(self.grid_width, self.grid_height, self.blocker_prob, self.desired_walkability)

        return self.grid

    def set_grid(self, grid):
        self.grid = grid
        self.grid_height = grid.shape[0]
        self.grid_width = grid.shape[1]

    def generate_shortcut_hubs(self, num_shortcut_hubs):
        pass  # TODO: Implement this function.

    def set_shortcut_hubs(self, shortcut_hubs):
        self.shortcut_hubs = shortcut_hubs

    def generate_shortcuts(self, shortcut_hub_radii):
        pass  # TODO: Implement this function.

    def render(self):
        # Set up the pygame window.
        pygame.init()
        pygame.display.set_caption("Shortcut World")

        # Set up some colours.
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREY = (128, 128, 128)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)

        # Set up some parameters for rendering.
        cell_size = 8
        grid_offset = 32
        fps = 165

        width = self.grid_width * cell_size + 2 * grid_offset
        height = self.grid_height * cell_size + 2 * grid_offset

        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        # Wait for the user to close the window.
        running = True
        while running:
            # Limit the frame rate.
            clock.tick(fps)

            # The background should be grey.
            screen.fill(GREY)

            # Draw the grid cells, centred on the screen.
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    if self.grid[y, x]:
                        pygame.draw.rect(
                            screen,
                            WHITE,
                            pygame.Rect(
                                grid_offset + x * cell_size,
                                grid_offset + y * cell_size,
                                cell_size,
                                cell_size,
                            ),
                        )
                    else:
                        pygame.draw.rect(
                            screen,
                            BLACK,
                            pygame.Rect(
                                grid_offset + x * cell_size,
                                grid_offset + y * cell_size,
                                cell_size,
                                cell_size,
                            ),
                        )

            # Update the display.
            pygame.display.update()

            # Process events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # If the user presses escape, stop the loop.
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # If the user presses the R key, regenerate the grid.
                    elif event.key == pygame.K_r:
                        self.grid = self.regenerate_grid()

                # If the user closes the window, stop the loop.
                if event.type == pygame.QUIT:
                    running = False

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

        # def _process_grid(self, grid, num_iterations=5):

    def _ca_grid(self, grid, num_iterations=20):
        # This function implements a cellular automaton to dilate the cells of the grid.
        # We use the following rules:
        # - a tile becomes a wall if it was a wall and 4 or more of its eight neighbors were walls,
        # - or if it was not a wall and 5 or more neighbors were.
        # Put more succinctly, a tile is a wall if the 3x3 region centered on it contained at least 5 walls.
        # Walls are False (0) and open tiles are True (1). You need to take this into account in calculating the number of walls.
        # We implement this in as efficient a way as possible, making use of vectorisation where we can.

        for _ in range(num_iterations):
            # Count the number of walls in the 3x3 region around each cell.
            num_walls = np.zeros_like(grid, dtype=int)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    num_walls += np.roll(grid, (dy, dx), axis=(0, 1))

            # Update the grid according to the rules.
            grid = np.logical_or(
                np.logical_and(grid, num_walls >= 4), np.logical_and(np.logical_not(grid), num_walls >= 5)
            )

        return grid

    def _isolate_largest_component(self, grid):
        # return grid, self.grid_width * self.grid_height
        # Use flood fill to find all passable regions of the grid.
        regions = []
        visited = np.zeros_like(grid, dtype=bool)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if grid[y, x] and not visited[y, x]:
                    region = self._flood_fill(grid, visited, x, y)
                    regions.append(region)

        # Find the largest region.
        largest_region = max(regions, key=len)

        # Fill in all other regions.
        for region in regions:
            if region is not largest_region:
                for y, x in region:
                    grid[y, x] = False

        return grid, len(largest_region)

    def _flood_fill(self, grid, visited, x, y):
        region = []
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            region.append((y, x))
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height and grid[new_y, new_x]:
                    stack.append((new_x, new_y))

        return region


if __name__ == "__main__":
    generator = ShortcutGenerator()
    generator.seed(0)

    generator.generate_grid(grid_height=100, grid_width=100, blocker_prob=0.475, desired_walkability=0.6)
    generator.generate_shortcut_hubs(num_shortcut_hubs=8)
    generator.generate_shortcuts(shortcut_hub_radii=[2, 4, 6])

    generator.render()

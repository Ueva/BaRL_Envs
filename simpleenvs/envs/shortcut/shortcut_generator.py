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
    def __init__(
        self,
        grid_height: int = None,
        grid_width: int = None,
        blocker_prob: float = 0.0,
        desired_walkability: float = 0.5,
        grid: npt.NDArray[np.bool_] = None,
        num_shortcut_hubs: int = None,
        shortcut_hubs: List[Tuple[int, int]] = None,
        shortcut_connections: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = None,
        shortcut_hub_radii: List[int] = None,
        shortcut_hub_costs: List[float] = None,
    ):
        """
        Initializes the SingleLevelShortcutGenerator with the given parameters.

        Args:
            grid (npt.NDArray[np.bool_], optional): A predefined grid to use. Defaults to None.
            grid_height (int, optional): The height of the grid. Defaults to None.
            grid_width (int, optional): The width of the grid. Defaults to None.
            blocker_prob (float, optional): The probability of a cell being blocked. Defaults to 0.0.
            desired_walkability (float, optional): The minimum proportion of cells that should be walkable. Defaults to 0.5.
            shortcut_hubs (List[Tuple[int, int]], optional): A list of predefined shortcut hubs. Defaults to None.
            num_shortcut_hubs (int, optional): The number of shortcut hubs to generate. Defaults to None.
            shortcut_connections (List[Tuple[int, Tuple[int, int], Tuple[int, int]]], optional): A list of predefined shortcut connections of the form (level, (source coord), (target coord)). Defaults to None.
            shortcut_hub_radii (List[int], optional): A list of radii for each shortcut hub. Defaults to None.
            shortcut_hub_costs (List[float], optional): A list of costs for using each shortcut hub. Defaults to None.

        Raises:
            ValueError: If both a grid and grid dimensions are provided.
            ValueError: If neither a grid nor grid dimensions are provided.
            ValueError: If both shortcut hubs and the number of shortcut hubs are provided.
            ValueError: If neither shortcut hubs nor the number of shortcut hubs are provided.
            ValueError: If both shortcut connections and shortcut hub radii are provided.
            ValueError: If neither shortcut connections nor shortcut hub radii are provided.
            ValueError: If shortcut hub costs are not provided.

        """

        # If a grid has been provided, use it.
        if grid is not None:
            if grid_height is not None or grid_width is not None:
                raise ValueError("Please specify either a grid to use, or parameters to generate a grid.")

            self.grid_height = grid.shape[0]
            self.grid_width = grid.shape[1]
            self.grid = grid
        # Otherwise, generate a grid.
        else:
            if grid_height is None or grid_width is None:
                raise ValueError("Please specify either a grid to use, or parameters to generate a grid.")

            self.grid_height = grid_height
            self.grid_width = grid_width
            self.blocker_prob = blocker_prob
            self.desired_walkability = desired_walkability
            self.grid = self.generate_grid(grid_width, grid_height, blocker_prob, desired_walkability)

        # If shortcut hubs have been specified, use them.
        if shortcut_hubs is not None:
            if num_shortcut_hubs is not None:
                raise ValueError(
                    "Please specify either a list of shortcut hubs to use, or parameters to generate shortcut hubs."
                )
            self.shortcut_hubs = shortcut_hubs
            self.num_shortcut_hubs = len(shortcut_hubs)
        # Otherwise, generate shortcut hubs.
        else:
            if num_shortcut_hubs is None:
                raise ValueError(
                    "Please specify either a list of shortcut hubs to use, or specify how many to generate."
                )
            self.num_shortcut_hubs = num_shortcut_hubs
            self.shortcut_hubs = self.generate_shortcut_hubs(num_shortcut_hubs)

        # If shortcut connections have been specified, use them.
        if shortcut_connections is not None:
            if shortcut_hub_radii is not None:
                raise ValueError(
                    "Please specify either a list of shortcut connections to use, or parameters to generate shortcut connections."
                )
            self.shortcut_connections = shortcut_connections
        # Otherwise, generate shortcut connections.
        else:
            if shortcut_hub_radii is None:
                raise ValueError(
                    "Please specify either a list of shortcut connections to use, or parameters to generate shortcut connections."
                )
            self.shortcut_hub_radii = shortcut_hub_radii
            self.shortcuts = self.generate_shortcuts(shortcut_hub_radii)

        if shortcut_hub_costs is None:
            ValueError("Please specify the costs of using each level of shortcut actions.")
            self.shortcut_hub_costs = shortcut_hub_costs

    def generate_grid(
        self, grid_width: int, grid_height: int, blocker_prob: float, desired_walkability: float
    ) -> npt.NDArray[np.bool_]:
        found_valid_grid = False
        while not found_valid_grid:
            grid = np.random.rand(self.grid_height, self.grid_width) > self.blocker_prob

            # Dilate the cells to form larger, smoother connected regions.
            grid = self._dilate_grid(grid, num_iterations=20)

            # Isolate the largest connected region.
            grid, largest_region_size = self._isolate_largest_component(grid)

            if largest_region_size / (self.grid_height * self.grid_width) >= desired_walkability:
                found_valid_grid = True

        return grid

    def generate_shortcut_hubs(self, num_shortcut_hubs):
        pass  # TODO: Implement this function.

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
                        self.grid = self.generate_grid(
                            self.grid_width,
                            self.grid_height,
                            self.blocker_prob,
                            self.desired_walkability,
                        )

                # If the user closes the window, stop the loop.
                if event.type == pygame.QUIT:
                    running = False

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _dilate_grid(self, grid, num_iterations=20):
        for _ in range(20):
            for y in range(1, self.grid_height - 1):
                for x in range(1, self.grid_width - 1):
                    if np.sum(grid[y - 1 : y + 2, x - 1 : x + 2]) >= 5:
                        grid[y, x] = True
        return grid

    def _isolate_largest_component(self, grid):
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
    generator = ShortcutGenerator(
        grid_height=100,
        grid_width=100,
        blocker_prob=0.65,
        desired_walkability=0.6,
        num_shortcut_hubs=3,
        shortcut_hub_radii=[2, 4, 6],
        shortcut_hub_costs=[-1.0, -5.0, -10.0],
    )
    generator.render()

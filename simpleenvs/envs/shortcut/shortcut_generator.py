import scipy
import random
import pygame
import distinctipy

import numpy as np
import numpy.typing as npt
import networkx as nx

from typing import List, Tuple


import scipy.spatial

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
        self.shortcut_hubs: List[Tuple[Tuple[int, int]]] = None
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

        if self.shortcut_hubs is not None:
            self.regenerate_shortcut_hubs()

        return self.grid

    def set_grid(self, grid):
        self.grid = grid
        self.grid_height = grid.shape[0]
        self.grid_width = grid.shape[1]

    def generate_shortcut_hubs(self, num_shortcut_hubs, num_iterations=5):
        # Store the number of shortcut hubs for later use.
        self.num_shortcut_hubs = num_shortcut_hubs

        # Create an array of walkable cells.
        walkable_cells = np.array(
            [(y, x) for y in range(self.grid_height) for x in range(self.grid_width) if self.grid[y, x]]
        )
        num_cells = len(walkable_cells)

        # Build the graph of walkable cells.
        G = nx.Graph()
        for y, x in walkable_cells:
            G.add_node((y, x))
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                n_y, n_x = y + dy, x + dx
                if 0 <= n_y < self.grid_height and 0 <= n_x < self.grid_width and self.grid[n_y, n_x]:
                    G.add_edge((y, x), (n_y, n_x))

        # Randomly initialise N shortcut hubs.
        rand_indices = np.random.choice(num_cells, size=num_shortcut_hubs, replace=False)
        hubs = walkable_cells[rand_indices]

        for _ in range(num_iterations):
            # Assign each cell to the nearest hub using shortest path distance.
            distances = np.zeros((num_cells, num_shortcut_hubs))
            for i, hub in enumerate(hubs):
                lengths = nx.single_source_shortest_path_length(G, tuple(hub))
                for j, cell in enumerate(walkable_cells):
                    distances[j, i] = lengths.get(tuple(cell), np.inf)
            assignments = np.argmin(distances, axis=1)

            # Update the hubs to be the centroids of the cells assigned to them.
            new_hubs = []
            for i in range(num_shortcut_hubs):
                assigned_cells = walkable_cells[assignments == i]
                if len(assigned_cells) > 0:
                    centroid = np.mean(assigned_cells, axis=0)
                    closest_idx = np.argmin(np.linalg.norm(assigned_cells - centroid, axis=1))
                    new_hubs.append(assigned_cells[closest_idx])
                else:
                    new_hubs.append(walkable_cells[np.random.choice(num_cells)])

            hubs = np.array(new_hubs)

        self.shortcut_hubs = [tuple(hub) for hub in hubs]

        ### Store a few bits of information for rendering. ###
        # Create a dictionary mapping each hub to a colour.
        self.hub_colours = {
            hub: colour
            for hub, colour in zip(
                self.shortcut_hubs,
                distinctipy.get_colors(
                    num_shortcut_hubs,
                    exclude_colors=[(0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                ),
            )
        }

        # Create a dictionary mapping each walkable cell to the closest hub.
        distances = np.zeros((num_cells, num_shortcut_hubs))
        for i, hub in enumerate(hubs):
            lengths = nx.single_source_shortest_path_length(G, tuple(hub))
            for j, cell in enumerate(walkable_cells):
                distances[j, i] = lengths.get(tuple(cell), np.inf)
        assignments = np.argmin(distances, axis=1)
        self.cell_to_hub = {
            tuple(cell): self.shortcut_hubs[hub_idx] for cell, hub_idx in zip(walkable_cells, assignments)
        }

        return self.shortcut_hubs

    def regenerate_shortcut_hubs(self):
        """
        Convenience method to regenerate the shortcut hubs using the currently stored parameters.

        Raises:
            ValueError: Raised if `.generate_shortcut_hubs()` has not been called yet (i.e., there are no stored parameters to re-generate the shortcut hubs from).

        See also:
            `.generate_shortcut_hubs()`
        """
        # Can only be called after a .generate_shortcut_hubs() has already been called.
        if self.num_shortcut_hubs is None:
            raise ValueError("Please call .generate_shortcut_hubs() before calling .regenerate_shortcut_hubs().")

        # Regenerate the shortcut hubs using the currently stored parameters.
        self.shortcut_hubs = self.generate_shortcut_hubs(self.num_shortcut_hubs)

        return self.shortcut_hubs

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

            draw_hub_regions = pygame.key.get_pressed()[pygame.K_LSHIFT]

            # Draw the grid cells, centred on the screen.
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    if self.grid[y, x]:
                        if not draw_hub_regions:
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
                            hub_colour = self.hub_colours[self.cell_to_hub[(y, x)]]
                            pygame.draw.rect(
                                screen,
                                self._colour_float_to_int(hub_colour),
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

            # Highlight shortcut hubs in red.
            for hub in self.shortcut_hubs:
                pygame.draw.rect(
                    screen,
                    RED,
                    pygame.Rect(
                        grid_offset + hub[1] * cell_size,
                        grid_offset + hub[0] * cell_size,
                        cell_size,
                        cell_size,
                    ),
                )

            # Process events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # If the user presses escape, stop the loop.
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # If the user presses the R key, regenerate the grid.
                    elif event.key == pygame.K_r:
                        self.grid = self.regenerate_grid()
                    # If the user preses the H key, regenerate the shortcut hubs.
                    elif event.key == pygame.K_h:
                        self.shortcut_hubs = self.generate_shortcut_hubs(self.num_shortcut_hubs)

                # If the user closes the window, stop the loop.
                if event.type == pygame.QUIT:
                    running = False

            # Update the display.
            pygame.display.update()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

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
                    region = self._flood_fill(grid, visited, y, x)
                    regions.append(region)

        # Find the largest region.
        largest_region = max(regions, key=len)

        # Fill in all other regions.
        for region in regions:
            if region is not largest_region:
                for y, x in region:
                    grid[y, x] = False

        return grid, len(largest_region)

    def _flood_fill(self, grid, visited, y, x):
        region = []
        stack = [(y, x)]
        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            region.append((y, x))
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_y, new_x = y + dy, x + dx
                if 0 <= new_y < self.grid_height and 0 <= new_x < self.grid_width and grid[new_y, new_x]:
                    stack.append((new_y, new_x))

        return region

    def _colour_float_to_int(self, colour):
        return tuple(int(255 * c) for c in colour)

    def _colour_int_to_float(self, colour):
        return tuple(c / 255 for c in colour)


if __name__ == "__main__":
    generator = ShortcutGenerator()
    generator.seed(0)

    generator.generate_grid(grid_height=100, grid_width=150, blocker_prob=0.475, desired_walkability=0.6)
    generator.generate_shortcut_hubs(num_shortcut_hubs=11)
    generator.generate_shortcuts(shortcut_hub_radii=[2, 4, 6])

    generator.render()

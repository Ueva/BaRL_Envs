import random
import pygame
import itertools
import distinctipy

import numpy as np
import numpy.typing as npt
import igraph as ig

from typing import List, Tuple

# This class generates a "shortcut" world.
# The world consists of a single grid. Each grid cell is connected to its four neighbours (up, down, left, right), unless it is blocked.
# The agent must navigate from some start cell to some target cell on the grid.
# Some positions on the grid are designated as "shortcuts". Each shortcut is connected to other shortcuts on the grid, allowing for immediate (but
# more costly) movement between them. There can be multiple levels of shortcut actions, with each level allowing movement between more distant shortcuts
# than the last (but costing more to use).
# Moving around the grid using shortcuts can be thought of as moving around a city using different modes of transport:
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
        self.shortcut_hubs: List[Tuple[int, int]] = None
        self.shortcut_connections: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = None
        self.shortcut_hub_radii: List[int] = None

    def generate_grid(
        self, grid_height: int, grid_width: int, blocker_prob: float, desired_walkability: float
    ) -> npt.NDArray[np.bool_]:
        """
        Generates a grid of a specific height and width.

        Args:
            grid_height (int): _description_
            grid_width (int): The width of the grid.
            blocker_prob (float): _description_
            desired_walkability (float): _description_

        Returns:
            npt.NDArray[np.bool_]: _description_
        """
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

    def regenerate_grid(self) -> npt.NDArray[np.bool_]:
        """
        Convenience method to regenerate the grid using the currently stored parameters.

        Returns:
            npt.NDArray[np.bool_]: The new grid.

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
        self.grid = self.generate_grid(self.grid_height, self.grid_width, self.blocker_prob, self.desired_walkability)

        if self.shortcut_hubs is not None:
            self.regenerate_shortcut_hubs()

        return self.grid

    def set_grid(self, grid: npt.NDArray[np.bool_]):
        """
        Stores a pre-defined grid for use in the generator.

        Args:
            grid (npt.NDArray[np.bool_]): The grid to store.
        """
        self.grid = grid
        self.grid_height = grid.shape[0]
        self.grid_width = grid.shape[1]

    def generate_shortcut_hubs(self, num_shortcut_hubs: int, num_iterations: int = 10) -> List[Tuple[int, int]]:
        """
        Generates a set of shortcut hubs on the grid.

        Args:
            num_shortcut_hubs (int): The number of shortcut hubs to generate.
            num_iterations (int, optional): The number of iterations to use in the k-means algorithm. Defaults to 10.

        Returns:
            List[Tuple[int, int]]: The list of cells selected as shortcut hubs.
        """
        # Store the number of shortcut hubs for later use.
        self.num_shortcut_hubs = num_shortcut_hubs

        # Create an array of walkable cells.
        walkable_cells = np.argwhere(self.grid)
        num_cells = len(walkable_cells)

        # Build the graph of walkable cells.
        edges = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x]:
                    if y + 1 < self.grid_height and self.grid[y + 1, x]:
                        edges.append((y * self.grid_width + x, (y + 1) * self.grid_width + x))
                    if x + 1 < self.grid_width and self.grid[y, x + 1]:
                        edges.append((y * self.grid_width + x, y * self.grid_width + x + 1))

        graph = ig.Graph(edges=edges, directed=False)

        # Randomly initialise N shortcut hubs.
        rand_indices = np.random.choice(num_cells, size=num_shortcut_hubs, replace=False)
        hubs = walkable_cells[rand_indices]

        for _ in range(num_iterations):
            # Assign each cell to the nearest hub using shortest path distance.
            distances = np.zeros((num_cells, num_shortcut_hubs))
            for i, hub in enumerate(hubs):
                lengths = np.array(graph.distances(source=hub[0] * self.grid_width + hub[1])[0]).flatten()
                distances[:, i] = lengths[walkable_cells[:, 0] * self.grid_width + walkable_cells[:, 1]]
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

        ### Store a few pieces of information for rendering. ###
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
            lengths = np.array(graph.distances(source=hub[0] * self.grid_width + hub[1])[0]).flatten()
            distances[:, i] = lengths[walkable_cells[:, 0] * self.grid_width + walkable_cells[:, 1]]
        assignments = np.argmin(distances, axis=1)
        self.cell_to_hub = {
            tuple(cell): self.shortcut_hubs[hub_idx] for cell, hub_idx in zip(walkable_cells, assignments)
        }

        return self.shortcut_hubs

    def regenerate_shortcut_hubs(self) -> List[Tuple[int, int]]:
        """
        Convenience method to regenerate the shortcut hubs using the currently stored parameters.

        Returns:
            List[Tuple[int, int]]: The new shortcut hubs.

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

        if self.shortcut_connections is not None:
            self.regenerate_shortcut_connections()

        return self.shortcut_hubs

    def set_shortcut_hubs(self, shortcut_hubs: List[Tuple[int, int]]):
        """
        Set a specific list of cells (y, x) to be used as shortcut hubs.

        Args:
            shortcut_hubs (List[Tuple[int, int]]): The list of cells to use as shortcut hubs.
        """
        self.shortcut_hubs = shortcut_hubs

    def generate_shortcut_connections(self, num_levels: int, additional_edges_fraction: float = 0.2):
        """
        Generate shortcut connections for multiple levels of shortcuts.
        """
        self.num_levels = num_levels

        # Step 1: Build the graph of all walkable cells.
        edges = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x]:
                    if y + 1 < self.grid_height and self.grid[y + 1, x]:
                        edges.append((y * self.grid_width + x, (y + 1) * self.grid_width + x))
                    if x + 1 < self.grid_width and self.grid[y, x + 1]:
                        edges.append((y * self.grid_width + x, y * self.grid_width + x + 1))

        graph = ig.Graph(edges=edges, directed=False)

        all_shortcut_connections = []
        current_level_hubs = self.shortcut_hubs  # Start with all hubs for level 1.

        # Step 3: Generate shortcut connections level by level.
        for level in range(1, num_levels + 1):
            # Print the previous level's hubs.
            # print(f"Level {level} hubs: {current_level_hubs}")

            # Step 2: Rebuild the hub_edges for the current level's hubs only.
            hub_edges = []
            for i, hub1 in enumerate(current_level_hubs):
                for j, hub2 in enumerate(current_level_hubs):
                    if i < j:
                        # Calculate the shortest path distance between hubs.
                        length = graph.distances(
                            source=hub1[0] * self.grid_width + hub1[1], target=hub2[0] * self.grid_width + hub2[1]
                        )[0][0]
                        hub_edges.append((i, j, length))

            # Step 4: Create a graph for the current level's hubs.
            hub_graph = ig.Graph.TupleList(hub_edges, weights=True, directed=False)

            # Step 5: Build the MST for the current level of hubs.
            hub_graph_mst = hub_graph.spanning_tree(weights=hub_graph.es["weight"])

            # Step 6: Add additional edges not in the MST.
            num_additional_edges = int(additional_edges_fraction * len(hub_edges))
            non_mst_edges = [edge for edge in hub_edges if edge not in hub_graph_mst.get_edgelist()]
            non_mst_edges.sort(key=lambda edge: edge[2])  # Sort by weight (distance).
            additional_edges = non_mst_edges[:num_additional_edges]

            for edge in additional_edges:
                hub_graph_mst.add_edge(edge[0], edge[1], weight=edge[2])

            # Step 7: Add the shortcut connections for the current level.
            for edge in hub_graph_mst.es:
                hub1 = current_level_hubs[edge.source]
                hub2 = current_level_hubs[edge.target]
                all_shortcut_connections.append((level, hub1, hub2))

            # Step 8: Update hub_edges for the next level.
            new_edges = [(edge.source, edge.target, edge["weight"]) for edge in hub_graph_mst.es]
            hub_edges.extend(new_edges)

            # Step 9: Select important hubs for the next level, but only from the current level's hubs.
            if level < num_levels:
                # Select the most central hubs within the current level's set.
                selected_hubs_indices = self._select_important_nodes(
                    hub_graph, range(len(current_level_hubs)), len(current_level_hubs) // 1.5
                )

                # Map the selected hubs' indices back to global indices.
                selected_hubs = [current_level_hubs[i] for i in selected_hubs_indices]

                # Now we set the current_level_hubs to the selected hubs for the next level.
                current_level_hubs = selected_hubs

        # Step 10: Sort the connections by level in descending order.
        all_shortcut_connections.sort(key=lambda connection: connection[0], reverse=True)

        self.shortcut_connections = all_shortcut_connections
        return all_shortcut_connections

    def regenerate_shortcut_connections(self) -> List[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
        """
        Convenience method to regenerate the shortcut connections using the currently stored parameters.

        Returns:
            List[Tuple[int, Tuple[int, int], Tuple[int, int]]]: The new shortcut connections.
        """
        # Can only be called after a .generate_shortcut_connections() has already been called.
        if self.num_levels is None:
            raise ValueError(
                "Please call .generate_shortcut_connections() before calling .regenerate_shortcut_connections()."
            )

        # Regenerate the shortcut connections using the currently stored parameters.
        self.shortcut_connections = self.generate_shortcut_connections(self.num_levels)

        return self.shortcut_connections

    def set_shortcut_connections(self, shortcut_connections: List[Tuple[int, Tuple[int, int], Tuple[int, int]]]):
        """
        Set a specific list of connections between shortcut hubs.

        Args:
            shortcut_connections (List[Tuple[int, Tuple[int, int], Tuple[int, int]]]): The list of connections to create between shortcut hubs.
        """
        self.shortcut_connections = shortcut_connections

    def set_shortcut_costs(self, shortcut_costs: List[float]):
        """
        Set the costs of using the different levels of shortcuts.

        Args:
            shortcut_costs (List[float]): The cost of using each level of shortcut.
        """
        self.shortcut_costs = shortcut_costs

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
        font = pygame.font.SysFont(None, 36)
        cell_size = 8
        grid_padding = 64
        fps = 165

        width = self.grid_width * cell_size + 2 * grid_padding
        height = self.grid_height * cell_size + 2 * grid_padding

        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        show_levels = list(range(1, self.num_levels + 1))

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
                                    grid_padding + x * cell_size,
                                    grid_padding + y * cell_size,
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
                                    grid_padding + x * cell_size,
                                    grid_padding + y * cell_size,
                                    cell_size,
                                    cell_size,
                                ),
                            )
                    else:
                        pygame.draw.rect(
                            screen,
                            BLACK,
                            pygame.Rect(
                                grid_padding + x * cell_size,
                                grid_padding + y * cell_size,
                                cell_size,
                                cell_size,
                            ),
                        )

            # Draw the connections between shortcut hubs in green.
            for level, hub1, hub2 in self.shortcut_connections:
                if level not in show_levels:
                    continue
                # Calculate the color and thickness based on the number of levels.
                color_intensity = max(0, 255 - (level - 1) * (255 // self.num_levels))
                color = (0, color_intensity, 0)
                thickness = 2 + (level - 1) * (5 // self.num_levels)
                pygame.draw.line(
                    screen,
                    color,
                    (
                        grid_padding + hub1[1] * cell_size + cell_size // 2,
                        grid_padding + hub1[0] * cell_size + cell_size // 2,
                    ),
                    (
                        grid_padding + hub2[1] * cell_size + cell_size // 2,
                        grid_padding + hub2[0] * cell_size + cell_size // 2,
                    ),
                    thickness,
                )

            # Highlight shortcut hubs in red.
            for hub in self.shortcut_hubs:
                pygame.draw.rect(
                    screen,
                    RED,
                    pygame.Rect(
                        grid_padding + hub[1] * cell_size,
                        grid_padding + hub[0] * cell_size,
                        cell_size,
                        cell_size,
                    ),
                )

            # In the top region, display the currently shown levels.
            text = font.render("Showing levels: " + ", ".join(str(level) for level in show_levels) + ".", True, WHITE)
            screen.blit(text, (grid_padding, grid_padding // 2))

            # In the bottom region, display what each hotkey does.
            text = font.render("R: Regenerate Grid  H: Regenerate Hubs  ESC: Quit", True, WHITE)
            screen.blit(text, (grid_padding, height - grid_padding))
            text = font.render("1-9: Show Specific Level  0: Show All Levels", True, WHITE)
            screen.blit(text, (grid_padding, height - grid_padding + 32))

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
                    # If the user presses a number key (1-9), only show that level of shortcuts.
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        show_levels = [int(pygame.key.name(event.key))]
                    # If the user presses the 0 key, show all levels of shortcuts.
                    elif event.key == pygame.K_0:
                        show_levels = list(range(1, self.num_levels + 1))

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

    def _select_important_nodes(self, graph, node_indices, num_nodes):
        """
        Select nodes that balance maximizing pairwise distances and closeness to other nodes.

        Parameters:
            graph (igraph.Graph): The graph to analyze.
            node_indices (list): List of node indices to consider.
            num_nodes (int): Number of nodes to select.

        Returns:
            list: Selected nodes' indices.
        """
        # Step 1: Calculate all pairwise shortest path distances.
        pairwise_distances = []
        for i in range(len(node_indices)):
            for j in range(i + 1, len(node_indices)):
                dist = graph.distances(source=node_indices[i], target=node_indices[j], weights=graph.es["weight"])[0][0]
                pairwise_distances.append(((node_indices[i], node_indices[j]), dist))

        # Step 2: Sort pairwise distances by distance (maximize distance).
        pairwise_distances.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Initialize selected nodes with an empty list.
        selected_nodes = set()

        # Select the first node of the pair with the largest distance.
        selected_nodes.add(pairwise_distances[0][0][0])
        selected_nodes.add(pairwise_distances[0][0][1])

        # Step 4: Select remaining nodes based on two objectives: maximizing pairwise distance and being close to others.
        while len(selected_nodes) < num_nodes:
            max_score = -float("inf")
            best_node = None

            for node in node_indices:
                if node in selected_nodes:
                    continue

                # Maximize distance between selected nodes.
                total_distance = sum(
                    graph.distances(source=node, target=selected_node)[0][0] for selected_node in selected_nodes
                )

                # Calculate closeness to other nodes (minimizing distance to other selected nodes).
                total_closeness = sum(
                    graph.distances(source=node, target=other_node)[0][0]
                    for other_node in node_indices
                    if other_node not in selected_nodes
                )

                # Combine both objectives into a score: maximize distance and prioritize being close to others.
                score = total_distance - total_closeness  # Maximize distance, minimize closeness to others.

                if score > max_score:
                    max_score = score
                    best_node = node

            # Add the best node to the selected nodes.
            selected_nodes.add(best_node)

        return list(selected_nodes)


if __name__ == "__main__":
    generator = ShortcutGenerator()
    generator.seed(0)

    generator.generate_grid(grid_height=100, grid_width=100, blocker_prob=0.475, desired_walkability=0.6)
    generator.generate_shortcut_hubs(num_shortcut_hubs=20)
    generator.generate_shortcut_connections(num_levels=4)
    generator.set_shortcut_costs(shortcut_costs=[-1.0, -5.0, -10.0, -20.0])

    generator.render()

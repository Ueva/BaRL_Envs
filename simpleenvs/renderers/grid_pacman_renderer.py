import sys

import numpy as np

from copy import deepcopy

import pygame
from pygame.locals import *

# Colour Constants.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (59, 86, 165)
YELLOW = (242, 235, 11)
AQUA = (0, 255, 255)

# Tile Types
FLOOR = "."
WALL = "#"
START = "S"
GOAL = "G"
AGENT = "A"
GHOST = "X"

colours = {FLOOR: BLACK, WALL: BLUE, AGENT: YELLOW, START: RED, GOAL: GREEN, GHOST: AQUA}

TILESIZE = 16


class GridPacManRenderer(object):
    def __init__(self, level_layout, start_state=None, goal_states=None):
        self._update_room_layout(level_layout, start_state, goal_states)

        # Initialise pygame and display window.
        pygame.init()
        self.display_window = pygame.display.set_mode((self.width * TILESIZE, self.height * TILESIZE))

    def _update_room_layout(self, level_layout, start_state=None, goal_states=None):
        self.level = level_layout.tolist()
        self.height = len(level_layout)
        self.width = len(level_layout[0])

        self.start_state = start_state
        self.goal_states = goal_states

        for y in range(self.height):
            for x in range(self.width):
                if self.level[y][x] == GHOST:
                    self.level[y][x] = FLOOR

    def update(self, agent_position, ghost_positions, level_layout, start_state=None, goal_states=None):
        pygame.event.get()

        self._update_room_layout(level_layout, start_state, goal_states)

        current_rooms = self.level[:]

        # Designate the start tile.
        if self.start_state is not None:
            i, j = self.start_state
            current_rooms[i][j] = START

        # Designate the goal tile.
        if self.goal_states is not None:
            for goal_state in self.goal_states:
                i, j = goal_state
                current_rooms[i][j] = GOAL

        # Designate the agent's current tile.
        i, j = agent_position
        current_rooms[i][j] = AGENT

        # Designate ghost tiles.
        for ghost_position in ghost_positions:
            current_rooms[ghost_position[0]][ghost_position[1]] = GHOST

        # Draw the room.
        for y in range(0, self.height):
            for x in range(0, self.width):
                pygame.draw.rect(
                    self.display_window,
                    colours[current_rooms[y][x]],
                    (x * TILESIZE, y * TILESIZE, TILESIZE, TILESIZE),
                )

        # Update the display.
        pygame.display.update()

    def close(self):
        pygame.quit()
        sys.exit()

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
BLUE = (0, 0, 255)

# Tile Types
FLOOR = "."
WALL = "#"
START = "S"
GOAL = "G"
AGENT = "A"

colours = {FLOOR: WHITE, WALL: BLACK, AGENT: BLUE, START: RED, GOAL: GREEN}

TILESIZE = 8


class RoomRenderer(object):
    def __init__(self, room_layout, start_state=None, goal_states=None):

        self._update_room_layout(room_layout, start_state, goal_states)

        # Initialise pygame and display window.
        pygame.init()
        self.display_window = pygame.display.set_mode((self.width * TILESIZE, self.height * TILESIZE))

    def _update_room_layout(self, room_layout, start_state=None, goal_states=None):
        self.rooms = room_layout.tolist()
        for i in range(len(self.rooms)):
            self.rooms[i] = [FLOOR if cell == START or cell == GOAL else cell for cell in self.rooms[i]]

        self.height = len(room_layout)
        self.width = len(room_layout[0])

        self.start_state = start_state
        self.goal_states = goal_states

    def update(self, agent_position, room_layout, start_state=None, goal_states=None):
        pygame.event.get()

        self._update_room_layout(room_layout, start_state, goal_states)

        # Make a copy of the 'clean' room layout we can restore from later.
        backup = deepcopy(self.rooms)
        current_rooms = self.rooms[:]

        # Designate the start tile.
        if self.start_state is not None:
            i, j = self.start_state
            current_rooms[i][j] = START

        # Designate the goal tile.
        if self.goal_states is not None:
            for i, j in self.goal_states:
                current_rooms[i][j] = GOAL

        # Designate the agent's current tile.
        i, j = agent_position
        current_rooms[i][j] = AGENT

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

        # Reset room representation.
        self.rooms = backup

    def close(self):
        pygame.quit()

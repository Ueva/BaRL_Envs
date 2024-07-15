import sys

import numpy as np

from copy import deepcopy

import pygame
from pygame.locals import *

# Colour Constants.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Dimensions.
HEIGHT = 5
WIDTH = 5
TILESIZE = 32

# Location Constants.
LOCATIONS = {
    (0, 0): YELLOW,
    (0, 4): RED,
    (3, 0): BLUE,
    (4, 4): GREEN,
}

TAXI_RANKS = [0, 3, 20, 24, -1]

# Import texture files.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from . import taxi_renderer_resources

with pkg_resources.path(taxi_renderer_resources, "taxi_full.png") as path:
    taxi_full_path = path

with pkg_resources.path(taxi_renderer_resources, "taxi_empty.png") as path:
    taxi_empty_path = path

with pkg_resources.path(taxi_renderer_resources, "passenger.png") as path:
    passenger_path = path

with pkg_resources.path(taxi_renderer_resources, "goal_flag.png") as path:
    goal_flag_path = path


class TaxiRenderer(object):
    def __init__(self):

        # Initialise pygame and display window.
        pygame.init()
        self.display_window = pygame.display.set_mode((WIDTH * TILESIZE, HEIGHT * TILESIZE))

        # Load textures.
        self.taxi_full_texture = pygame.transform.scale(
            pygame.image.load(taxi_full_path).convert_alpha(), (TILESIZE, TILESIZE)
        )
        self.taxi_empty_texture = pygame.transform.scale(
            pygame.image.load(taxi_empty_path).convert_alpha(), (TILESIZE, TILESIZE)
        )
        self.passenger_texture = pygame.transform.scale(
            pygame.image.load(passenger_path).convert_alpha(), (TILESIZE, TILESIZE)
        )
        self.goal_flag_texture = pygame.transform.scale(
            pygame.image.load(goal_flag_path).convert_alpha(), (TILESIZE, TILESIZE)
        )

    def update(self, state):
        pygame.event.get()

        # Draw the empty gridworld.
        for y in range(0, HEIGHT):
            for x in range(0, WIDTH):
                # Draw taxi rank.
                if (x, y) in LOCATIONS.keys():
                    pygame.draw.rect(
                        self.display_window,
                        LOCATIONS[(x, y)],
                        (x * TILESIZE, (HEIGHT - y - 1) * TILESIZE, TILESIZE, TILESIZE),
                    )
                # Draw empty tile.
                else:
                    pygame.draw.rect(
                        self.display_window,
                        WHITE,
                        (x * TILESIZE, (HEIGHT - y - 1) * TILESIZE, TILESIZE, TILESIZE),
                    )

        # Draw leftmost wall.
        pygame.draw.rect(
            self.display_window,
            BLACK,
            (1 * TILESIZE - 0.05 * TILESIZE, (HEIGHT - 1 - 1) * TILESIZE, 0.1 * TILESIZE, 2 * TILESIZE),
        )

        # Draw centre wall.
        pygame.draw.rect(
            self.display_window,
            BLACK,
            (2 * TILESIZE - 0.05 * TILESIZE, (HEIGHT - 4 - 1) * TILESIZE, 0.1 * TILESIZE, 2 * TILESIZE),
        )

        # Draw rightmost wall.
        pygame.draw.rect(
            self.display_window,
            BLACK,
            (3 * TILESIZE - 0.05 * TILESIZE, (HEIGHT - 1 - 1) * TILESIZE, 0.1 * TILESIZE, 2 * TILESIZE),
        )

        # Decompose state.
        taxi_position, passenger_position, destination_position = state

        # Draw the taxi.
        taxi_x, taxi_y = self._number_to_coords(taxi_position)
        if TAXI_RANKS[passenger_position] == -1:
            self.display_window.blit(self.taxi_full_texture, (taxi_x * TILESIZE, (HEIGHT - taxi_y - 1) * TILESIZE))
        else:
            self.display_window.blit(self.taxi_empty_texture, (taxi_x * TILESIZE, (HEIGHT - taxi_y - 1) * TILESIZE))

        # Draw the passenger.
        if TAXI_RANKS[passenger_position] != -1:
            passenger_x, passenger_y = self._number_to_coords(TAXI_RANKS[passenger_position])
            self.display_window.blit(
                self.passenger_texture, (passenger_x * TILESIZE, (HEIGHT - passenger_y - 1) * TILESIZE)
            )

        # Draw the goal flag.
        goal_x, goal_y = self._number_to_coords(TAXI_RANKS[destination_position])
        self.display_window.blit(self.goal_flag_texture, (goal_x * TILESIZE, (HEIGHT - goal_y - 1) * TILESIZE))

        # Update the display.
        pygame.display.update()

    def close(self):
        pygame.quit()

    def _number_to_coords(self, square_number):
        taxi_y, taxi_x = divmod(square_number, 5)
        return taxi_x, taxi_y

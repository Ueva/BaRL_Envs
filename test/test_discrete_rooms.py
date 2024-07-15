import pytest

import numpy as np

from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms


def test_reset():
    env = DiscreteXuFourRooms()

    state = env.reset()
    assert state == (2, 2)


def test_goal(self):
    env = DiscreteXuFourRooms()

    assert (10, 10) in env.terminal_states


def test_start(self):
    env = DiscreteXuFourRooms()

    assert (2, 2) in env.terminal_states

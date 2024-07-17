import pytest

from simpleenvs.envs.discrete_rooms import XuFourRooms


def test_reset():
    env = XuFourRooms()

    state = env.reset()
    assert state == (2, 2)


def test_goal():
    env = XuFourRooms()

    assert (10, 10) in env.terminal_states


def test_start():
    env = XuFourRooms()

    assert (2, 2) in env.initial_states


def test_reset():
    env = XuFourRooms()
    state = env.reset()

    assert state == (2, 2)


def test_goal():
    env = XuFourRooms()

    assert (10, 10) in env.terminal_states


def test_start():
    env = XuFourRooms()

    assert (2, 2) in env.initial_states


def test_get_successors():
    env = XuFourRooms()

    # Check initial state.
    state = (2, 2)
    successors = env.get_successors(state)
    assert len(successors) == 4
    assert (((1, 2), -0.001), 0.25) in successors
    assert (((3, 2), -0.001), 0.25) in successors
    assert (((2, 1), -0.001), 0.25) in successors
    assert (((2, 3), -0.001), 0.25) in successors

    # Check wall state.
    state = (1, 1)
    successors = env.get_successors(state)
    assert len(successors) == 4
    assert (((1, 1), -0.001), 0.25) in successors
    assert (((2, 1), -0.001), 0.25) in successors
    assert (((1, 1), -0.001), 0.25) in successors
    assert (((1, 2), -0.001), 0.25) in successors

    # Check terminal state.
    successors = env.get_successors((9, 10))
    assert len(successors) == 4
    assert (((8, 10), -0.001), 0.25) in successors
    assert (((10, 10), 0.999), 0.25) in successors
    assert (((9, 9), -0.001), 0.25) in successors
    assert (((9, 11), -0.001), 0.25) in successors


def test_is_terminal():
    env = XuFourRooms()

    assert env.is_state_terminal((4, 2)) == False
    assert env.is_state_terminal((10, 10)) == True


def test_state_space():
    env = XuFourRooms()

    assert len(env.state_space) == 104  # 11x11 grid with 17 walls = 104 states.

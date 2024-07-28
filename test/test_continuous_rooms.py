import pytest
import numpy as np

from simpleenvs.envs.continuous_rooms import ContinuousFourRooms


def test_reset():
    env = ContinuousFourRooms()
    observation, _ = env.reset()
    state = env._get_cell(observation)

    assert np.all(np.array([2.0, 2.0]) <= state)
    assert np.all(state < np.array([3.0, 3.0]))


def test_initial():
    env = ContinuousFourRooms()

    assert len(env.initial_states) == 1
    assert (2, 2) in env.initial_states


def test_goal():
    env = ContinuousFourRooms()

    assert len(env.terminal_states) == 1
    assert (10, 10) in env.terminal_states


def test_step():
    env = ContinuousFourRooms()
    observation, _ = env.reset()
    state = env._get_cell(observation)

    next_observation, reward, terminal, truncated, _ = env.step(1)  # Move down one step.
    next_state = env._get_cell(next_observation)

    # Check that the reward, terminal, and truncated values are as expected.
    assert reward == -0.01
    assert terminal == False
    assert truncated == False

    # Check that the next state is within the expected range.
    expected_y_range = (state[0] + 1.0 - 0.3, state[0] + 1.0)
    expected_x_range = (state[1] - 0.1, state[1] + 0.1)
    assert next_state[0] >= expected_y_range[0]
    assert next_state[0] < expected_y_range[1]
    assert next_state[1] >= expected_x_range[0]
    assert next_state[1] < expected_x_range[1]


def test_terminal():
    # Define a state 0.5 state units above the goal. Any downward action should result in the agent reaching the goal.
    state = np.array([9.5, 10.5])

    env = ContinuousFourRooms()
    observation, _ = env.reset(state)

    next_observation, reward, terminal, truncated, _ = env.step(1)

    state = env._get_cell(observation)
    next_state = env._get_cell(next_observation)

    print(state)
    print(next_state)

    # Check that the reward, terminal, and truncated values are as expected.
    assert reward == 1.0 + -0.01
    assert terminal == True
    assert truncated == False

    # Check that the next state is within the expected range.
    expected_y_range = (9.5 + 1.0 - 0.3, 9.5 + 1.0)
    expected_x_range = (10.5 - 0.1, 10.5 + 0.1)
    assert next_state[0] >= expected_y_range[0]
    assert next_state[0] < expected_y_range[1]
    assert next_state[1] >= expected_x_range[0]
    assert next_state[1] < expected_x_range[1]


def test_state_obs_conversions():
    env = ContinuousFourRooms()

    ## Check that you can transform from state coordinates to observation coordinates and back.
    # Y Direction.
    y_state_initial = 2.5
    y_obs = env.y_interp(y_state_initial)
    y_state_final = env.y_interp_inv(y_obs)
    assert y_state_final == pytest.approx(y_state_initial)

    # X Direction.
    x_state_initial = 2.5
    x_obs = env.x_interp(x_state_initial)
    x_state_final = env.x_interp_inv(x_obs)
    assert x_state_final == pytest.approx(x_state_initial)


def test_state_obs_conversion_integrated():
    env = ContinuousFourRooms()

    state_initial = np.array([2.5, 3.5])
    obs = env._get_observation(state_initial)
    state_final = env._get_cell(obs)
    assert np.all(state_final == pytest.approx(state_initial))

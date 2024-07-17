import pytest

from simpleenvs.envs.discrete_rooms import (
    BasicRewardRoom,
    DoubleRewardRoom,
    BasicPenaltyRoom,
    DoublePenaltyRoom,
)


class TestBasicRewardRoom:
    """
    Testing for the basic reward room.
    The room is an 11x11 grid with the start in the top left, the reward in the
    bottom right and an extra reward to be picked up in the bottom left.

    As result of the above, the room has 2 terminal states, (10,10) and (10,10,1).
    The total number of states should also be double -1 (as the state to pick up
    the reward is present across both levels)
    """

    def test_standard_behaviour(self):
        env = BasicRewardRoom()

        # Check that the initial states have been loaded correctly.
        assert len(env.initial_states) == 1
        assert (2, 2) in env.initial_states

        # Check that the terminal states have been loaded correctly.
        assert len(env.terminal_states) == 2
        assert (10, 10, 1) in env.terminal_states
        assert (10, 10) in env.terminal_states

        # Check .reset works correctly.
        state = env.reset()
        assert state == (2, 2)

        # Check .get_successors works correctly.
        successors = env.get_successors(state)
        assert len(successors) == 4
        assert (((1, 2), -0.001), 0.25) in successors
        assert (((3, 2), -0.001), 0.25) in successors
        assert (((2, 1), -0.001), 0.25) in successors
        assert (((2, 3), -0.001), 0.25) in successors

    def test_extended_state_space(self):
        env = BasicRewardRoom()

        assert len(env.state_space) == 241  # 11x11 grid x2 for item -1 for overlap item position

    def test_successor_reward(self):
        env = BasicRewardRoom()

        successors = env.get_successors((9, 2))
        assert len(successors) == 4
        assert (((8, 2), -0.001), 0.25) in successors
        assert (((10, 2, 1), 9.999), 0.25) in successors
        assert (((9, 1), -0.001), 0.25) in successors
        assert (((9, 3), -0.001), 0.25) in successors

    def test_collect_reward(self):
        env = BasicRewardRoom()

        state = env.reset((9, 2))
        state, reward, done, _ = env.step(1)
        assert state == (10, 2, 1)
        assert reward == 9.999
        assert done == False

    def test_get_successors_item(self):
        env = BasicRewardRoom()

        successors = env.get_successors((9, 2, 1))
        assert len(successors) == 4

        assert (((10, 2, 1), -0.001), 0.25) in successors  # should not pick up the reward again
        assert ((8, 2, 1), -0.001), 0.25 in successors
        assert (((9, 1, 1), -0.001), 0.25) in successors
        assert (((9, 3, 1), -0.001), 0.25) in successors

    def test_get_item_locations(self):
        env = BasicRewardRoom()

        assert env.get_item_locations() == [(10, 2)]

    def test_return_to_collected_reward(self):
        env = BasicRewardRoom()

        state = env.reset((9, 2))
        state, reward, done, _ = env.step(1)
        assert state == (10, 2, 1)
        assert reward == 9.999
        assert done == False

        state, reward, done, _ = env.step(0)
        assert state == (9, 2, 1)
        assert reward == -0.001
        assert done == False

        state, reward, done, _ = env.step(1)
        assert state == (10, 2, 1)
        assert reward == -0.001
        assert done == False

    def test_expected_state_path(self):
        """
        Test that goes through an entire episode from start to item to terminal state
        """
        env = BasicRewardRoom()

        expected = [
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2, 1),
            (10, 3, 1),
            (10, 4, 1),
            (10, 5, 1),
            (10, 6, 1),
            (10, 7, 1),
            (10, 8, 1),
            (10, 9, 1),
            (10, 10, 1),
        ]
        actions = [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
        state = env.reset()
        for s, a in zip(expected, actions):
            ns, _, _, _ = env.step(a)
            assert ns == s
            state = ns
        assert env.is_state_terminal(state) == True


class TestDoubleRewardRoom:
    """
    Testing for the double reward room
    The room is an 11x11 grid with the start in the top left.
    There are two additional rewards, one in the bottom left and one in the bottom right.

    As result of the above, the room has 4 terminal states:
        (10,10)
        (10,10,1)
        (10,10,0,1)
        (10,10,1,1)
    """

    def test_information(self):
        env = DoubleRewardRoom()

        assert len(env.terminal_states) == 4
        assert len(env.initial_states) == 1
        assert (10, 10) in env.terminal_states
        assert (10, 10, 1) in env.terminal_states
        assert (10, 10, 0, 1) in env.terminal_states
        assert (10, 10, 1, 1) in env.terminal_states
        assert (2, 2) in env.initial_states

    def test_expected_state_path(self):
        """
        Full episode test that collects both item and visits terminal state
        """
        env = DoubleRewardRoom()

        # go from start to bottom left to top right to bottom right
        expected = [
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2, 0, 1),
            (10, 3, 0, 1),
            (9, 3, 0, 1),
            (9, 4, 0, 1),
            (8, 4, 0, 1),
            (8, 5, 0, 1),
            (7, 5, 0, 1),
            (7, 6, 0, 1),
            (6, 6, 0, 1),
            (6, 7, 0, 1),
            (5, 7, 0, 1),
            (5, 8, 0, 1),
            (4, 8, 0, 1),
            (4, 9, 0, 1),
            (3, 9, 0, 1),
            (3, 10, 0, 1),
            (2, 10, 1, 1),
            (3, 10, 1, 1),
            (4, 10, 1, 1),
            (5, 10, 1, 1),
            (6, 10, 1, 1),
            (7, 10, 1, 1),
            (8, 10, 1, 1),
            (9, 10, 1, 1),
            (10, 10, 1, 1),
        ]
        actions = [1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        state = env.reset()
        for i, (s, a) in enumerate(zip(expected, actions)):
            ns, _, _, _ = env.step(a)
            assert ns == s
            state = ns
        assert env.is_state_terminal(state) == True

        # go from start to top right to bottom left to bottom right
        expected = [
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10, 1),
            (3, 10, 1),
            (3, 9, 1),
            (4, 9, 1),
            (4, 8, 1),
            (5, 8, 1),
            (5, 7, 1),
            (6, 7, 1),
            (6, 6, 1),
            (7, 6, 1),
            (7, 5, 1),
            (8, 5, 1),
            (8, 4, 1),
            (9, 4, 1),
            (9, 3, 1),
            (10, 3, 1),
            (10, 2, 1, 1),
            (10, 3, 1, 1),
            (10, 4, 1, 1),
            (10, 5, 1, 1),
            (10, 6, 1, 1),
            (10, 7, 1, 1),
            (10, 8, 1, 1),
            (10, 9, 1, 1),
            (10, 10, 1, 1),
        ]
        actions = [3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        state = env.reset()
        for i, (s, a) in enumerate(zip(expected, actions)):
            ns, _, _, _ = env.step(a)
            assert ns == s
            state = ns
        assert env.is_state_terminal(state) == True


class TestBasicPenaltyRoom:

    def test_standard_behaviour(self):
        env = BasicPenaltyRoom()

        assert len(env.terminal_states) == 2

    def test_extended_state_space(self):
        env = BasicPenaltyRoom()

        assert len(env.state_space) == 241  # 11x11 grid x2 for item -1 for overlap item position

    def test_successor_reward(self):
        env = BasicPenaltyRoom()

        successors = env.get_successors((9, 2))
        assert len(successors) == 4
        assert (((8, 2), -0.001), 0.25) in successors
        assert (((10, 2, 1), -10.001), 0.25) in successors
        assert (((9, 1), -0.001), 0.25) in successors
        assert (((9, 3), -0.001), 0.25) in successors

    def test_collect_reward(self):
        env = BasicPenaltyRoom()

        state = env.reset((9, 2))
        state, reward, done, _ = env.step(1)
        assert state == (10, 2, 1)
        assert reward == -10.001
        assert done == False

    def test_get_successors_item(self):
        env = BasicPenaltyRoom()

        successors = env.get_successors((9, 2, 1))
        assert len(successors) == 4

        assert (((10, 2, 1), -0.001), 0.25) in successors  # should not pick up the reward again
        assert (((8, 2, 1), -0.001), 0.25) in successors
        assert (((9, 1, 1), -0.001), 0.25) in successors
        assert (((9, 3, 1), -0.001), 0.25) in successors

    def test_get_item_locations(self):
        env = BasicPenaltyRoom()

        assert env.get_item_locations() == [(10, 2)]


class TestDoublePenaltyRoom:
    """
    Testing for the double penalty room
    The room is an 11x11 grid with the start in the top left.
    There are two penalties, one in the bottom left and one in the bottom right.

    As result of the above, the room has 4 terminal states:
        (10,10)
        (10,10,1)
        (10,10,0,1)
        (10,10,1,1)
    """

    def test_information(self):
        env = DoublePenaltyRoom()

        assert len(env.terminal_states) == 4
        assert (10, 10) in env.terminal_states
        assert (10, 10, 1) in env.terminal_states
        assert (10, 10, 0, 1) in env.terminal_states
        assert (10, 10, 1, 1) in env.terminal_states

    def test_pick_up_penalty(self):
        env = DoublePenaltyRoom()

        successors = env.get_successors((9, 2))
        assert len(successors) == 4
        assert (((8, 2), -0.001), 0.25) in successors
        assert (((10, 2, 0, 1), -10.001), 0.25) in successors
        assert (((9, 1), -0.001), 0.25) in successors
        assert (((9, 3), -0.001), 0.25) in successors

        _ = env.reset(state=(9, 2))
        ns, r, _, _ = env.step(1)
        assert ns == (10, 2, 0, 1)
        assert r == -10.001

        # pick up the second penalty
        _ = env.reset(state=(2, 9))
        ns, r, _, _ = env.step(3)
        assert ns == (2, 10, 1)
        assert r == -10.001

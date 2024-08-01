import numpy as np
import pytest
from simpleenvs.envs.discrete_rooms import (
    BasicRewardRoom,
    DoubleRewardRoom,
    BasicPenaltyRoom,
    DoublePenaltyRoom,
    FourRoomsFireWall,
    FourRoomsPenalty,
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

    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = BasicRewardRoom()

    def test_standard_behaviour(self):
        state = self.env.reset()
        assert len(self.env.terminal_states) == 2
        assert len(self.env.initial_states) == 1
        assert state == (2, 2)
        assert (10, 10) in self.env.terminal_states
        assert (10, 10, 1) in self.env.terminal_states
        assert (2, 2) in self.env.initial_states
        successors = self.env.get_successors(state)
        assert len(successors) == 4
        assert (((1, 2), -0.001), 0.25) in successors
        assert (((3, 2), -0.001), 0.25) in successors
        assert (((2, 1), -0.001), 0.25) in successors
        assert (((2, 3), -0.001), 0.25) in successors

    def test_extended_state_space(self):
        assert len(self.env.state_space) == 241
        # 11x11 grid x2 for item -1 for overlap item position

    def test_successor_reward(self):
        successors = self.env.get_successors((9, 2))
        assert len(successors) == 4
        assert (((8, 2), -0.001), 0.25) in successors
        assert (((10, 2, 1), 9.999), 0.25) in successors
        assert (((9, 1), -0.001), 0.25) in successors
        assert (((9, 3), -0.001), 0.25) in successors

    def test_collect_reward(self):
        state = self.env.reset((9, 2))
        state, reward, done, _ = self.env.step(1)
        assert state == (10, 2, 1)
        assert reward == 9.999
        assert not done

    def test_get_successors_item(self):
        s = self.env.get_successors((9, 2, 1))
        assert len(s) == 4
        assert (((10, 2, 1), -0.001), 0.25) in s
        assert (((8, 2, 1), -0.001), 0.25) in s
        assert (((9, 1, 1), -0.001), 0.25) in s
        assert (((9, 3, 1), -0.001), 0.25) in s

    def test_get_item_locations(self):
        assert self.env.get_item_locations() == [(10, 2)]

    def test_return_to_collected_reward(self):
        state = self.env.reset((9, 2))
        state, reward, done, _ = self.env.step(1)
        assert state == (10, 2, 1)
        assert reward == 9.999
        assert not done
        state, reward, done, _ = self.env.step(0)
        assert state == (9, 2, 1)
        assert reward == -0.001
        assert not done
        state, reward, done, _ = self.env.step(1)
        assert state == (10, 2, 1)
        assert reward == -0.001
        assert not done

    def test_expected_state_path(self):
        state = self.env.reset((9, 2))
        ns, r, _, _ = self.env.step(1)
        assert ns == (10, 2, 1)
        assert r == 9.999

        state = self.env.reset((10, 9))
        ns, r, d, _ = self.env.step(3)
        assert ns == (10, 10)
        assert r == 1
        assert d
        assert self.env.is_state_terminal(ns)


class TestDoubleRewardRoom:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = DoubleRewardRoom()

    def test_information(self):
        assert len(self.env.terminal_states) == 4
        assert (10, 10) in self.env.terminal_states
        assert (10, 10, 1) in self.env.terminal_states
        assert (10, 10, 0, 1) in self.env.terminal_states
        assert (10, 10, 1, 1) in self.env.terminal_states
        assert (2, 2) in self.env.initial_states

    def test_extended_state_space(self):
        state = self.env.reset((9, 2))
        ns, r, d, _ = self.env.step(1)
        assert ns == (10, 2, 0, 1)
        assert r == 9.999
        assert not d

        state = self.env.reset((3, 10, 0, 1))
        ns, r, d, _ = self.env.step(0)
        assert ns == (2, 10, 1, 1)
        assert r == 9.999
        assert not d

        state = self.env.reset((9, 10, 1, 1))
        ns, r, d, _ = self.env.step(1)
        assert ns == (10, 10, 1, 1)
        assert r == 1
        assert d


class TestBasicPenaltyRoom:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = BasicPenaltyRoom()

    def test_standard_behaviour(self):
        assert len(self.env.terminal_states) == 2

    def test_extended_state_space(self):
        assert len(self.env.state_space) == 241
        # 11x11 grid x2 for item -1 for overlap item position

    def test_successor_reward(self):
        successors = self.env.get_successors((9, 2))
        assert len(successors) == 4
        assert (((8, 2), -0.001), 0.25) in successors
        assert (((10, 2, 1), -10.001), 0.25) in successors
        assert (((9, 1), -0.001), 0.25) in successors
        assert (((9, 3), -0.001), 0.25) in successors

    def test_collect_reward(self):
        state = self.env.reset((9, 2))
        state, reward, done, _ = self.env.step(1)
        assert state == (10, 2, 1)
        assert reward == -10.001
        assert not done

    def test_get_successors_item(self):
        s = self.env.get_successors((9, 2, 1))
        assert len(s) == 4
        assert (((10, 2, 1), -0.001), 0.25) in s
        assert (((8, 2, 1), -0.001), 0.25) in s
        assert (((9, 1, 1), -0.001), 0.25) in s
        assert (((9, 3, 1), -0.001), 0.25) in s

    def test_get_item_locations(self):
        assert self.env.get_item_locations() == [(10, 2)]


class TestDoublePenaltyRoom:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = DoublePenaltyRoom()

    def test_information(self):
        assert len(self.env.terminal_states) == 4
        assert (10, 10) in self.env.terminal_states
        assert (10, 10, 1) in self.env.terminal_states
        assert (10, 10, 0, 1) in self.env.terminal_states
        assert (10, 10, 1, 1) in self.env.terminal_states

    def test_pick_up_penalty(self):
        successors = self.env.get_successors((9, 2))
        assert len(successors) == 4
        assert (((8, 2), -0.001), 0.25) in successors
        assert (((10, 2, 0, 1), -10.001), 0.25) in successors
        assert (((9, 1), -0.001), 0.25) in successors
        assert (((9, 3), -0.001), 0.25) in successors
        state = self.env.reset(state=(9, 2))
        ns, r, _, _ = self.env.step(1)
        assert ns == (10, 2, 0, 1)
        assert r == -10.001

        # pick up the second penalty
        state = self.env.reset(state=(2, 9))
        ns, r, _, _ = self.env.step(3)
        assert ns == (2, 10, 1)
        assert r == -10.001


class TestPersistentPenaltyRoom:
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

    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = FourRoomsFireWall()

    def test_information(self):
        assert len(self.env.terminal_states) == 1
        assert (10, 10) in self.env.terminal_states

    def test_persistent_penalty(self):
        successors = self.env.get_successors((2, 6))
        assert len(successors) == 4
        assert (((1, 6), -0.001), 0.25) in successors
        assert (((3, 6), -0.001), 0.25) in successors
        assert (((2, 7), -50.001), 0.25) in successors
        assert (((2, 5), -0.001), 0.25) in successors

        successors = self.env.get_successors((5, 6))
        assert len(successors) == 4
        assert (((4, 6), -0.001), 0.25) in successors
        assert (((6, 6), -50.001), 0.25) in successors
        assert (((5, 7), -50.001), 0.25) in successors
        assert (((5, 5), -0.001), 0.25) in successors

        # visit the penalty
        state = self.env.reset(state=(2, 6))
        ns, r, _, _ = self.env.step(3)
        assert ns == (2, 7)
        assert r == -50.001

        # move back up
        ns, r, _, _ = self.env.step(2)
        assert ns == (2, 6)
        assert r == -0.001

        # go back to penalty
        ns, r, _, _ = self.env.step(3)
        assert ns == (2, 7)
        assert r == -50.001


class TestFourRoomsPenalty:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = FourRoomsPenalty()

    def test_state_space_init(self):
        assert (10, 2) not in self.env.state_space
        assert (10, 2, 1) in self.env.state_space
        print(f"transition matrix:")
        for k, v in self.env.transition_matrix.items():
            print(f"{k} : {v}")
        tm_states = set([k[0] for k in self.env.transition_matrix.keys()])
        next_states = set()
        for t in self.env.transition_matrix.values():
            next_states.update([ns for ((ns, _), _) in t])
        print(f"tm_states: {tm_states}")
        print(f"next_states: {next_states}")
        print(f"(10,2) in tm_states: {(10,2) in tm_states}")
        print(f"(10,2) in next_states: {(10,2) in next_states}")
        print(f"self.env.reset((10,2,1)) = {self.env.reset((10,2,1))}")

        assert self.env.reset((10, 2, 1)) == (10, 2, 1)


# pytest start instead
if __name__ == "__main__":
    pytest.main(
        [
            "-q",
            "--disable-warnings",
            "--durations=5",
            "--durations-min=0.1",  # limit for slowest 5 tests
            # limit for slowest 5 tests
        ]
    )

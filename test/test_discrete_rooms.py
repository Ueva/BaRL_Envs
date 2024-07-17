import pytest

import numpy as np
import unittest # unittest as standard library
from simpleenvs.envs.discrete_rooms import (
        DiscreteXuFourRooms, 
        BasicRewardRoom, 
        DoubleRewardRoom, 
        BasicPenaltyRoom, 
        DoublePenaltyRoom,
        FourRoomsFireWall
)


def test_reset():
    env = DiscreteXuFourRooms()

    state = env.reset()
    assert state == (2, 2)


def test_goal():
    env = DiscreteXuFourRooms()

    assert (10, 10) in env.terminal_states


def test_start():
    env = DiscreteXuFourRooms()

    assert (2, 2) in env.initial_states

class TestDiscreteXuFourRooms(unittest.TestCase):
    def setUp(self):
        self.env = DiscreteXuFourRooms()

    def test_reset(self):
        state = self.env.reset()
        self.assertEqual(state, (2,2))

    def test_goal(self):
        self.assertIn((10,10), self.env.terminal_states)

    def test_start(self):
        self.assertIn((2,2), self.env.initial_states)

    def test_get_successors(self):
        state = (2,2)
        successors = self.env.get_successors(state)
        self.assertEqual(len(successors), 4)
        self.assertIn( (((1,2),-0.001),0.25), successors)
        self.assertIn( (((3,2),-0.001),0.25), successors)
        self.assertIn( (((2,1),-0.001),0.25), successors)
        self.assertIn( (((2,3),-0.001),0.25), successors)

        # check wall state
        state = (1,1)
        successors = self.env.get_successors(state)
        self.assertEqual(len(successors), 4)
        self.assertIn( (((1,1),-0.001),0.25), successors)
        self.assertIn( (((2,1),-0.001),0.25), successors)
        self.assertIn( (((1,1),-0.001),0.25), successors)
        self.assertIn( (((1,2),-0.001),0.25), successors)

        # check terminal state
        successors = self.env.get_successors((9,10))
        self.assertEqual(len(successors), 4)
        self.assertIn( (((8,10),-0.001),0.25), successors)
        self.assertIn( (((10,10),0.999),0.25), successors)
        self.assertIn( (((9,9),-0.001),0.25), successors)
        self.assertIn( (((9,11),-0.001),0.25), successors)
    
    def test_is_terminal(self):
        self.assertFalse(self.env.is_state_terminal((4,2)))
        self.assertTrue(self.env.is_state_terminal((10,10)))

    def test_state_space(self):
        self.assertEqual(len(self.env.state_space), 104) # 11x11 grid with 17 walls = 104 statesw
        
class TestBasicRewardRoom(unittest.TestCase):
    """
    Testing for the basic reward room.
    The room is an 11x11 grid with the start in the top left, the reward in the 
    bottom right and an extra reward to be picked up in the bottom left.

    As result of the above, the room has 2 terminal states, (10,10) and (10,10,1).
    The total number of states should also be double -1 (as the state to pick up 
    the reward is present across both levels)
    """
    def setUp(self):
        self.env = BasicRewardRoom()

    def test_standard_behaviour(self):
        state = self.env.reset()
        self.assertEqual(len(self.env.terminal_states), 2)
        self.assertEqual(len(self.env.initial_states), 1)
        self.assertEqual(state, (2,2))
        self.assertIn((10,10), self.env.terminal_states)
        self.assertIn((10,10,1), self.env.terminal_states)
        self.assertIn((2,2), self.env.initial_states)
        successors = self.env.get_successors(state)
        self.assertEqual(len(successors), 4)
        self.assertIn( (((1,2),-0.001),0.25), successors)
        self.assertIn( (((3,2),-0.001),0.25), successors)
        self.assertIn( (((2,1),-0.001),0.25), successors)
        self.assertIn( (((2,3),-0.001),0.25), successors)

    def test_extended_state_space(self):
        self.assertEqual(len(self.env.state_space), 241) # 11x11 grid x2 for item -1 for overlap item position
    
    def test_successor_reward(self):
        successors = self.env.get_successors((9,2))
        self.assertEqual(len(successors), 4)
        self.assertIn( (((8,2),-0.001),0.25), successors)
        self.assertIn( (((10,2,1),9.999),0.25), successors)
        self.assertIn( (((9,1),-0.001),0.25), successors)
        self.assertIn( (((9,3),-0.001),0.25), successors)

    def test_collect_reward(self):
        state = self.env.reset((9,2))
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(state, (10,2,1))
        self.assertEqual(reward, 9.999)
        self.assertFalse(done)

    def test_get_successors_item(self):
        s = self.env.get_successors((9,2,1))
        self.assertEqual(len(s), 4)

        self.assertIn( (((10,2,1),-0.001),0.25), s) # should not pick up the reward again
        self.assertIn( (((8,2,1),-0.001),0.25), s)
        self.assertIn( (((9,1,1),-0.001),0.25), s)
        self.assertIn( (((9,3,1),-0.001),0.25), s)

    def test_get_item_locations(self):
        self.assertEqual(self.env.get_item_locations(), [(10,2)])

    def test_return_to_collected_reward(self):
        state = self.env.reset((9,2))
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(state, (10,2,1))
        self.assertEqual(reward, 9.999)
        self.assertFalse(done)
        # print(f"self.env.transition_matrix: {self.env.transition_matrix}")
        state, reward, done, _ = self.env.step(0)
        self.assertEqual(state, (9,2,1))
        self.assertEqual(reward, -0.001)
        self.assertFalse(done)
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(state, (10,2,1))
        self.assertEqual(reward, -0.001)
        self.assertFalse(done)

    def test_expected_state_path(self):
        """
        Test that goes through an entire episode from start to item to terminal state
        """
        expected = [
                (3,2), (4,2), (5,2), (6,2), (7,2), (8,2),
                (9,2), (10,2,1), (10,3,1), (10,4,1), (10,5,1),
                (10,6,1), (10,7,1), (10,8,1), (10,9,1), (10,10,1)
                ]
        actions = [ 1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3 ]
        state = self.env.reset()
        for s,a in zip(expected, actions):
            ns, _, _, _ = self.env.step(a)
            self.assertEqual(ns, s)
            state = ns
        self.assertTrue(self.env.is_state_terminal(state))


class TestDoubleRewardRoom(unittest.TestCase):
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
    def setUp(self):
        self.env = DoubleRewardRoom()

    def test_information(self):
        self.assertEqual(len(self.env.terminal_states), 4)
        self.assertEqual(len(self.env.initial_states), 1)
        self.assertIn((10,10), self.env.terminal_states)
        self.assertIn((10,10,1), self.env.terminal_states)
        self.assertIn((10,10,0,1), self.env.terminal_states)
        self.assertIn((10,10,1,1), self.env.terminal_states)
        self.assertIn((2,2), self.env.initial_states)


    def test_expected_state_path(self):
        """
        Full episode test that collects both item and visits terminal state
        """
        # go from start to bottom left to top right to bottom right
        expected = [
                (3,2), (4,2), (5,2), (6,2), (7,2), (8,2),
                (9,2), (10,2,0,1), (10,3,0,1), (9,3,0,1), (9,4,0,1),
                (8,4,0,1), (8,5,0,1), (7,5,0,1), (7,6,0,1), (6,6,0,1),
                (6,7,0,1), (5,7,0,1), (5,8,0,1), (4,8,0,1), (4,9,0,1),
                (3,9,0,1), (3,10,0,1), (2,10,1,1), (3,10,1,1), 
                (4,10,1,1), (5,10,1,1), (6,10,1,1), (7,10,1,1),
                (8,10,1,1), (9,10,1,1), (10,10,1,1)
                ]
        actions = [ 1,1,1,1,1,1,1,1,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,1,1,1,1,1,1,1,1 ]
        state = self.env.reset()
        for i,(s,a) in enumerate(zip(expected, actions)):
            ns, _, _, _ = self.env.step(a)
            self.assertEqual(ns, s)
            state = ns
        self.assertTrue(self.env.is_state_terminal(state))


        # go from start to top right to bottom left to bottom right
        expected = [
                (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10,1),
                (3,10,1), (3,9,1), (4,9,1), (4,8,1), (5,8,1), (5,7,1),
                (6,7,1), (6,6,1), (7,6,1), (7,5,1), (8,5,1), (8,4,1),
                (9,4,1), (9,3,1), (10,3,1), (10,2,1,1), (10,3,1,1),
                (10,4,1,1), (10,5,1,1), (10,6,1,1), (10,7,1,1),
                (10,8,1,1), (10,9,1,1), (10,10,1,1)
                ]
        actions = [ 3,3,3,3,3,3,3,3,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,3,3,3,3,3,3,3 ]
        state = self.env.reset()
        for i,(s,a) in enumerate(zip(expected, actions)):
            ns, _, _, _ = self.env.step(a)
            self.assertEqual(ns, s)
            state = ns
        self.assertTrue(self.env.is_state_terminal(state))


class TestBasicPenaltyRoom(unittest.TestCase):
    def setUp(self):
        self.env = BasicPenaltyRoom()

    def test_standard_behaviour(self):
        self.assertEqual(len(self.env.terminal_states), 2)

    def test_extended_state_space(self):
        self.assertEqual(len(self.env.state_space), 241) # 11x11 grid x2 for item -1 for overlap item position
    
    def test_successor_reward(self):
        successors = self.env.get_successors((9,2))
        self.assertEqual(len(successors), 4)
        self.assertIn( (((8,2),-0.001),0.25), successors)
        self.assertIn( (((10,2,1),-10.001),0.25), successors)
        self.assertIn( (((9,1),-0.001),0.25), successors)
        self.assertIn( (((9,3),-0.001),0.25), successors)

    def test_collect_reward(self):
        state = self.env.reset((9,2))
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(state, (10,2,1))
        self.assertEqual(reward, -10.001)
        self.assertFalse(done)

    def test_get_successors_item(self):
        s = self.env.get_successors((9,2,1))
        self.assertEqual(len(s), 4)

        self.assertIn( (((10,2,1),-0.001),0.25), s) # should not pick up the reward again
        self.assertIn( (((8,2,1),-0.001),0.25), s)
        self.assertIn( (((9,1,1),-0.001),0.25), s)
        self.assertIn( (((9,3,1),-0.001),0.25), s)

    def test_get_item_locations(self):
        self.assertEqual(self.env.get_item_locations(), [(10,2)])


class TestDoublePenaltyRoom(unittest.TestCase):
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
    def setUp(self):
        self.env = DoublePenaltyRoom()

    def test_information(self):
        self.assertEqual(len(self.env.terminal_states), 4)
        self.assertIn((10,10), self.env.terminal_states)
        self.assertIn((10,10,1), self.env.terminal_states)
        self.assertIn((10,10,0,1), self.env.terminal_states)
        self.assertIn((10,10,1,1), self.env.terminal_states)

    def test_pick_up_penalty(self):
        successors = self.env.get_successors((9,2))
        self.assertEqual(len(successors), 4)
        self.assertIn( (((8,2),-0.001),0.25), successors)
        self.assertIn( (((10,2,0,1),-10.001),0.25), successors)
        self.assertIn( (((9,1),-0.001),0.25), successors)
        self.assertIn( (((9,3),-0.001),0.25), successors)
        state = self.env.reset(state=(9,2))
        ns, r, _, _ = self.env.step(1)
        self.assertEqual(ns, (10,2,0,1))
        self.assertEqual(r, -10.001)
        
        # pick up the second penalty
        state = self.env.reset(state=(2,9))
        ns, r, _, _ = self.env.step(3)
        self.assertEqual(ns, (2,10,1))
        self.assertEqual(r, -10.001)


class TestPersistentPenaltyRoom(unittest.TestCase):
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
    def setUp(self):
        self.env = FourRoomsFireWall()

    def test_information(self):
        self.assertEqual(len(self.env.terminal_states), 1)
        self.assertIn((10,10), self.env.terminal_states)

    def test_persistent_penalty(self):
        successors = self.env.get_successors((2,6))
        self.assertEqual(len(successors), 4)
        self.assertIn( (((1,6),-0.001),0.25), successors)
        self.assertIn( (((3,6),-0.001),0.25), successors)
        self.assertIn( (((2,7),-50.001),0.25), successors)
        self.assertIn( (((2,5),-0.001),0.25), successors)

        successors = self.env.get_successors((5,6))
        self.assertIn( (((4,6),-0.001),0.25), successors)
        self.assertIn( (((6,6),-50.001),0.25), successors)
        self.assertIn( (((5,7),-50.001),0.25), successors)
        self.assertIn( (((5,5),-0.001),0.25), successors)

        
        # visit the penalty
        state = self.env.reset(state=(2,6))
        ns, r, _, _ = self.env.step(3)
        self.assertEqual(ns, (2,7))
        self.assertEqual(r, -50.001)
        # move back up
        ns, r, _, _ = self.env.step(2)
        self.assertEqual(ns, (2,6))
        self.assertEqual(r, -0.001)
        # go back to penalty
        ns, r, _, _ = self.env.step(3)
        self.assertEqual(ns, (2,7))
        self.assertEqual(r, -50.001)


if __name__ == '__main__':
    unittest.main()

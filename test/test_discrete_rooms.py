import numpy as np
import unittest # unittest as standard library
from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms, BasicRewardRoom, DoubleRewardRoom


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
        self.assertIn( (((10,10),1),0.25), successors)
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
        self.assertEqual(len(self.env.state_space), 241) # 11x11 grid x2 for gold -1 for overlap gold position
    
    def test_successor_reward(self):
        successors = self.env.get_successors((9,2))
        self.assertEqual(len(successors), 4)
        self.assertIn( (((8,2),-0.001),0.25), successors)
        self.assertIn( (((10,2,1),10.0),0.25), successors)
        self.assertIn( (((9,1),-0.001),0.25), successors)
        self.assertIn( (((9,3),-0.001),0.25), successors)

    def test_collect_reward(self):
        state = self.env.reset((9,2))
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(state, (10,2,1))
        self.assertEqual(reward, 10.0)
        self.assertFalse(done)

    def test_get_successors_gold(self):
        s = self.env.get_successors((9,2,1))
        self.assertEqual(len(s), 4)

        self.assertIn( (((10,2,1),-0.001),0.25), s) # should not pick up the reward again
        self.assertIn( (((8,2,1),-0.001),0.25), s)
        self.assertIn( (((9,1,1),-0.001),0.25), s)
        self.assertIn( (((9,3,1),-0.001),0.25), s)

    def test_get_gold_locations(self):
        self.assertEqual(self.env.get_gold_locations(), [(10,2)])

    def test_return_to_collected_reward(self):
        state = self.env.reset((9,2))
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(state, (10,2,1))
        self.assertEqual(reward, 10.0)
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
        expected = [
                (3,2),
                (4,2),
                (5,2),
                (6,2),
                (7,2),
                (8,2),
                (9,2),
                (10,2,1),
                (10,3,1),
                (10,4,1),
                (10,5,1),
                (10,6,1),
                (10,7,1),
                (10,8,1),
                (10,9,1),
                (10,10,1)
                ]
        actions = [
                1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3
                ]
        state = self.env.reset()
        for s,a in zip(expected, actions):
            ns, _, _, _ = self.env.step(a)
            self.assertEqual(ns, s)
            state = ns
        self.assertTrue(self.env.is_state_terminal(state))


if __name__ == '__main__':
    unittest.main()

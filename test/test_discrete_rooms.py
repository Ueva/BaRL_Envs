import numpy as np
import unittest # unittest as standard library
from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms


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

    
if __name__ == '__main__':
    unittest.main()

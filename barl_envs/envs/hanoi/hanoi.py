# Much of this code is adapted from: https://github.com/RobertTLange/gym-hanoi/blob/master/gym_hanoi/envs/hanoi_env.py
import sys
import gym
import math
import copy
import itertools
import numpy as np

from barl_envs.renderers import HanoiRenderer


class HanoiEnvironment(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_disks=3, num_poles=3):

        self.num_disks = num_disks
        self.num_poles = num_poles

        # Define action-space and state-space.
        self.action_space = gym.spaces.Discrete(
            math.factorial(self.num_poles) / math.factorial(self.num_poles - 2)
        )
        self.state_space = gym.tuple(
            gym.spaces.Tuple(self.num_disks * (gym.spaces.Discrete(self.num_poles),))
        )

        # Initialise action mappings.
        self.action_list = list(itertools.permutations(list(range(self.num_poles)), 2))

        # Initialise environment state variables.
        self.current_state = None
        self.goal_state = self.num_disks * (self.num_poles,)
        self.terminal = True

        self.renderer = None

    def reset(self):
        self.current_state = self.num_disks * (0,)
        self.terminal = False
        return copy.deepcopy(self.current_state)

    def step(self, action):
        if self.terminal:
            raise RuntimeError("Please call env.reset() before starting a new episode.")

        # Initialise transition info.
        info = {"invalid_action": False}

        new_state = list(copy.deepcopy(self.current_state))
        source_pole, dest_pole = self.action_list[action]

        # If the chosen action is legal, determine the next state.
        if self._is_action_legal((source_pole, dest_pole)):
            disk_to_move = min(self._disks_on_pole(source_pole))
            new_state[disk_to_move] = dest_pole
            new_state = tuple(new_state)
        # If the chosen action is illegal, state doesn't change.
        else:
            info["invalid_action":True]

        # Reward is 10 for reaching the goal state, -1 otherwise.
        reward = 10 if new_state == self.goal_state else -1

        # Only the goal state is terminal.
        self.done = True if new_state == self.goal_state else False

        # Update current state.
        self.current_state = new_state

        return copy.deepcopy(self.current_state), reward, self.done, info

    def render(self, mode="human"):
        pass

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def get_available_actions(self, state=None):
        if state == None:
            state = self.current_state

        legal_actions = []
        for i in range(len(self.action_list)):
            if self._is_action_legal(self.action_list[i], state=state):
                legal_actions.append(i)

        return legal_actions

    def get_action_mask(self, state=None):
        if state == None:
            state = self.current_state

        legal_actions = self.get_available_actions(state=state)
        all_actions = list(range(len(self.action_list)))
        legal_action_mask = map(lambda action: action in legal_actions, all_actions)

        return list(legal_action_mask)

    def _is_action_legal(self, action, state=None):
        if state == None:
            state = self.current_state

        source_pole, dest_pole = action
        source_disks = self._disks_on_pole(source_pole, state=state)
        dest_disks = self._disks_on_pole(dest_pole, state=state)

        if source_disks == []:
            # Cannot move a disk from an empty pole!
            return False
        else:
            if dest_disks == []:
                # Can always move a disk to an empty pole!
                return True
            else:
                # Otherwise, only allow the move if the smallest disk on the
                # source pole is smaller than the smallest disk on destination pole.
                return min(source_disks) < min(dest_disks)

    def _disks_on_pole(self, pole, state=None):
        if state == None:
            state = self.current_state
        return [disk for disk in range(self.num_disks) if state[disk] == pole]
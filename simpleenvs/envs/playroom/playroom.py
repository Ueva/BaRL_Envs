import copy
import random
import itertools

from collections import defaultdict

import numpy as np
import networkx as nx

from itertools import cycle

from simpleoptions import BaseEnvironment, TransitionMatrixBaseEnvironment

ITEM_TO_INDEX = {
    "LIGHT_SWITCH": 0,
    "MUSIC_SWITCH": 1,
    "BALL": 2,
    "BELL": 3,
}

INDEX_TO_ITEM = {
    0: "LIGHT_SWITCH",
    1: "MUSIC_SWITCH",
    2: "BALL",
    3: "BELL",
}

ACTION_TO_INDEX = {
    "RANDOM_LOOK": 0,
    "HAND_TO_EYE": 1,
    "MARKER_TO_EYE": 2,
    "EYE_TO_HAND": 3,
    "EYE_TO_MARKER": 4,
    "INTERACT": 5,
}

INDEX_TO_ACTION = {
    0: "RANDOM_LOOK",
    1: "HAND_TO_EYE",
    2: "MARKER_TO_EYE",
    3: "EYE_TO_HAND",
    4: "EYE_TO_MARKER",
    5: "INTERACT",
}

# State representation: (Eye Item, Hand Item, Marker Item, Light, Music, Bell)


class PlayroomEnvironment(TransitionMatrixBaseEnvironment):
    def __init__(self, initial_states_order=None, seed=None):
        self.state_space = set(self.generate_interaction_graph(directed=True).nodes)

        if initial_states_order is None:
            self.initial_states_order = None
        else:
            self.initial_states_order = cycle(initial_states_order)

        if seed is not None:
            self.seed(seed)

        super().__init__()

    def reset(self, state=None):
        # If an initial state is specified, use it.
        if state is not None:
            self.current_state = copy.deepcopy(state)
        # Else, if we have a defined initial state order, use the next initial state.
        elif self.initial_states_order is not None:
            self.current_state = next(self.initial_states_order)
        # Else, randomly sample an initial state.
        else:
            initial_eye_item = random.randint(0, 3)
            initial_hand_item = random.randint(0, 3)
            initial_marker_item = random.randint(0, 3)
            self.current_state = (initial_eye_item, initial_hand_item, initial_marker_item, False, False, False)

        return copy.deepcopy(self.current_state)

    def step(self, action, state=None):
        if state is None:
            next_state, reward, terminal, info = super().step(action, state=self.current_state)
        else:
            next_state, reward, terminal, info = super().step(action, state=state)

        self.current_state = next_state

        return next_state, reward, terminal, info

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return {0, 1, 2, 3, 4, 5}

    def get_available_actions(self, state=None):
        if state is None:
            state = self.current_state

        # If the state is terminal, there are no actions available.
        if self.is_state_terminal(state):
            return []

        eye_item = state[0]
        hand_item = state[1]

        # If the Eye and Hand are on the same item...
        if eye_item == hand_item:
            # ...and the item they are on is the Light Switch, Music Switch, or Ball...
            if (
                INDEX_TO_ITEM[eye_item] == "LIGHT_SWITCH"
                or INDEX_TO_ITEM[eye_item] == "MUSIC_SWITCH"
                or INDEX_TO_ITEM[eye_item] == "BALL"
            ):
                # The agent has access to all actions.
                return [0, 1, 2, 3, 4, 5]

        # Else, the agent does not have access to the INTERACT action.
        return [0, 1, 2, 3, 4]

    def get_action_mask(self, state=None):
        if state is None:
            state = self.current_state

        if self.is_state_terminal(state):
            return [0, 0, 0, 0, 0, 0]

        if len(self.get_available_actions(state)) == 5:
            return [1, 1, 1, 1, 1, 1]
        else:
            return [1, 1, 1, 1, 1, 0]

    def render(self):
        pass

    def close(self):
        pass

    def is_state_terminal(self, state=None):
        if state is None:
            state = self.current_state

        # A state is terminal if the music is on and the bell is ringing.
        return state[4] and state[5]

    def get_initial_states(self):
        items = [0, 1, 2, 3]
        return list(itertools.product(items, items, items, [False], [False], [False]))

    def get_successors(self, state=None, actions=None):
        if state is None:
            state = self.current_state

        if actions is None:
            actions = self.get_available_actions(state=state)

        successors = []
        for action in actions:
            eye_item, hand_item, marker_item, light, music, bell = state

            # If bell is currently ringing, stop it.
            bell = False

            # 0 - Look at a Random Item.
            if INDEX_TO_ACTION[action] == "RANDOM_LOOK":
                items = [0, 1, 2, 3]
                for item in items:
                    successors.append((((item, hand_item, marker_item, light, music, bell), -0.001), 1.0 / len(items)))
            # 1 - Move Hand to Eye.
            elif INDEX_TO_ACTION[action] == "HAND_TO_EYE":
                # Succeeds with probability 0.75.
                successors.append((((eye_item, eye_item, marker_item, light, music, bell), -0.001), 0.75))

                # Has no effect with probability 0.25.
                successors.append((((eye_item, hand_item, marker_item, light, music, bell), -0.001), 0.25))

            # 2 - Move Marker to Eye,
            elif INDEX_TO_ACTION[action] == "MARKER_TO_EYE":
                # Succeeds with probability 0.75.
                successors.append((((eye_item, hand_item, eye_item, light, music, bell), -0.001), 0.75))

                # Has no effect with probability 0.25.
                successors.append((((eye_item, hand_item, marker_item, light, music, bell), -0.001), 0.25))

            # 3 -  Move Eye to Hand.
            elif INDEX_TO_ACTION[action] == "EYE_TO_HAND":
                successors.append((((hand_item, hand_item, marker_item, light, music, bell), -0.001), 1.0))

            # 4 -  Move Eye to Marker.
            elif INDEX_TO_ACTION[action] == "EYE_TO_MARKER":
                eye_item = marker_item

                # Succeeds with probability 0.75.
                successors.append((((marker_item, hand_item, marker_item, light, music, bell), -0.001), 0.75))

                # Has no effect with probability 0.25.
                successors.append((((eye_item, hand_item, marker_item, light, music, bell), -0.001), 0.25))

            # 5 - Interact with Item.
            elif INDEX_TO_ACTION[action] == "INTERACT":
                # If Eye and Hand are on Light Switch, toggle Light.
                if INDEX_TO_ITEM[eye_item] == "LIGHT_SWITCH" and INDEX_TO_ITEM[hand_item] == "LIGHT_SWITCH":
                    # Succeeds with probability 0.75.
                    successors.append((((eye_item, hand_item, marker_item, not light, music, bell), -0.001), 0.75))

                    # Has no effect with probability 0.25.
                    successors.append((((eye_item, hand_item, marker_item, light, music, bell), -0.001), 0.25))

                # If Eye and Hand are on Music Switch...
                elif INDEX_TO_ITEM[eye_item] == "MUSIC_SWITCH" and INDEX_TO_ITEM[hand_item] == "MUSIC_SWITCH":
                    # ...and the Light is On, toggle Music.
                    if light:
                        # Succeeds with probability 0.75.
                        successors.append((((eye_item, hand_item, marker_item, light, not music, bell), -0.001), 0.75))

                        # Has no effect with probability 0.25.
                        successors.append((((eye_item, hand_item, marker_item, light, music, bell), -0.001), 0.25))

                    # Otherwise, the Music Switch does nothing.
                # If the Eye and Hand are on the Ball, kick the ball at the Marker.
                # If the Marker is currently on the Bell, ring the Bell.
                elif INDEX_TO_ITEM[eye_item] == "BALL" and INDEX_TO_ITEM[hand_item] == "BALL":
                    if INDEX_TO_ITEM[marker_item] == "BELL":

                        # Succeeds with probability 0.75.
                        # If the music is on when the ball hits the bell and causes it to ring,
                        # the agent earns a reward of 1.0, and the episode ends.
                        if music:
                            successors.append(
                                (((eye_item, hand_item, ITEM_TO_INDEX["BALL"], light, music, True), 1.0), 0.75)
                            )
                        # Otherwise, the agent earns a reward of -0.001 and the episode continues.
                        else:
                            successors.append(
                                (((eye_item, hand_item, ITEM_TO_INDEX["BALL"], light, music, True), -0.001), 0.75)
                            )

                        # Has no effect with probability 0.25.
                        successors.append((((eye_item, hand_item, marker_item, light, music, bell), -0.001), 0.25))
                    else:
                        # Succeeds with probability 0.75.
                        successors.append(
                            (((eye_item, hand_item, ITEM_TO_INDEX["BALL"], light, music, bell), -0.001), 0.75)
                        )

                        # Has no effect with probability 0.25.
                        successors.append((((eye_item, hand_item, marker_item, light, music, bell), -0.001), 0.25))

        return successors

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

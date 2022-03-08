import os
import pickle
import random

import numpy as np

from agent_code.__agent.constants import INDICES_BY_ACTION, SAVED_Q_VALUES_FILE_PATH, SAVED_INDICES_BY_STATE_FILE_PATH, \
    ACTIONS

EPSILON = 0.0


def setup(self):
    self.q_values = np.zeros((1, len(INDICES_BY_ACTION)))
    self.indices_by_state = {None: np.zeros(len(INDICES_BY_ACTION))}
    # Todo: Refactor this.
    self.rounds_not_saved = 0
    self.statistics = None
    if self.train:
        if os.path.isfile(SAVED_Q_VALUES_FILE_PATH) and os.path.isfile(SAVED_INDICES_BY_STATE_FILE_PATH):
            with open(SAVED_Q_VALUES_FILE_PATH, "rb") as file:
                self.q_values = pickle.load(file)
            with open(SAVED_INDICES_BY_STATE_FILE_PATH, "rb") as file:
                self.indices_by_state = pickle.load(file)
    else:
        # Uses the exploited policy.
        raise ArithmeticError()


def act(self, game_state: dict):
    if self.train:
        if str(state_to_features(game_state)) not in self.indices_by_state or random.random() < EPSILON:
            return np.random.choice(ACTIONS)
        else:
            index_state = self.indices_by_state[str(state_to_features(game_state))]
            index_best_action = np.argmax(self.q_values[index_state])
            return ACTIONS[index_best_action]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(convert_agent_state_to_feature(game_state['self']))
    for other in game_state['others']:
        channels.append(convert_agent_state_to_feature(other))
    for bomb in game_state['bombs']:
        channels.append(convert_bomb_state_to_feature(bomb))
    for coin in game_state['coins']:
        channels.append(convert_coin_state_to_feature(coin))
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

def convert_agent_state_to_feature(agent_state):
    return np.array([agent_state[3][0], agent_state[3][1], 1 if agent_state[2] else 0])

def convert_bomb_state_to_feature(bomb_state):
    return np.array([bomb_state[0][0], bomb_state[0][1], bomb_state[1]])

def convert_coin_state_to_feature(coin_state):
    return np.array([coin_state[0], coin_state[1], -1])

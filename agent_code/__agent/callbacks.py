import os
import pickle
import random
from collections import namedtuple

import numpy as np

from agent_code.__agent.constants import \
    ACTIONS, EPSILON, DETECTION_RADIUS, AMOUNT_ELEMENTS, DECAY, BATCH_SIZE, INPUT_SHAPE, MAX_AGENT_COUNT, \
    MAX_BOMB_COUNT, MAX_COIN_COUNT, MAIN_MODEL_FILE_PATH
from agent_code import rule_based_agent
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.python.client import device_lib


def setup(self):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())
    self.statistics = None
    self.epsilon = EPSILON()
    self.batch_size = BATCH_SIZE
    self.model = init_model()

    if os.path.isdir(MAIN_MODEL_FILE_PATH):
        self.model = keras.models.load_model(MAIN_MODEL_FILE_PATH)


def init_model():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding='same', activation="relu", input_shape=INPUT_SHAPE))
    model.add(Conv2D(64, (5, 5), padding='same', activation="relu"))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(ACTIONS), activation='linear'))

    opt = tf.keras.optimizers.Adam()
    model.compile(loss='mse', optimizer=opt)
    return model


def act(self, game_state: dict):
    if random.random() <= self.epsilon:
        return np.random.choice(ACTIONS)
    else:
        features = np.expand_dims(state_to_features(game_state), axis=0)
        q_values = self.model.predict(features)[0]
        index_best_action = np.argmax(q_values)
        return ACTIONS[index_best_action]


def state_to_features(game_state: dict) -> np.array:
    features = state_to_features_cnn(game_state)
    return features

    if game_state is None:
        return None

    channels = []
    channels.append(convert_agent_state_to_feature(game_state['self']))
    for other in game_state['others']:
        channels.append(convert_agent_state_to_feature(other))
    for bomb in game_state['bombs']:
        channels.append(convert_bomb_state_to_feature(bomb))
    for coin in game_state['coins']:
        channels.append(convert_coin_state_to_feature(coin))
    stacked_channels = np.stack(channels)
    return str(stacked_channels.reshape(-1))


def state_to_features_cnn(game_state: dict) -> np.array:
    if game_state is None:
        return np.zeros(INPUT_SHAPE)
    agent_features = np.zeros((MAX_AGENT_COUNT, 3))
    bomb_features = np.zeros((MAX_BOMB_COUNT, 3))
    coin_features = np.zeros((MAX_COIN_COUNT, 3))
    index = 0
    agent_features[index] = convert_agent_state_to_feature(game_state['self'])
    others_sorted = sort_others_consistently(game_state['others'])
    bombs_sorted = sort_bombs_consistently(game_state['bombs'])
    coins_sorted = sort_coins_consistently(game_state['coins'])
    for other in others_sorted:
        agent_features[index] = convert_agent_state_to_feature(other)
        index += 1
    index = 0
    for bomb in bombs_sorted:
        bomb_features[index] = convert_bomb_state_to_feature(bomb)
        index += 1
    index = 0
    for coin in coins_sorted:
        coin_features[index] = convert_coin_state_to_feature(coin)
        index += 1
    stacked_channels = np.concatenate([agent_features, bomb_features, coin_features])
    return stacked_channels.reshape(INPUT_SHAPE)


def state_to_features_custom1(game_state: dict) -> np.array:
    if game_state is None:
        return None

    channels = []
    self_position = game_state['self'][3]
    channels.append(convert_agent_state_to_feature(game_state['self']))
    # for other in game_state['others']:
    #    if dist(self_position, other[3]):
    #        channels.append(convert_agent_state_to_feature(other))
    for bomb in game_state['bombs']:
        channels.append(convert_bomb_state_to_feature(bomb))
    for coin in game_state['coins']:
        if dist(self_position, coin):
            channels.append(convert_coin_state_to_feature(coin))
    stacked_channels = np.stack(channels)
    return str(stacked_channels.reshape(-1))


def state_to_features_limited_detection(game_state: dict) -> np.array:
    if game_state is None:
        return None

    channels = []
    self_position = game_state['self'][3]
    channels.append(convert_agent_state_to_feature(game_state['self']))
    for other in game_state['others']:
        if dist(self_position, other[3]):
            channels.append(convert_agent_state_to_feature(other))
    for bomb in game_state['bombs']:
        if dist(self_position, bomb[0]):
            channels.append(convert_bomb_state_to_feature(bomb))
    for coin in game_state['coins']:
        if dist(self_position, coin):
            channels.append(convert_coin_state_to_feature(coin))
    stacked_channels = np.stack(channels)
    return str(stacked_channels.reshape(-1))


Entity = namedtuple('Entity',
                    ('category', 'position', 'extra_value'))


def state_to_features_n_closest(game_state: dict) -> np.array:
    if game_state is None:
        return None

    channels = []
    self_position = game_state['self'][3]
    channels.append(convert_agent_state_to_feature(game_state['self']))

    entities = []
    for other in game_state['others']:
        entities.append(feature_to_entity(1, convert_agent_state_to_feature(other)))
    for bomb in game_state['bombs']:
        entities.append(feature_to_entity(2, convert_bomb_state_to_feature(bomb)))
    for coin in game_state['coins']:
        entities.append(feature_to_entity(3, convert_coin_state_to_feature(coin)))

    entities.sort(key=lambda entity: dist(self_position, entity.position))
    entities = entities[:AMOUNT_ELEMENTS]
    entities.sort(key=lambda entity: entity.category)
    for entity in entities:
        channels.append(np.array([entity.position[0], entity.position[1], entity.extra_value]))
    stacked_channels = np.stack(channels)
    return str(stacked_channels.reshape(-1))


def sort_others_consistently(others: np.array) -> np.array:
    coordinates = np.array([other[3] for other in others], dtype=(np.dtype([('x', int), ('y', int)])))
    sorted_indices = np.argsort(coordinates, order=('x', 'y'))
    return [others[i] for i in sorted_indices]


def sort_bombs_consistently(bombs: np.array) -> np.array:
    coordinates = np.array([bomb[0] for bomb in bombs], dtype=(np.dtype([('x', int), ('y', int)])))
    sorted_indices = np.argsort(coordinates, order=('x', 'y'))
    return [bombs[i] for i in sorted_indices]


def sort_coins_consistently(coins: np.array) -> np.array:
    coordinates = np.array(coins, dtype=(np.dtype([('x', int), ('y', int)])))
    sorted_indices = np.argsort(coordinates, order=('x', 'y'))
    return [coins[i] for i in sorted_indices]


def convert_agent_state_to_feature(agent_state):
    return np.array([agent_state[3][0], agent_state[3][1], 1 if agent_state[2] else 0])


def convert_bomb_state_to_feature(bomb_state):
    return np.array([bomb_state[0][0], bomb_state[0][1], bomb_state[1]])


def convert_coin_state_to_feature(coin_state):
    return np.array([coin_state[0], coin_state[1], -1])


def feature_to_entity(order, feature):
    position = np.array([feature[0], feature[1]])
    return Entity(order, position, feature[2])


def dist(vector_tuple1, vector_tuple2):
    return np.abs(vector_tuple1[0] - vector_tuple2[0]) <= DETECTION_RADIUS and np.abs(
        vector_tuple1[1] - vector_tuple2[1]) <= DETECTION_RADIUS

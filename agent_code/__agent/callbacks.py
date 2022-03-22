import os
import pickle
import random
from collections import namedtuple

import numpy as np

from agent_code.__agent.constants import \
    ACTIONS, EPSILON, DETECTION_RADIUS, AMOUNT_ELEMENTS, BATCH_SIZE, INPUT_SHAPE, MAX_AGENT_COUNT, \
    MAX_BOMB_COUNT, MAX_COIN_COUNT, MAX_CRATE_COUNT, MAIN_MODEL_FILE_PATH, MIN_ALLOWED_BOMB_TIMER, CENTER_POSITION, \
    MIRRORED_ACTIONS_BY_ACTION_Y_AXIS, MIRRORED_ACTIONS_BY_ACTION_X_AXIS, MIN_FRACTION, VALIDATION_PERFORMANCE_ROUNDS
import tensorflow.keras.optimizers
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.client import device_lib


def setup(self):
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("-- No GPU --")
    self.statistics = None
    self.epsilon = EPSILON()
    self.batch_size = BATCH_SIZE
    self.model = init_model()
    self.previous_bombs = []
    self.min_batch_fraction_size = MIN_FRACTION * BATCH_SIZE
    self.prev_game_state = None
    self.validation_rounds = VALIDATION_PERFORMANCE_ROUNDS
    if os.path.isdir(MAIN_MODEL_FILE_PATH):
        self.model = tensorflow.keras.models.load_model(MAIN_MODEL_FILE_PATH)

    if self.train:
        self.target_model = clone_model(self.model)
        self.model_updates = 0
        self.target_model.set_weights(self.model.get_weights())


def init_model():
    model = Sequential()
    model.add(tf.keras.Input(shape=INPUT_SHAPE))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(len(ACTIONS), activation='linear'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(loss='mse', optimizer=opt)
    return model


def act(self, game_state: dict):
    if self.validation_rounds > 0 or random.random() > self.epsilon:
        self_position = game_state['self'][3]
        features = np.expand_dims(state_to_features(self, game_state), axis=0)
        q_values = self.model.predict(features)[0]
        indices_best_actions = np.flip(np.argsort(q_values))

        for index in indices_best_actions:
            action = ACTIONS[index]
            if self.prev_game_state is not None and (self.prev_game_state == features).all() and action == 'WAIT':
                continue
            elif not is_action_valid(game_state['self'], game_state['field'], action):
                continue
            self.prev_game_state = features
            return action

        self.prev_game_state = features
        return ACTIONS[indices_best_actions[0]]
    else:
        return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'], p=[.21, .21, .21, .21, .06, 0.1])


def is_action_valid(self_data, field, action):
    if action == 'BOMB':
        return self_data[2]

    self_position = self_data[3]
    if action == "RIGHT":
        return field[self_position[0] + 1, self_position[1]] == 0
    if action == "LEFT":
        return field[self_position[0] - 1, self_position[1]] == 0
    if action == "UP":
        return field[self_position[0], self_position[1] - 1] == 0
    if action == "DOWN":
        return field[self_position[0], self_position[1] + 1] == 0
    else:
        return True


def state_to_features(self, game_state: dict) -> np.array:
    features = state_to_features_cnn(self, game_state)
    return features


def state_to_features_cnn(self, game_state: dict) -> np.array:
    if game_state is None:
        return np.zeros(INPUT_SHAPE)
    agent_features = np.zeros((MAX_AGENT_COUNT, 3))
    valid_actions_feature = get_valid_actions_feature(game_state['self'], game_state['field'])
    bomb_features = np.zeros((MAX_BOMB_COUNT, 3))
    coin_features = np.zeros((MAX_COIN_COUNT, 2))
    crate_features = np.zeros((MAX_CRATE_COUNT, 2))
    index = 0
    agent_features[index] = convert_agent_state_to_feature(game_state['self'])
    index += 1
    others_sorted = sort_others_consistently(game_state['others'])
    bombs_sorted = sort_bombs_consistently(game_state['bombs']+self.previous_bombs)
    coins_sorted = sort_coins_consistently(game_state['coins'])
    # crates_sorted = np.argwhere(game_state['field'] == 1)
    self.previous_bombs.clear()
    for other in others_sorted:
        agent_features[index] = convert_agent_state_to_feature(other)
        index += 1
    index = 0
    for bomb in bombs_sorted:
        bomb_feature = convert_bomb_state_to_feature(bomb)
        bomb_features[index] = bomb_feature
        index += 1

        if bomb_feature[2] <= 0 and bomb_feature[2] > MIN_ALLOWED_BOMB_TIMER:
            bomb_updated = (bomb[0], bomb[1] - 1)
            self.previous_bombs.append(bomb_updated)
    index = 0
    for coin in coins_sorted:
        coin_features[index] = convert_coin_state_to_feature(coin)
        index += 1
    index = 0
    # for crate in crates_sorted:
    #     crate_features[index] = convert_crate_state_to_feature(crate)
    #     index += 1

    vector = np.concatenate([agent_features.reshape(-1), valid_actions_feature, bomb_features.reshape(-1), coin_features.reshape(-1), crate_features.reshape(-1)])
    return vector.reshape(-1)


def get_valid_actions_feature(self_data, game_field):
    self_position = self_data[3]
    can_plant_bomb = self_data[2]

    can_move_left = game_field[self_position[0] - 1, self_position[1]] == 0
    can_move_up = game_field[self_position[0], self_position[1] - 1] == 0 # Yes, it's has to be -.
    can_move_right = game_field[self_position[0] + 1, self_position[1]] == 0
    can_move_down = game_field[self_position[0], self_position[1] + 1] == 0

    return np.array([can_move_right, can_move_left, can_move_up, can_move_down, can_plant_bomb])


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
    return np.array([coin_state[0], coin_state[1]])


def convert_crate_state_to_feature(crate_state):
    return np.array([crate_state[0], crate_state[1]])


def convert_explosion_state_to_feature(explosion_coordinate, explosion_duration):
    return np.array([explosion_coordinate[0], explosion_coordinate[1], explosion_duration])


def feature_to_entity(order, feature):
    position = np.array([feature[0], feature[1]])
    return Entity(order, position, feature[2])


def dist(vector_tuple1, vector_tuple2):
    return np.abs(vector_tuple1[0] - vector_tuple2[0]) <= DETECTION_RADIUS and np.abs(
        vector_tuple1[1] - vector_tuple2[1]) <= DETECTION_RADIUS

def mirror_features(game_state):
    new_game_dict = {}
    agent_data = game_state['self']
    others_data = game_state['others']
    bombs_data = game_state['bombs']
    coins_data = game_state['coins']
    game_field = game_state['field'].copy()
    self_position = (agent_data[3][0], agent_data[3][1])
    if self_position[0] > CENTER_POSITION[0]:
        # Mirror by y-axis.
        game_field = np.fliplr(game_field)
        agent_data = (agent_data[0], agent_data[1], agent_data[2], get_mirrored_position(agent_data[3], False))
        new_others_data = []
        for other in others_data:
            new_others_data.append((other[0], other[1], other[2], get_mirrored_position(other[3], False)))
        others_data = new_others_data
        new_bombs_data = []
        for bomb in bombs_data:
            new_bombs_data.append((get_mirrored_position(bomb[0], False), bomb[1]))
        bombs_data = new_bombs_data
        new_coins_data = []
        for coin in coins_data:
            new_coins_data.append((get_mirrored_position(coin, False)))
        coins_data = new_coins_data
    if self_position[1] > CENTER_POSITION[1]:
        # Mirror by x-axis.
        game_field = np.flipud(game_field)
        agent_data = (agent_data[0], agent_data[1], agent_data[2], get_mirrored_position(agent_data[3], True))
        new_others_data = []
        for other in others_data:
            new_others_data.append((other[0], other[1], other[2], get_mirrored_position(other[3], True)))
        others_data = new_others_data
        new_bombs_data = []
        for bomb in bombs_data:
            new_bombs_data.append((get_mirrored_position(bomb[0], True), bomb[1]))
        bombs_data = new_bombs_data
        new_coins_data = []
        for coin in coins_data:
            new_coins_data.append((get_mirrored_position(coin, True)))
        coins_data = new_coins_data
    new_game_dict['self'] = agent_data
    new_game_dict['others'] = others_data
    new_game_dict['bombs'] = bombs_data
    new_game_dict['coins'] = coins_data
    new_game_dict['field'] = game_field
    return new_game_dict


def mirror_action(self_position, action):
    returned_action = action
    if self_position[0] > CENTER_POSITION[0]:
        returned_action = MIRRORED_ACTIONS_BY_ACTION_Y_AXIS[returned_action]
    if self_position[1] > CENTER_POSITION[1]:
        returned_action = MIRRORED_ACTIONS_BY_ACTION_X_AXIS[returned_action]
    return returned_action


def get_mirrored_position(position_tuple, x_axis = True):
    diff = (CENTER_POSITION[0] - position_tuple[0], CENTER_POSITION[1] - position_tuple[1])
    if x_axis:
        return (position_tuple[0], CENTER_POSITION[1] + diff[1])
    else:
        # Y-Axis.
        return (CENTER_POSITION[0] + diff[0], position_tuple[1])

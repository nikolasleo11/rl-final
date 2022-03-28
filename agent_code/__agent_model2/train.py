from collections import namedtuple
import pickle
from typing import List

from agent_code.__agent.constants import INDICES_BY_ACTION, \
    MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN, GENERATE_STATISTICS, EPSILON_UPDATE_RATE, DECAY, \
    EPSILON, BATCH_SIZE, ACTIONS, DISCOUNT, ROUNDS_TO_PLOT, NEUTRAL_REWARD, MIN_FRACTION, \
    TARGET_MODEL_UPDATE_RATE, TRAINING_DATA_MODE, VALIDATION_PERFORMANCE_ROUNDS, INPUT_SHAPE, \
    LEARNING_FACTOR, DESTROY_CRATE_MODEL_FILE_PATH, SEEK_COIN_MODEL_FILE_PATH, DODGE_BOMB_MODEL_FILE_PATH, \
    DESTROY_ENEMY_MODEL_FILE_PATH
import numpy as np
import events as e
from .callbacks import state_to_features
from .statistics_data import RoundBasedStatisticsData, NeuralNetworkData

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    self.transitions = []
    self.behaviour_returns = []
    self.prev_behaviour = (self.current_behaviour[0], self.current_behaviour[1], self.current_behaviour[2])
    if GENERATE_STATISTICS:
        self.statistics = NeuralNetworkData()
        self.statistics.append_validation()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    if old_game_state == None:
        return

    transition = Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state),
                            reward_from_events(self, self_action, new_game_state, events))

    append_transition_data(self, transition)
    if GENERATE_STATISTICS:
        self.statistics.update_transition_statistics(transition.reward)
        if self.validation_rounds > 0:
            self.statistics.update_validation_statistics(VALIDATION_PERFORMANCE_ROUNDS - self.validation_rounds,
                                                         transition.reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    round_number = last_game_state['round']
    transition = Transition(state_to_features(self, last_game_state), last_action, None,
                            reward_from_events(self, last_action, None, events))
    append_transition_data(self, transition)
    if self.validation_rounds > 0:
        self.validation_rounds -= 1
    if DECAY and round_number % EPSILON_UPDATE_RATE == 0:
        self.epsilon = EPSILON(self.epsilon)
        if GENERATE_STATISTICS:
            self.statistics.add_epsilon(self.epsilon)
    if round_number % MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN == 0:
        with open(SEEK_COIN_MODEL_FILE_PATH, "wb") as file:
            pickle.dump(self.seek_coin_evaluator, file)
        with open(DESTROY_CRATE_MODEL_FILE_PATH, "wb") as file:
            pickle.dump(self.destroy_crate_evaluator, file)
        with open(DODGE_BOMB_MODEL_FILE_PATH, "wb") as file:
            pickle.dump(self.dodge_bomb_evaluator, file)
        with open(DESTROY_ENEMY_MODEL_FILE_PATH, "wb") as file:
            pickle.dump(self.destroy_enemy_evaluator, file)
    if GENERATE_STATISTICS:
        self.statistics.update_round_statistics()
        if round_number % ROUNDS_TO_PLOT == 0:
            self.statistics.plot()


def reward_from_events(self, action, new_game_state, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 25,
        e.KILLED_OPPONENT: 100,
        e.INVALID_ACTION: -20,
        e.COIN_FOUND: 5,
        e.GOT_KILLED: -15,
        e.KILLED_SELF: -15,
        e.WAITED: -5
    }
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    max_bomb_danger_points = -20
    bomb_targets_crate = 10
    bomb_targets_enemy = 20

    if new_game_state is not None:
        self_position = new_game_state['self'][3]
        for bomb in new_game_state['bombs']:
            bomb_position = bomb[0]
            dist = (bomb_position[0] - self_position[0], bomb_position[1] - self_position[1])
            if (dist[0] == 0 and abs(dist[1]) <= 3) or (dist[1] == 0 and abs(dist[0]) <= 3):
                reward_sum += max_bomb_danger_points / 4 * (4 - bomb[1])
            if dist[0] == 0 and dist[1] == 0 and bomb[1]==3:
                # Bomb has been planted.

                predicted_explosions_in_map = np.copy(new_game_state['field'])
                predicted_explosions_in_map[predicted_explosions_in_map == 1] = 0
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                explosion_power = 3
                bomb_pos = bomb[0]
                predicted_explosions_in_map[bomb_pos[0], bomb_pos[1]] = 1
                for direction in directions:
                    for i in range(explosion_power):
                        j = i + 1
                        new_pos = (bomb_pos[0] + j * direction[0], bomb_pos[1] + j * direction[1])
                        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= predicted_explosions_in_map.shape[
                            0] or new_pos[1] >= predicted_explosions_in_map.shape[1] or \
                                predicted_explosions_in_map[new_pos[0], new_pos[1]] != 0:
                            break
                        predicted_explosions_in_map[new_pos[0], new_pos[1]] = 1
                        if new_game_state['field'][new_pos[0], new_pos[1]] == 1:
                            reward_sum += bomb_targets_crate
                        for others in new_game_state['others']:
                            others_position = others[3]
                            if new_pos[0] == others_position[0] and new_pos[1] == others_position[1]:
                                reward_sum += bomb_targets_enemy

    if reward_sum == 0:
        reward_sum = NEUTRAL_REWARD
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def append_transition_data(self, transition):
    self.transitions.append(transition)
    behaviour_is_done = self.prev_behaviour[0] == self.current_behaviour[0]
    c1 = self.prev_behaviour[1] == self.current_behaviour[1]
    c2 = self.prev_behaviour[2] == self.current_behaviour[2]
    is_first_element_array = isinstance(c1, np.ndarray)
    is_second_element_array = isinstance(c2, np.ndarray)
    if is_first_element_array:
        behaviour_is_done = behaviour_is_done and c1.all()
    else:
        behaviour_is_done = behaviour_is_done and c1
    if is_second_element_array:
        behaviour_is_done = behaviour_is_done and c2.all()
    else:
        behaviour_is_done = behaviour_is_done and c2
    if behaviour_is_done and self.prev_behaviour[0]:
        self.transitions.reverse()
        return_ = 0
        for transition in self.transitions:
            return_ = transition.reward + DISCOUNT * return_
        self.behaviour_returns.append([self.prev_behaviour[0], self.prev_behaviour[2], return_])
        self.transitions.clear()
        if len(self.behaviour_returns) > self.required_batch_size:
            train_models(self)
    self.prev_behaviour = (self.current_behaviour[0], self.current_behaviour[1], self.current_behaviour[2])

def train_models(self):
    transition_data_by_behaviour = {}
    for behaviour_return in self.behaviour_returns:
        if not behaviour_return[0] in transition_data_by_behaviour:
            transition_data_by_behaviour[behaviour_return[0]] = []
        transition_data_by_behaviour[behaviour_return[0]].append([behaviour_return[1], behaviour_return[2]])
    for behaviour_name in transition_data_by_behaviour:
        data = transition_data_by_behaviour[behaviour_name]
        affected_model = None
        if behaviour_name == "COIN":
            affected_model = self.seek_coin_evaluator
        elif behaviour_name == "CRATE":
            affected_model = self.destroy_crate_evaluator
        elif behaviour_name == "DODGE":
            affected_model = self.dodge_bomb_evaluator
        elif behaviour_name == "ENEMY":
            affected_model = self.destroy_enemy_evaluator
        else:
            raise ArithmeticError

        X = np.array([row[0] for row in data])
        y = np.array([row[1] for row in data])

        predicted_y = affected_model.predict(X).reshape(-1)
        fitted_y = predicted_y + LEARNING_FACTOR * (y - predicted_y)
        affected_model.fit(X, fitted_y)
    self.required_batch_size += BATCH_SIZE
    self.validation_rounds = VALIDATION_PERFORMANCE_ROUNDS + 1
    self.statistics.append_validation()


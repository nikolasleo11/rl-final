from collections import namedtuple
import pickle
from typing import List

import tensorflow.keras.callbacks

from agent_code.__agent.constants import INDICES_BY_ACTION, \
    MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN, GENERATE_STATISTICS, EPSILON_UPDATE_RATE, DECAY, \
    EPSILON, BATCH_SIZE, ACTIONS, MAIN_MODEL_FILE_PATH, DISCOUNT, ROUNDS_TO_PLOT, NEUTRAL_REWARD, MIN_FRACTION, \
    TARGET_MODEL_UPDATE_RATE, TRAINING_DATA_MODE, MEMORY_SIZE, VALIDATION_PERFORMANCE_ROUNDS, INPUT_SHAPE
import numpy as np
import events as e
import tensorflow as tf
from .callbacks import state_to_features
from .statistics_data import RoundBasedStatisticsData, NeuralNetworkData

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    self.transitions = []
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
    # Idea: Add your own events to hand out rewards
    # if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if old_game_state == None:
        return

    transition = Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state),
                            reward_from_events(self, new_game_state, events))

    self.transitions.append(transition)
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
                            reward_from_events(self, None, events))
    append_and_train(self, transition)
    if self.validation_rounds > 0:
        self.validation_rounds -= 1
    if DECAY and round_number % EPSILON_UPDATE_RATE == 0:
        self.epsilon = EPSILON(self.epsilon)
        self.min_batch_fraction_size = EPSILON(self.min_batch_fraction_size)
        if GENERATE_STATISTICS:
            self.statistics.add_epsilon(self.epsilon)
    if round_number % MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN == 0:
        self.model.save(MAIN_MODEL_FILE_PATH)
    if GENERATE_STATISTICS:
        self.statistics.update_round_statistics()
        if round_number % ROUNDS_TO_PLOT == 0:
            self.statistics.plot()


def reward_from_events(self, new_game_state, events: List[str]) -> int:
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

    max_coin_aura_points = 10
    max_bomb_danger_points = -12

    if new_game_state is not None:
        self_position = new_game_state['self'][3]
        auras = np.copy(new_game_state['field'])
        auras[auras==1] = -1
        fields_to_check = np.full(auras.shape, False)

        for coin in new_game_state['coins']:
            auras[coin[1], coin[0]] = max_coin_aura_points + 1
            fields_to_check[coin[1], coin[0]] = True
            is_done = False
            updated_aura = max_coin_aura_points
            while not is_done or updated_aura > 0:
                is_done = True
                field_copy = np.copy(fields_to_check)
                for y, row in enumerate(fields_to_check):
                    for x, element in enumerate(row):
                        if element:
                            field_copy[y, x] = False
                            dists_to_check = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                            for dist in dists_to_check:
                                aura = auras[y + dist[1], x + dist[0]]
                                if aura > -1 and aura < updated_aura:
                                    is_done = False
                                    field_copy[y + dist[1], x + dist[0]] = True
                                    auras[y + dist[1], x + dist[0]] = updated_aura

                fields_to_check = field_copy
                updated_aura -= 1
        auras[auras == -1] = 0
        reward_sum += auras[self_position[1], self_position[0]]
        for bomb in new_game_state['bombs']:
            bomb_position = bomb[0]
            dist = (bomb_position[0] - self_position[0], bomb_position[1] - self_position[1])
            if dist[0] == 0 and abs(dist[1]) <= 3:
                reward_sum += max_bomb_danger_points / 4 * (4 - bomb[1])

    if reward_sum == 0:
        reward_sum = NEUTRAL_REWARD
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def append_and_train(self, transition):
    self.transitions.append(transition)
    if GENERATE_STATISTICS:
        self.statistics.update_transition_statistics(transition.reward)
        if self.validation_rounds > 0:
            self.statistics.update_validation_statistics(VALIDATION_PERFORMANCE_ROUNDS - self.validation_rounds, transition.reward)
    batch = get_training_batch(self)
    if len(batch) >= BATCH_SIZE:
        X = []
        ys = []
        next_states = np.array([t.next_state if t.next_state is not None else np.zeros(INPUT_SHAPE) for t in batch])
        current_states = np.array([t.state for t in batch])
        q_values_next_states = self.target_model.predict(next_states)
        q_values_current_states = self.model.predict(current_states)
        for i, transition in enumerate(batch):
            index_action = INDICES_BY_ACTION[transition.action]
            q_value_next_state = q_values_next_states[i]
            expected_return = transition.reward
            if transition.next_state is not None:
                expected_return += DISCOUNT * np.max(q_value_next_state)
            q_values_state = q_values_current_states[i]
            q_values_state[index_action] = expected_return
            X.append(transition.state)
            ys.append(q_values_state)
        history = tensorflow.keras.callbacks.History()
        self.model.fit(np.array(X), np.array(ys), batch_size=BATCH_SIZE, verbose=1, callbacks=[history])
        self.transitions.clear()

        if GENERATE_STATISTICS:
            self.statistics.update_model_statistics(history.history['loss'][0])
        self.model_updates += 1
        if self.model_updates % TARGET_MODEL_UPDATE_RATE == 0:
            print("Updating the target model.")
            self.target_model.set_weights(self.model.get_weights())

        self.validation_rounds = VALIDATION_PERFORMANCE_ROUNDS + 1 # Auto -1 after this function.
        self.statistics.append_validation()


def get_training_batch(self):
    transitions = self.transitions
    # 0 = Use unmodified batch, 1 = use balanced batch, 2 = use subsample of bigger batch as batch, 3 = 1 + 2
    if TRAINING_DATA_MODE == 0:
        return transitions
    if TRAINING_DATA_MODE == 1:
        return get_balanced_batch_if_possible(self)
    if TRAINING_DATA_MODE == 2:
        return get_subsample_if_possible(transitions)
    else:
        if len(transitions) > MEMORY_SIZE:
            return get_balanced_batch_if_possible(self)
        else:
            return np.empty(0)


def get_balanced_batch_if_possible(self):
    transitions = self.transitions
    if len(transitions) < BATCH_SIZE:
        return np.empty(0)
    positive_transitions = [transition for transition in transitions if transition.reward > NEUTRAL_REWARD]
    neutral_transitions = [transition for transition in transitions if transition.reward == NEUTRAL_REWARD]
    negative_transitions = [transition for transition in transitions if transition.reward < NEUTRAL_REWARD]
    minimum = np.min([len(positive_transitions), len(neutral_transitions), len(negative_transitions)])
    if minimum < self.min_batch_fraction_size:
        return np.empty(0)
    np.random.shuffle(positive_transitions)
    np.random.shuffle(neutral_transitions)
    np.random.shuffle(negative_transitions)
    transition_groups = [positive_transitions, neutral_transitions, negative_transitions]
    transition_groups = sorted(transition_groups, key=lambda x: len(x), reverse=False)
    minimum = min(minimum, int(BATCH_SIZE / 3))
    remaining_size = BATCH_SIZE - minimum
    size0 = minimum
    size1 = min(round(remaining_size / 2), len(transition_groups[1]))
    size2 = BATCH_SIZE - size0 - size1
    returned = transition_groups[0][:size1] + transition_groups[1][:size1] + transition_groups[2][:size2]
    np.random.shuffle(returned)
    return returned


def get_subsample_if_possible(transitions):
    if len(transitions) < MEMORY_SIZE:
        return np.empty(0)
    np.random.shuffle(transitions)
    return transitions[:BATCH_SIZE]

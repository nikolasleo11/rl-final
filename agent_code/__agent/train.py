from collections import namedtuple
import pickle
from typing import List

import keras.callbacks

from agent_code.__agent.constants import INDICES_BY_ACTION, \
    MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN, GENERATE_STATISTICS, EPSILON_UPDATE_RATE, DECAY, \
    EPSILON, BATCH_SIZE, ACTIONS, MAIN_MODEL_FILE_PATH, DISCOUNT, ROUNDS_TO_PLOT
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
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if old_game_state == None:
        return
    transition = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))

    append_and_train(self, transition)


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
    transition = Transition(state_to_features(last_game_state), last_action, state_to_features(None), reward_from_events(self, events))
    append_and_train(self, transition)
    if DECAY and round_number % EPSILON_UPDATE_RATE == 0:
        self.epsilon = EPSILON(self.epsilon)
        if GENERATE_STATISTICS:
            self.statistics.add_epsilon(self.epsilon)
    if round_number % MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN == 0:
        self.model.save(MAIN_MODEL_FILE_PATH)
    if GENERATE_STATISTICS:
        self.statistics.update_round_statistics(0)
        if round_number % ROUNDS_TO_PLOT == 0:
            self.statistics.plot()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -100,
        e.COIN_FOUND: 2,
        e.GOT_KILLED: -50,
        e.KILLED_SELF: -100,
        e.SURVIVED_ROUND: 100
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    if reward_sum == 0:
        reward_sum = -1
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        
    return reward_sum

  
def append_and_train(self, transition):
    self.transitions.append(transition)
    is_terminal = transition.next_state is None
    if GENERATE_STATISTICS:
        self.statistics.update_transition_statistics(transition.reward)
    if len(self.transitions) >= BATCH_SIZE:
        X = []
        ys = []
        for i, transition in enumerate(self.transitions):
            index_action = INDICES_BY_ACTION[transition.action]
            q_values_next_state = self.model.predict(np.expand_dims(transition.next_state, axis=0))[0]
            expected_return = transition.reward + DISCOUNT * np.max(q_values_next_state) if not is_terminal else transition.reward
            q_values_state = self.model.predict(np.expand_dims(transition.state, axis=0))[0]
            q_values_state[index_action] = expected_return
            X.append(transition.state)
            ys.append(q_values_state)
        history = keras.callbacks.History()
        self.model.fit(np.array(X), np.array(ys), batch_size=BATCH_SIZE, verbose=1, callbacks=[history])
        self.transitions.clear()

        if GENERATE_STATISTICS:
            self.statistics.update_model_statistics(history.history['loss'][0])


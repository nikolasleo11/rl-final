from collections import namedtuple, deque
import matplotlib.pyplot as plt
import pickle
from typing import List
from agent_code.__agent.constants import INDICES_BY_ACTION, SAVED_Q_VALUES_FILE_PATH, SAVED_INDICES_BY_STATE_FILE_PATH, \
    ACTIONS, LEARNING_FACTOR, MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN, GENERATE_STATISTICS, TRANSITION_HISTORY_SIZE, RECORD_ENEMY_TRANSITIONS
import numpy as np

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    if GENERATE_STATISTICS:
        self.statistics = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

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
    transition = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))
    update_q_values(self, transition)


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
    transition = Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events))
    update_q_values(self, transition)
    self.rounds_not_saved += 1
    # Store the model
    if self.rounds_not_saved >= MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN:
        self.rounds_not_saved = 0
        with open(SAVED_Q_VALUES_FILE_PATH, "wb") as file:
            pickle.dump(self.q_values, file)
        with open(SAVED_INDICES_BY_STATE_FILE_PATH, "wb") as file:
            pickle.dump(self.indices_by_state, file)

        if GENERATE_STATISTICS:
            s = self.statistics.T
            plt.plot(s[3], s[0] / s[2], '.')
            plt.ylabel('New discovered States in %')
            plt.xlabel('Amount of Rounds')
            plt.show()
            plt.plot(s[3], s[1], '.')
            plt.ylabel('Rewards per Round')
            plt.xlabel('Amount of Rounds')
            plt.show()
            print(s)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 500,
        e.INVALID_ACTION: -20,
        e.COIN_FOUND: 20,
        e.GOT_KILLED: -50,
        e.KILLED_SELF: -1000,
        e.OPPONENT_ELIMINATED: 100,
        e.SURVIVED_ROUND: 100,
        e.BOMB_DROPPED: -500
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    if reward_sum == 0:
        reward_sum = -1
        
    return reward_sum


def append_state_if_not_covered_yet(self, state):
    state_str = str(state)
    if len(self.indices_by_state) != self.q_values.shape[0]:
        raise IndexError("The amount of states in the mapping and the q table don't match. "
                         + str(len(self.indices_by_state)) + " vs " + str(self.q_values.shape[0]))
    if state_str not in self.indices_by_state:
        self.indices_by_state[state_str] = len(self.indices_by_state) - 1
        self.q_values = np.append(self.q_values, [np.zeros(len(ACTIONS))], axis=0)


def update_q_values(self, transition):
    '''Updates the q-values using the bellman expectancy equation with greedy policy improvement.'''

    if GENERATE_STATISTICS:
        generate_statistics(self, transition)

    append_state_if_not_covered_yet(self, transition.state)
    append_state_if_not_covered_yet(self, transition.next_state)

    index_state = self.indices_by_state[str(transition.state)]
    index_next_state = self.indices_by_state[str(transition.next_state)]
    index_action = INDICES_BY_ACTION[transition.action]

    self.q_values[index_state, index_action] += LEARNING_FACTOR * (transition.reward + np.max(self.q_values[index_next_state]) - self.q_values[index_state, index_action])


def generate_statistics(self, transition):
    is_final_round = transition.next_state is None
    state_str = str(transition.state)
    state2_str = str(transition.next_state)

    self.statistics[len(self.statistics) - 1][1] += transition.reward
    self.statistics[len(self.statistics) - 1][2] += 1
    if state_str not in self.indices_by_state:
        self.statistics[len(self.statistics) - 1][0] += 1
        
    if state2_str not in self.indices_by_state:
        self.statistics[len(self.statistics) - 1][0] += 1

    if is_final_round:
        self.statistics[len(self.statistics) - 1][0] += self.statistics[len(self.statistics) - 2][0]
        self.statistics[len(self.statistics) - 1][2] += self.statistics[len(self.statistics) - 2][2]
        self.statistics[len(self.statistics) - 1][3] = self.statistics[len(self.statistics) - 2][3] + 1
        self.statistics = np.append(self.statistics, [[0, 0, 0, 0]], axis=0)

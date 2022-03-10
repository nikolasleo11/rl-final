import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from agent_code.__agent.constants import DECAY, STATISTICS_PLOTS_FOLDER_PATH


class RoundBasedStatisticsData:
    def __init__(self, agent, expanded_mode=False, amount_x=100, drop_data_after_saving=True):
        self.drop_data_after_saving = drop_data_after_saving
        self.agent = agent
        self.amount_x = amount_x
        self.total_amount_of_new_states = [0]
        self.total_amount_of_transitions = [0]
        self.total_amount_of_states_per_round = [0]
        self.total_amount_of_moves_per_round = [0]
        self.total_rewards_per_round = [0]
        self.expanded_mode = expanded_mode
        if DECAY:
            self.epsilons = [self.agent.epsilon]
        if self.expanded_mode:
            self.total_received_amount_of_states = []
            self.q_value_deltas_per_state = []

    def update_statistics(self, transition):
        is_final_round = transition.next_state is None
        self.total_rewards_per_round[-1] += transition.reward
        self.total_amount_of_moves_per_round[-1] += 1
        self.total_amount_of_transitions[-1] += 1

        if transition.state not in self.agent.indices_by_state:
            self.total_amount_of_new_states[-1] += 1

        if transition.next_state not in self.agent.indices_by_state:
            self.total_amount_of_new_states[-1] += 1

        if is_final_round:
            self.total_amount_of_states_per_round[-1] = self.agent.q_values.shape[0]
            self.total_amount_of_new_states.append(0)
            self.total_amount_of_transitions.append(0)
            self.total_amount_of_states_per_round.append(0)
            self.total_amount_of_moves_per_round.append(0)
            self.total_rewards_per_round.append(0)

    def update_expanded_statistics(self, index_state, delta):
        if self.expanded_mode:
            while index_state > len(self.total_received_amount_of_states) - 1:
                self.total_received_amount_of_states.append(0)
                self.q_value_deltas_per_state.append([])
            self.total_received_amount_of_states[index_state] += 1
            self.q_value_deltas_per_state[index_state].append(delta)

    def add_epsilon(self, epsilon):
        self.epsilons.append(epsilon)

    def plot(self, show_plot=True, save=False):
        save_path = STATISTICS_PLOTS_FOLDER_PATH + "/" + str(int(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        if save and not os.path.exists(STATISTICS_PLOTS_FOLDER_PATH):
            os.makedirs(STATISTICS_PLOTS_FOLDER_PATH)
        amount_x = min(len(self.total_rewards_per_round), self.amount_x)
        batch_size = math.floor(len(self.total_rewards_per_round) / amount_x)
        x_max = (len(self.total_rewards_per_round) - len(self.total_rewards_per_round) % batch_size)
        xs = np.array(range(round(x_max / batch_size)))
        total_amount_of_states_per_round = np.array(self.total_amount_of_states_per_round)[:x_max]
        total_amount_of_moves_per_round = np.array(self.total_amount_of_moves_per_round)[:x_max]
        total_rewards_per_round = np.array(self.total_rewards_per_round)[:x_max]

        plt.scatter(xs, np.mean((np.array(self.total_amount_of_new_states)[:x_max] / np.array(
            self.total_amount_of_transitions)[:x_max]).reshape(-1, batch_size), axis=1))
        plt.title("Average ratio of new states vs total over time")
        plt.xlabel("* " + str(batch_size) + " Rounds")
        plt.ylabel("Average ratio of new states vs total")
        if save:
            plt.savefig(save_path + 'a.png')
        if show_plot:
            plt.show()

        plt.scatter(xs, np.mean(total_amount_of_states_per_round.reshape(-1, batch_size), axis=1))
        plt.title("Average amount of discovered states over time")
        plt.xlabel("* " + str(batch_size) + " Rounds")
        plt.ylabel("Average amount of discovered states")
        if save:
            plt.savefig(save_path + 'b.png')
        if show_plot:
            plt.show()

        plt.scatter(xs, np.mean(total_amount_of_moves_per_round.reshape(-1, batch_size), axis=1))
        plt.title("Average amount of moves per round over time")
        plt.xlabel("* " + str(batch_size) + " Rounds")
        plt.ylabel("Average amount of moves per round")
        if save:
            plt.savefig(save_path + 'c.png')
        if show_plot:
            plt.show()

        plt.scatter(xs, np.mean(total_rewards_per_round.reshape(-1, batch_size), axis=1))
        plt.title("Returns over time")
        plt.xlabel("* " + str(batch_size) + " Rounds")
        plt.ylabel("Average Returns")
        if save:
            plt.savefig(save_path + 'd.png')
        if show_plot:
            plt.show()

        if self.expanded_mode:
            total_received_amount_of_states = np.array(self.total_received_amount_of_states)
            total_received_amount_of_states = np.flip(np.sort(total_received_amount_of_states[(total_received_amount_of_states >= 1) & (total_received_amount_of_states <= np.max(total_received_amount_of_states) * 0.9)]))
            plt.plot(total_received_amount_of_states)
            plt.title("All states traversed once or more")
            plt.xlabel("Individual states")
            plt.ylabel("Amount")
            if save:
                plt.savefig(save_path + 'e.png')
            if show_plot:
                plt.show()

            q_value_deltas = np.array([np.array([np.mean(deltas), np.std(deltas)]) for deltas in self.q_value_deltas_per_state if len(deltas) > 0]).T

            plt.errorbar(np.array(range(q_value_deltas.shape[1])), q_value_deltas[0], q_value_deltas[1], fmt='.', ecolor='red', barsabove=True)
            plt.title("Mean q-value changes of states")
            plt.xlabel("Individual states")
            plt.ylabel("Mean change & deviation")

            if save:
                plt.savefig(save_path + 'f.png')
            if show_plot:
                plt.show()
        if DECAY:
            plt.scatter(np.array(range(len(self.epsilons))), self.epsilons)
            plt.title("Epsilon over time")
            plt.xlabel("Time")
            plt.ylabel("Epsilon")
            if save:
                plt.savefig(save_path + 'g.png')
            if show_plot:
                plt.show()

        if save and self.drop_data_after_saving:
            self.total_amount_of_new_states = [0]
            self.total_amount_of_transitions = [0]
            self.total_amount_of_states_per_round = [0]
            self.total_amount_of_moves_per_round = [0]
            self.total_rewards_per_round = [0]

            if self.expanded_mode:
                self.total_received_amount_of_states = []
                self.q_value_deltas_per_state = []

            if DECAY:
                self.epsilons = [self.agent.epsilon]
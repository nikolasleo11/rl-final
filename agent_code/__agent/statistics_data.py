import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

FOLDER_PATH = "saved_plots"


class RoundBasedStatisticsData:
    def __init__(self, agent, amount_x = 100, drop_data_after_saving = True):
        self.drop_data_after_saving = drop_data_after_saving
        self.agent = agent
        self.amount_x = amount_x
        self.total_amount_of_new_states = [0]
        self.total_amount_of_transitions = [0]
        self.total_amount_of_states_per_round = [0]
        self.total_amount_of_moves_per_round = [0]
        self.total_rewards_per_round = [0]

    def update_statistics(self, transition):
        is_final_round = transition.next_state is None
        self.total_rewards_per_round[-1] += transition.reward
        self.total_amount_of_moves_per_round[-1] += 1
        self.total_amount_of_transitions[-1] += 1

        if str(transition.state) not in self.agent.indices_by_state:
            self.total_amount_of_new_states[-1] += 1
        if str(transition.next_state) not in self.agent.indices_by_state:
            self.total_amount_of_new_states[-1] += 1

        if is_final_round:
            self.total_amount_of_states_per_round[-1] = self.agent.q_values.shape[0]
            self.total_amount_of_new_states.append(0)
            self.total_amount_of_transitions.append(0)
            self.total_amount_of_states_per_round.append(0)
            self.total_amount_of_moves_per_round.append(0)
            self.total_rewards_per_round.append(0)

    def plot(self, show_plot = True, save=False):
        save_path = FOLDER_PATH + "/" + str(int(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        if save and not os.path.exists(FOLDER_PATH):
            os.makedirs(FOLDER_PATH)
        amount_x = min(len(self.total_rewards_per_round), self.amount_x)
        batch_size = math.floor(len(self.total_rewards_per_round) / amount_x)
        x_max = (len(self.total_rewards_per_round) - len(self.total_rewards_per_round) % batch_size)
        xs = np.array(range(round(x_max / batch_size)))
        total_amount_of_states_per_round = np.array(self.total_amount_of_states_per_round)[:x_max]
        total_amount_of_moves_per_round = np.array(self.total_amount_of_moves_per_round)[:x_max]
        total_rewards_per_round = np.array(self.total_rewards_per_round)[:x_max]

        plt.scatter(xs, np.mean((np.array(self.total_amount_of_new_states) / np.array(self.total_amount_of_transitions)).reshape(-1, batch_size), axis=1))
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

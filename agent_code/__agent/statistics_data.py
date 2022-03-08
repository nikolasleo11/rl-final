import matplotlib.pyplot as plt
import numpy as np

class StatisticsData:
    def __init__(self, round_batch_size = 100):
        self.save = save
        self.batch_size = round_batch_size
        self.total_amount_of_states_per_round = []
        self.total_amount_of_moves_per_round = []
        self.total_rewards_per_round = []

    def update_statistics(self, transition):
        is_final_round = transition.next_state is None
        self.total_rewards_per_round[-1] += transition.reward
        self.total_amount_of_moves_per_round[-1] += 1

        if is_final_round:
            self.total_amount_of_states_per_round[-1] = self.agent.q_values.shape[0]
            self.total_amount_of_states_per_round.append(0)
            self.total_amount_of_moves_per_round.append(0)
            self.total_rewards_per_round.append(0)

    def plot(self):
        plt.plot(self.statistics[:][0] / self.statistics[:][1], self.statistics[:][3])
        plt.show()
        plt.plot(self.statistics[:][2], self.statistics[:][3])
        plt.show()

    def save_figures(self):

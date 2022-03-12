# play --my-agent __agent --n-rounds 4000 --scenario coin-heaven --train 1 --no-gui
# Actions
import numpy as np

ACTIONS = np.array(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])
INDICES_BY_ACTION = {
'RIGHT':0, 'LEFT':1, 'UP':2, 'DOWN':3, 'BOMB':4, 'WAIT':5
}
# File paths
MAIN_MODEL_FILE_PATH = "main_model"
MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN = 400

# Exploration
BASE_EPSILON = 0.9
DECAY = True
DECAY_RATE = 0.9
EPSILON = lambda prev_epsilon=BASE_EPSILON/DECAY_RATE: prev_epsilon*DECAY_RATE if DECAY else BASE_EPSILON
EPSILON_UPDATE_RATE = 400 # Rounds

# Training
LEARNING_FACTOR = 0.1
GENERATE_STATISTICS = False
STATISTICS_PLOTS_FOLDER_PATH = "saved_plots"
MAX_AGENT_COUNT = 4
MAX_BOMB_COUNT = 8
MAX_COIN_COUNT = 48
INPUT_SHAPE = (MAX_AGENT_COUNT+MAX_BOMB_COUNT+MAX_COIN_COUNT, 3, 1)
# State-to-feature
DETECTION_RADIUS = 1.1
AMOUNT_ELEMENTS = 5

# Networks
BATCH_SIZE = 1024

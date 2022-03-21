# play --my-agent __agent --n-rounds 4000 --scenario coin-heaven --train 1 --no-gui
# Actions
import numpy as np

ACTIONS = np.array(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])
INDICES_BY_ACTION = {
'RIGHT':0, 'LEFT':1, 'UP':2, 'DOWN':3, 'BOMB':4, 'WAIT':5
}
MIRRORED_ACTIONS_BY_ACTION_X_AXIS = {
'RIGHT':'RIGHT', 'LEFT':'LEFT', 'UP':'DOWN', 'DOWN':'UP', 'BOMB':'BOMB', 'WAIT':'WAIT'
}
MIRRORED_ACTIONS_BY_ACTION_Y_AXIS = {
'RIGHT':'LEFT', 'LEFT':'RIGHT', 'UP':'UP', 'DOWN':'DOWN', 'BOMB':'BOMB', 'WAIT':'WAIT'
}
CENTER_POSITION = (8,8)
MIN_ALLOWED_BOMB_TIMER = -2

# File paths
MAIN_MODEL_FILE_PATH = "main_model"
MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN = 1000

# Exploration
BASE_EPSILON = 1
DECAY = True
DECAY_RATE = 0.8
EPSILON = lambda prev_epsilon=BASE_EPSILON/DECAY_RATE: prev_epsilon*DECAY_RATE if DECAY else BASE_EPSILON
EPSILON_UPDATE_RATE = 12000 / 6 # Rounds
# For optimal decay, use: Decay rate: 0.985, update rate: Total rounds / 100

# Training
DISCOUNT = 0.9
LEARNING_FACTOR = 0.1
MAX_AGENT_COUNT = 4
MAX_BOMB_COUNT = 12
MAX_COIN_COUNT = 50
MAX_CRATE_COUNT = 0
INPUT_SHAPE = (MAX_AGENT_COUNT+MAX_BOMB_COUNT+MAX_COIN_COUNT+MAX_CRATE_COUNT)*3
NEUTRAL_REWARD = -1

# Statistics
GENERATE_STATISTICS = True
STATISTICS_PLOTS_FOLDER_PATH = "saved_plots"
ROUNDS_TO_PLOT = 15000
PLOT = False
SAVE_PLOTS = True

# State-to-feature
DETECTION_RADIUS = 1.1
AMOUNT_ELEMENTS = 5

# Networks
MEMORY_SIZE = 5600
BATCH_SIZE = 1800
MIN_FRACTION = 0.2
TARGET_MODEL_UPDATE_RATE = 10
TRAINING_DATA_MODE = 2 # 0 = Use unmodified batch, 1 = use balanced batch, 2 = use subsample of bigger batch as batch, 3 = 1 + 2
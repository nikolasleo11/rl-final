# play --my-agent __agent --n-rounds 4000 --scenario coin-heaven --train 1 --no-gui
# Actions
ACTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']
INDICES_BY_ACTION = {
'RIGHT':0, 'LEFT':1, 'UP':2, 'DOWN':3, 'BOMB':4, 'WAIT':5
}
# File paths
SAVED_Q_VALUES_FILE_PATH = "q_values_solo_coin_heaven.pt"
SAVED_INDICES_BY_STATE_FILE_PATH = "indices_by_state_solo_coin_heaven.pt"
MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN = 10

# Exploration
EPSILON = 0.9

# Training
LEARNING_FACTOR = 0.1
GENERATE_STATISTICS = False

# State-to-feature
DETECTION_RADIUS = 1.1
AMOUNT_ELEMENTS = 5

# play --my-agent __agent --n-rounds 4000 --scenario coin-heaven --train 1 --no-gui
ACTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']
INDICES_BY_ACTION = {
'RIGHT':0, 'LEFT':1, 'UP':2, 'DOWN':3, 'BOMB':4, 'WAIT':5
}
SAVED_Q_VALUES_FILE_PATH = "q_values_solo_coin_heaven.pt"
SAVED_INDICES_BY_STATE_FILE_PATH = "indices_by_state_solo_coin_heaven.pt"
EPSILON = 0.0
DETECTION_RADIUS = 5.1
# Events
LEARNING_FACTOR = 0.1
MINIMUM_ROUNDS_REQUIRED_TO_SAVE_TRAIN = 1000
GENERATE_STATISTICS = False
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

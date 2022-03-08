# play --my-agent __agent --n-rounds 4000 --scenario coin-heaven --train 1 --no-gui
ACTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']
INDICES_BY_ACTION = {
'RIGHT':0, 'LEFT':1, 'UP':2, 'DOWN':3, 'BOMB':4, 'WAIT':5
}
SAVED_Q_VALUES_FILE_PATH = "q_values_solo_coin_heaven.pt"
SAVED_INDICES_BY_STATE_FILE_PATH = "indices_by_state_solo_coin_heaven.pt"
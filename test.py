import os
import unittest
from time import time

from agent_code.__agent.callbacks import mirror_features
from main import main


class MainTestCase(unittest.TestCase):
    def test_play(self):
        start_time = time()
        main(["play", "--n-rounds", "1", "--no-gui"])
        # Assert that log exists
        self.assertTrue(os.path.isfile("logs/game.log"))
        # Assert that game log way actually written
        self.assertGreater(os.path.getmtime("logs/game.log"), start_time)

    def test_mirrors(self):
        def are_equal(state1, state2):
            if state1['self'][0] != state2['self'][0] or state1['self'][1] != state2['self'][1] or state1['self'][2] != state2['self'][2] or state1['self'][3][0] != state2['self'][3][0] or state1['self'][3][1] != state2['self'][3][1]:
                return False
            if len(state1['others']) != len(state2['others']):
                return False
            for i in range(len(state1['others'])):
                other1 = state1['others'][i]
                other2 = state2['others'][i]

                if other1[0] != other2[0] or other1[1] != other2[1] or other1[
                    2] != other2[2] or other1[3][0] != other2[3][0] or other1[3][1] != \
                        other2[3][1]:
                    return False
            if len(state1['bombs']) != len(state2['bombs']):
                return False
            for i in range(len(state1['bombs'])):
                bombs1 = state1['bombs'][i]
                bombs2 = state2['bombs'][i]

                if bombs1[0][0] != bombs2[0][0] or bombs1[0][1] != \
                        bombs2[0][1]:
                    return False

            if len(state1['coins']) != len(state2['coins']):
                return False
            for i in range(len(state1['coins'])):
                coins1 = state1['coins'][i]
                coins2 = state2['coins'][i]

                if coins1[0] != coins2[0] or coins1[1] != \
                        coins2[1]:
                    return False
            return (state1['field'] == state2['field']).all()

        game_state = {}
        game_state['self'] = ('name', 0, False, (9, 1))
        game_state['others'] = [('name2', 0, False, (3, 5)), ('name2', 0, False, (1, 8))]
        game_state['bombs'] = [((4, 7), 2)]
        game_state['coins'] = [(9, 8), (6, 12), (15, 2)]
        game_state['field'] = [[1, 0, 0], [1, 1, 0], [0, 1, 0]]

        game_state2 = {}
        game_state2['self'] = ('name', 0, False, (7, 1))
        game_state2['others'] = [('name2', 0, False, (13, 5)), ('name2', 0, False, (15, 8))]
        game_state2['bombs'] = [((12, 7), 2)]
        game_state2['coins'] = [(7, 8), (10, 12), (1, 2)]
        game_state2['field'] = [[0, 0, 1], [0, 1, 1], [0, 1, 0]]

        game_state3 = {}
        game_state3['self'] = ('name', 0, False, (9, 15))
        game_state3['others'] = [('name2', 0, False, (3, 11)), ('name2', 0, False, (1, 8))]
        game_state3['bombs'] = [((4, 9), 2)]
        game_state3['coins'] = [(9, 8), (6, 4), (15, 14)]
        game_state3['field'] = [[0, 1, 0], [1, 1, 0], [1, 0, 0]]

        game_state4 = {}
        game_state4['self'] = ('name', 0, False, (7, 15))
        game_state4['others'] = [('name2', 0, False, (13, 11)), ('name2', 0, False, (15, 8))]
        game_state4['bombs'] = [((12, 9), 2)]
        game_state4['coins'] = [(7, 8), (10, 4), (1, 14)]
        game_state4['field'] = [[0, 1, 0], [0, 1, 1], [0, 0, 1]]

        mirrored_state1 = mirror_features(game_state)
        mirrored_state2 = mirror_features(game_state2)
        mirrored_state3 = mirror_features(game_state3)
        mirrored_state4 = mirror_features(game_state4)

        self.assertTrue(are_equal(mirrored_state1, mirrored_state2))
        self.assertTrue(are_equal(mirrored_state2, mirrored_state3))
        self.assertTrue(are_equal(mirrored_state3, mirrored_state4))
        self.assertTrue(are_equal(game_state2, mirrored_state1))



if __name__ == '__main__':
    unittest.main()

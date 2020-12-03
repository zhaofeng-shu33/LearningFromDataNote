import unittest
import numpy as np

import gym
import gym_maze

from qlearning import QTableLearning

class TestMaze(unittest.TestCase):
    def test_3x3_maze_q_table_learning(self):
        env = gym.make("maze-sample-3x3-v0", enable_render=False)
        alg = QTableLearning(env)
        alg.train()
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            current_state, reward, done, _ = env.step(action)
            print(current_state)
            if done:
                break
            done_cnt += 1
        print(done_cnt)

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np

from maze import MazeEnvSample3x3

from qlearning import QTableLearning

class TestMaze(unittest.TestCase):
    def test_3x3_maze(self):
        env = MazeEnvSample3x3()
        current_state, reward, done = env.step(1) # right
        self.assertEqual(current_state, [0, 1])
        current_state, reward, done = env.step(3) # down
        self.assertEqual(current_state, [1, 1])
        current_state, reward, done = env.step(3) # down
        self.assertEqual(current_state, [2, 1])
        current_state, reward, done = env.step(1) # right
        self.assertEqual(current_state, [2, 2])
        self.assertTrue(done)
        self.assertTrue(reward > 0)
        env.reset()
        self.assertEqual(env.state, [0, 0])

    def test_3x3_maze_q_table_learning(self):
        env = MazeEnvSample3x3()
        alg = QTableLearning(env)
        alg.train()
        done_cnt = 0
        current_state = env.reset()
        while True:
            action = alg.predict(current_state)
            current_state, reward, done = env.step(action)
            if done:
                break
            done_cnt += 1
        self.assertTrue(done_cnt < 10)

if __name__ == '__main__':
    unittest.main()
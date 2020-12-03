import unittest
import numpy as np

import gym
import gym_maze

from qlearning import QTableLearning

class TestMaze(unittest.TestCase):
    def test_3x3_maze_q_table_learning(self):
        env = gym.make("maze-sample-3x3-v0")
        alg = QTableLearning(env)
        alg.train()
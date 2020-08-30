import numpy as np
import datetime

np.random.seed(2020)

"define the environment"
class Environment:
    def __init__(self):
        self.tau = 0.02 # time delta between two consecutive steps
        self.delta_x = 4.8
        self.delta_theta = 12 * np.pi / 180
        self.delta_theta_prime = 45 * np.pi / 180
        self.F = 10
        self.m = 0.1
        self.M = 1
        self.L = 0.5
        self.g = 9.8
    def step(self, action, state):
        """Given an action, outputs the reward.
        Args:
            action: boolean, an action taken by the agent
            state: 1 x 4 array, the current state of the agent

        Outputs:
            next_state: 1 x 4 array, the next state of the agent
            reward: float, the reward at the next state
            done: bool, stop or continue learning
        """
        
        x = state[0]
        x_dot = state[2]
        theta = state[1]
        theta_dot = state[3]
        if action:
            F = self.F
        else:
            F = -1.0 * self.F
        # compute acceleration of x and theta
        _matrix_22 = [[self.m + self.M, self.m * self.L * np.cos(theta)],
                     [np.cos(theta), self.L]]
        _vector_2 = [F + self.m * self.L * theta_dot ** 2 * np.sin(theta),
                    self.g * np.sin(theta)]
        x_double_dot, theta_double_dot = np.linalg.solve(_matrix_22, _vector_22)
        # set new state
        x_prime = x + self.tau * x_dot
        theta_prime = theta + self.tau * theta_dot
        x_prime_dot = x_dot + self.tau * x_double_dot
        theta_prime_dot = theta_dot + self.tau * theta_double_dot
        next_state = [x_prime, theta_prime, x_prime_dot,
                      theta_prime_dot]
        # check reward
        if np.abs(theta_prime) < self.delta_theta:
            reward = 1
        else:
            reward = 0
        # check termination condition
        if np.abs(theta_prime) > self.delta_theta_prime or \
            np.abs(x_prime) > self.delta_x:
            done = True
        else:
            done = False

        return next_state, reward, done

    def reset(self, is_training=True):
        """Reset the agent state
        Args:
            is_training: bool, if True, the initial state of the agent will be set randomly,
                or the initial state will be (0,0) by default.

        Outputs:
            state: 1 x 4 array, the initial state.
        """
        if is_training:
            # randomly init
            init_x = 2 * self.delta_x * np.random.random() - self.delta_x
            init_theta = 2 * self.delta_theta_prime * np.random.random() - self.delta_theta_prime
            init_state = [init_x, init_theta, 0, 0]
        else:
            init_state = [0, 13 * np.pi / 180, 0, 0]

        return init_state

"define the agent"
class Agent:
    def __init__(self, *args, **kwargs):
        self.gamma = kwargs["gamma"]
        self.alpha = kwargs["alpha"]
        self.eps = kwargs["eps"]

        # Q function
        self.Q = None

    def do(self, state):
        """Know the current state, choose an action.
        Args:
            state: 1 x 4 array
        Outputs:
            action: an action the agenet decides to take, in [0, 1, 2, 3].
        """
        return ar_t[0][0]

    def learn(self, state, action, reward, next_state):
        """Learn from the environment.
        Args:
            state: str, the current state
            action: int, the action taken by itself
            reward: float, the reward after taking the action
            next_state: str, the next state

        Outputs:
            None
        """





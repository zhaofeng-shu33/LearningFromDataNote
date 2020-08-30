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



"define the agent"
class Agent:
    def __init__(self, *args, **kwargs):
        self.gamma = kwargs["gamma"]
        self.alpha = kwargs["alpha"]
        self.eps = kwargs["eps"]

        # self.Q = defaultdict(lambda: np.array([.0, .0, .0, .0]))
        self.Q = defaultdict(lambda: np.random.rand(4))

    def reset(self, is_training=True):
        """Reset the agent state
        Args:
            is_training: bool, if True, the initial state of the agent will be set randomly,
                or the initial state will be (0,0) by default.

        Outputs:
            state: str, the initial state.
        """
        
        self.reward = self.raw_reward.copy()

        if is_training:
            # randomly init
            while True:
                init_state = np.random.randint(0,4,(2))
                init_reward = self.reward[init_state[0],init_state[1]]
                # fixed condition adviced by Shi Mao
                if init_reward >= 0 and init_reward != 4:
                    init_state = init_state.tolist()
                    break
        else:
            init_state = [0,0]

        self.state = json.dumps(init_state)
        
        return self.state

    def do(self, state):
        """Know the current state, choose an action.
        Args:
            state: str, as "[0, 0]", "[0, 1]", etc.
        Outputs:
            action: an action the agenet decides to take, in [0, 1, 2, 3].
        """
        action_reward = self.Q[state]
        allowed_action_space = self.set_action_space(state)
        if np.random.rand(1) < self.eps:
            return np.random.choice(allowed_action_space)
        ar_t = [[i, action_reward[i]] for i in allowed_action_space]
        ar_t.sort(key=lambda x:x[1], reverse=True)
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
        allowed_action_space = self.set_action_space(state)
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state][allowed_action_space]) - self.Q[state][action])

    def set_action_space(self, state):
        state = json.loads(state)
        max_row, max_col = self.max_row, self.max_col
        row, col = state
        action_space = self.action_space.copy()

        if row == max_row:
            action_space.remove(3)
        if row == 0:
            action_space.remove(2)
        if col == max_col:
            action_space.remove(1)
        if col == 0:
            action_space.remove(0)
        return action_space


if __name__ == "__main__":

    "define the parameters"
    agent_params = {
        "gamma" : 0.8,        # discounted rate
        "alpha" : 0.1,        # learning rate
        "eps" : 0.9,          # initialize the e-greedy
        }

    max_iter = 10000

    # initialize the environment & the agent
    env = Environment()

    # the mouse cannot jump out of the grid
    agent_params["max_row"] = env.max_row - 1
    agent_params["max_col"] = env.max_col - 1
    smart_mouse = Agent(**agent_params)

    "start learning your policy"
    step = 0
    for step in range(max_iter):
        current_state = env.reset()
        while True:
            # act
            action = smart_mouse.do(current_state)

            # get reward from the environment
            next_state, reward, done = env.step(action, current_state)

            # learn from the reward
            smart_mouse.learn(current_state, action, reward, next_state)

            current_state = next_state

            if done:
                print("Step {} done, reward {}".format(step, reward))
                break 

        # epsilon decay
        if step % 100 == 0 and step > 0:
            eps = - 0.99 * step / max_iter + 1.0
            eps = np.maximum(0.1, eps)
            print("Eps:",eps)
            smart_mouse.eps = eps

    "evaluate your smart mouse"
    current_state = env.reset(False)
    trajectory_mat = env.reward.copy()
    trajectory_mat[0,0] = 5
    smart_mouse.eps = 0.0

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.ion()
    epoch = 0
    while True:
        action = smart_mouse.do(current_state)
        next_state, reward, done = env.step(action, current_state)
        current_state = next_state
        trajectory_mat[json.loads(current_state)[0], json.loads(current_state)[1]] = 5

        ax.matshow(trajectory_mat, cmap="coolwarm")
        plt.pause(0.5)

        print("===> [Eval] state:{}, reward:{} <===".format(current_state, reward))
        print("state value: \n",smart_mouse.Q[current_state])
        epoch += 1

        if done:
            plt.title(datetime.datetime.now().ctime())
            plt.savefig("1.png")
            print("***" * 10)
            if reward < 0 or epoch > 10:
                print("Your code does not work.")
            else:
                print("Your code works.")
            print("***" * 10)
            break
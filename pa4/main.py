import numpy as np
from cartpole import Environment, Agent

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

    smart_controller = Agent(**agent_params)

    "start learning your policy"
    for step in range(max_iter):
        current_state = env.reset()
        while True:
            # act
            action = smart_controller.do(current_state)

            # get reward from the environment
            next_state, reward, done = env.step(action, current_state)

            # learn from the reward
            smart_controller.learn(current_state, action, reward, next_state)

            current_state = next_state

            if done:
                print("Step {} done, reward {}".format(step, reward))
                break 

        # epsilon decay
        if step % 100 == 0 and step > 0:
            eps = - 0.99 * step / max_iter + 1.0
            eps = np.maximum(0.1, eps)
            print("Eps:", eps)
            smart_controller.eps = eps

    "evaluate your smart controller"
    # TBD
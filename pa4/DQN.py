from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''A deep neural network with input dimension = observation_space
output dimension = action_space
'''
class DQN:
    def __init__(self, observation_space, action_space):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.05))

    def predict(self, x):
        q_values = self.model.predict(x)
        return q_values

    def train(self, x, y):
        self.model.fit(x, y, verbose=0)

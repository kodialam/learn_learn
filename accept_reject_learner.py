import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import config

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))


class PointAccepter:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(config.INPUT_DIM*10, input_dim=config.INPUT_DIM, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def pred(self, x):
        return self.model.predict(np.array([x]), verbose=False)

    def fit(self, x, y):
        self.model.fit(np.array([x]), np.array([y]), verbose=False)


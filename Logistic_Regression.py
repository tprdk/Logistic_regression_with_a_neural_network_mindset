import numpy as np


def sigmoid(x_input):
    return 1 / (1 + np.exp(-x_input))



class Logistic_Regression:
    def __init__(self, epochs, learning_rate):
        self.epoch         = epochs
        self.learning_rate = learning_rate

        #initialize weights and bias
        self.weights = None
        self.bias    = 0.0


    def train_model(self, train_x, y_train):
        # get row col channel prop of images
        # channel is 3 if image is in rgb mode
        count    = len(train_x)
        row      = train_x[0].shape[0]
        col      = train_x[0].shape[1]
        channels = train_x[0].shape[2]

        #x_train shape (row * col * channel, 1)
        #y_train shape (1, count)
        #weight shape  (m, 1)
        x_train      = np.reshape(train_x, (row * col * channels, count))
        self.weights = np.zeros(shape=(row * col * channels, 1), dtype='float32')

        for epoch in range(self.epoch):
            prediction   = sigmoid(np.dot(self.weights.T, x_train) + self.bias)
            error        = prediction - y_train
            #calculate cost
            #cost formula =>
            cost = (-1 / count) * np.sum(y_train * np.log(prediction) + (1 - y_train) * np.log(1 - prediction))

            #print cost and wrong prediction count on every 1000. epoch
            if epoch % 1000 == 0:
                print(f'Epoch {epoch} - cost : {cost}')
                print(f'wrong prediction count : {np.sum(np.abs(y_train - [1 if i >= 0.5 else 0 for i in prediction[0]]))}')

            # calculating gradients
            d_weights = (1 / count) * np.dot(x_train, error.T)
            d_bias    = (1 / count) * np.sum(error)

            # updating weights and biases
            self.weights = self.weights - self.learning_rate * d_weights
            self.bias    = self.bias    - self.learning_rate * d_bias


    def test_model(self, test_x, y_test):
        count    = len(test_x)
        row      = test_x[0].shape[0]
        col      = test_x[0].shape[1]
        channels = test_x[0].shape[2]

        x_test   = np.reshape(test_x, (row * col * channels, count))

        prediction   = sigmoid(np.dot(self.weights.T, x_test) + self.bias)

        new_array   = [1.0 if predict >= 0.5 else 0.0 for predict in prediction[0]]

        print(new_array)


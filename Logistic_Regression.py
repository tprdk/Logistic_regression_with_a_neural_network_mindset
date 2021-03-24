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


    def train_model(self, x_train, y_train):
        #x_train shape (row * col * channel, 1)
        #y_train shape (1, count)
        #weight shape  (count, 1)
        count = x_train.shape[1]
        self.weights = np.zeros(shape=(x_train.shape[0], 1), dtype='float32')

        for epoch in range(self.epoch):
            prediction   = sigmoid(np.dot(self.weights.T, x_train) + self.bias)
            error        = prediction - y_train
            #calculate cost
            #cost formula => (-1 / count) * sum(y_train * log(prediction) + (1 -y_train) * log(1 - prediction))
            cost = (-1 / count) * np.sum(y_train * np.log(prediction) + (1 - y_train) * np.log(1 - prediction))

            #print cost and wrong prediction count on every 1000. epoch
            if epoch % 1000 == 0:
                print(f'Epoch {epoch} - cost : {cost}')

            # calculating gradients
            d_weights = (1 / count) * np.dot(x_train, error.T)
            d_bias    = (1 / count) * np.sum(error)

            # updating weights and biases
            self.weights = self.weights - self.learning_rate * d_weights
            self.bias    = self.bias    - self.learning_rate * d_bias


    def test_model(self, x_test, y_test):
        prediction              = sigmoid(np.dot(self.weights.T, x_test) + self.bias)
        test_count              = len(y_test)
        wrong_prediction_count  = np.sum(np.abs(y_test - [1 if i >= 0.5 else 0 for i in prediction[0]]))
        print(f'test samples count : {test_count} '
              f'wrong predictions count : {wrong_prediction_count} \n'
              f'accuracy : %{100 * (test_count - wrong_prediction_count )/test_count}')


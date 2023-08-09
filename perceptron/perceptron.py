import random
import numpy as np
from .activation import linear_function


class perceptron:
    def __init__(self):
        random.seed(1)
        """
        the weight matrix with random values between -1 and 1.
        The weight matrix has shape (3, 1) as it corresponds to 3 input
        features and 1 output.
        """
        self.weight = 2 * random.random() - 1
        self.bias = 0
        self.learning_rate = 0.001

    # Forward pass through the perceptron.
    def forward(self, x, activation=linear_function):
        """
        # Calculate the dot product of input "x" and the weight matrix
        "self.weight".
        # Then, add the bias "self.bias".
        # Finally, apply the linear activation function.
        """
        y_ = np.dot(x, self.weight) + self.bias
        return activation(y_)

    # Update rule for the perceptron during training.
    def update_rule(self, y, y_, X):
        """
        # Calculate the change in prediction (y - y_) and update the weight
        and bias
        # using the learning rate and the input "X".
        """
        change = y - y_
        self.weight = self.weight + self.learning_rate * (change) * X
        self.bias = self.bias + self.learning_rate * (change)

    # Training function for the perceptron.
    def train(
        self,
        train_inputs,
        train_outputs,
        epochs,
        activation=linear_function,
    ):
        for i in range(epochs):
            print(f"taining {i}")
            for X, y in zip(train_inputs, train_outputs):
                y_ = self.forward(X, activation)
                print(f"y = {y}, predict:{y_}")

                # Update the weights and bias using the update rule.
                self.update_rule(y, y_, X)

    # Prediction function for the perceptron.
    def predict(self, X):
        # Perform a forward pass to get the predicted output "y_".
        return self.forward(X)

import numpy as np
import random


# Activation function: Rectified Linear Unit (ReLU)
def relu_function(z):
    # Applies the ReLU activation function element-wise to the input array.
    return np.where(z > 0, z, 0)


# Activation function: Sigmoid
def sigmoid_function(z):
    # Applies the sigmoid activation function element-wise to the input array.
    return 1 / (1 + np.exp(-z))


# Activation function: Linear function (y = c + z)
def linear_function(z):
    c = random.random()
    # Returns the input array plus .
    return c + z


# Activation derivative


def sigmoid_derivative(x):
    # Returns the derivative of the sigmoid activation function.
    return x * (1 - x)

import numpy as np

from activation import sigmoid_derivative, sigmoid_function


class MLP:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation,
        derivative_activation,
    ) -> None:
        """
        This is a three layer neural network with:
        input layer:
        hidden layer:
        output layer:
        input_size: number of input neurons
        hidden_size: number of hidden neurons
        output_size: number of output neurons
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.derivative_activation = derivative_activation

        self.weights = np.random.randn(
            self.input_size, self.hidden_size
        )  # rowsXcolumns
        self.bias_hidden = np.zeros((1, self.hidden_size))  # rowsXcolumns
        self.bias_outputs = np.zeros((1, self.output_size))  # rowsXcolumns
        self.weights_outputs = np.random.randn(
            self.hidden_size,
            self.output_size,
        )
        self.learning_rate = 0.001

    def forward(self, X):
        self.hidden_activation = self.activation(
            np.dot(X, self.weights) + self.bias_hidden
        )
        self.output_activation = self.activation(
            np.dot(self.hidden_activation, self.weights_outputs) + self.bias_outputs
        )
        return self.output_activation

    def backpropergation(self, x, y, y_, lr):
        error = y - y_
        delta_output = error * self.derivative_activation(
            y_,
        )
        error_hidden = delta_output.dot(self.weights_outputs.T)
        delta_hidden = error_hidden * self.derivative_activation(self.hidden_activation)

        self.weights_outputs += self.hidden_activation.T.dot(delta_output) * lr
        self.bias_outputs += np.sum(delta_output) * lr
        self.weights += x.T.dot(delta_hidden) * lr
        self.bias_hidden += np.sum(delta_hidden) * lr

    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                y_ = self.forward(x)
                self.backpropergation(x, target, y_, lr)

    def predict(self, x):
        return self.forward(x)

    # Example usage


if __name__ == "__main__":
    X = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MLP(
        input_size=2,
        hidden_size=4,
        output_size=1,
        activation=sigmoid_function,
        derivative_activation=sigmoid_derivative,
    )
    mlp.train(X, y, epochs=10000, lr=0.1)

    for x in X:
        print(f"Input: {x}, Predicted: {mlp.predict(x)}")

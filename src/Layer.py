import numpy as np

class DenseLayer:
    def __init__(self, n_input, n_output):
        self.weights = 0.1 * np.random.randn(n_input, n_output)
        self.biases = np.zeros((1, n_output))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
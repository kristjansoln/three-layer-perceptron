"""
network.py

A module for implementing a three layer perceptron and gradient descent.

Author:
Kristjan Šoln
"""

import numpy as np


class Perceptron(object):

    def __init__(self, layer_sizes):
        self.number_of_layers = 4  # input layer + 2 neuron layers + output layer
        self.sizes = layer_sizes
        # Biases:
        self.biases = []
        for layer_size in layer_sizes[1:]:
            self.biases.append(np.random.randn(layer_size, 1))
        # Weights:
        self.weights = []
        self.weights = [np.random.randn(y, x)  # TODO: rewrite this your own way
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Network structure and actual neurons
        self.layers = [np.zeros(shape=x) for x in layer_sizes]
        """
        Example structure for layer_sizes = [5,4,3,2]:
            o  o    => output layer (with  biases and weights for each neuron to the previous layer)
           o  o  o    => first hidden layer (with  biases and weights for each neuron to the previous layer)
          o  o  o  o    => first hidden layer (with  biases and weights for each neuron to the previous layer)
         o  o  o  o  o    => input layer (no weights or biases assigned, input data is written directly to this layer)
        """

    def feedforward(self, input_array):
        # input layer value assignment
        if len(input_array) != len(self.layers[0]):
            raise Exception("Input data does not match input layer size")
        self.layers[0] = input_array  # Shallow copy. Correct to true copy if this causes problems.

        for i in range(1, len(self.layers)):
            print(i)
            for j in range(len(self.layers[i])):
                pl = self.layers[i - 1]  # NOTE: this can be moved up for efficiency
                w = self.weights[i - 1][j]
                b = self.biases[i - 1][j]
                # Neuron value calculation
                self.layers[i][j] = sigmoid(np.dot(pl, w) + b)

        return self.layers[-1]


def sigmoid(z):
    # Prevent overflow warnings
    if z > 100:
        z = 100
    elif z < -100:
        z = 100
    # Calculate the value
    return 1.0 / (1.0 + np.exp(-z))

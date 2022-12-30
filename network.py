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



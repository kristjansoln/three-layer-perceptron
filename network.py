"""
network.py

A module for implementing a three layer perceptron and gradient descent.

Author:
Kristjan Šoln
"""
from pprint import pprint

import numpy as np
import random


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

    def feedforward(self, input_array):  # TODO: TEST ME!
        """Calculate the output of the network based on a certain input."""
        # input layer value assignment
        if len(input_array) != len(self.layers[0]):
            raise Exception("Input data does not match input layer size")
        self.layers[0] = input_array  # Shallow copy. Correct to true copy if this causes problems.

        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                pl = self.layers[i - 1]  # NOTE: this can be moved up for efficiency
                w = self.weights[i - 1][j]
                b = self.biases[i - 1][j]
                # Neuron value calculation
                self.layers[i][j] = sigmoid(np.dot(pl, w) + b)

        return self.layers[-1]

    def train(self, X_input, y_input, epoch_num, batch_size):
        """Train the network"""
        if len(X_input) != len(y_input):
            raise Exception("Data and label array lenght do not match.")
        if epoch_num < 1:
            raise Exception("Invalid number of epochs")
        if batch_size < 1:
            raise Exception("Invalid number of batches")

        # Shuffle training data
        data_original = list(zip(X_input, y_input))
        data = random.sample(data_original, len(data_original))

        # Divide training data into batches
        batches = list(divide_into_batches(data, batch_size))

        # Main
        print("Training")
        for epoch_index in range(epoch_num):
            # Print out current epoch, calculate accuracy
            # NOTE: If get_accuracy won't get along well with large
            # datasets, make it do the evaluation on a smaller amount of data
            print("Epoch: %d/%d, accuracy: %.2f" % (epoch_index, epoch_num, self.get_accuracy(data)))
            for batch in batches:

                # Define empty nabla_w and nabla_b arrays of the correct dimensions (copied from the __init__ function)
                nabla_b = []
                for layer_size in self.sizes[1:]:
                    nabla_b.append(np.zeros(layer_size))
                nabla_w = [np.zeros([y, x])
                           for x, y in zip(self.sizes[:-1], self.sizes[1:])]

                # For each sample in a batch, calculate delta_nabla and add it to nabla
                for x, y_hat in batch:
                    delta_nabla_w, delta_nabla_b = self.backpropagation()
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                # Correct weights and biases according to nabla
                self.correct_weights(nabla_b, nabla_w)

    def get_accuracy(self, data):
        """Return the accuracy of the network for a set of data.
        This data should be
        x - individual input sample
        y_hat - expected result
        y - actual result
        """
        accuracy = 0
        for x, y_hat in data:  # For each sample in the data
            y = self.feedforward(x)
            a = np.argmax(y)
            accuracy += (np.argmax(y) == y_hat)
        return accuracy / len(data)

    def backpropagation(self):
        # Define empty nabla_w and nabla_b arrays of the correct dimensions (copied from the __init__ function)
        nb = []
        for layer_size in self.sizes[1:]:
            nb.append(np.zeros(layer_size))
        nw = [np.zeros([y, x])
              for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return nw, nb

    def correct_weights(self, nabla_b, nabla_w):
        # for layer_index in range(1,len(self.layers)):
        pass


def sigmoid(z):
    """The sigmoid function"""
    # Prevent overflow warnings
    if z > 100:
        z = 100
    elif z < -100:
        z = 100
    # Calculate the value
    return 1.0 / (1.0 + np.exp(-z))


def divide_into_batches(data, batch_size):
    """Yield successive batch_size-sized chunks from data"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

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
        self.layers_z = [np.zeros(shape=x) for x in layer_sizes]  # for storing neuron values before applying the sigmoid
        self.layers = [np.zeros(shape=x) for x in layer_sizes]
        """
        Example structure for layer_sizes = [5,4,3,2]:
            o  o    => output layer (with  biases and weights for each neuron to the previous layer)
           o  o  o    => first hidden layer (with  biases and weights for each neuron to the previous layer)
          o  o  o  o    => first hidden layer (with  biases and weights for each neuron to the previous layer)
         o  o  o  o  o    => input layer (no weights or biases assigned, input data is written directly to this layer)
        """
        self.input_layer_size = None
        self.output_layer_size = None


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
                self.layers_z[i][j] = np.dot(pl, w) + b
                self.layers[i][j] = sigmoid(self.layers_z[i][j])

        return self.layers[-1]

    def train(self, X_input, y_input, epoch_num, batch_size, beta):
        """Train the network"""
        if len(X_input) != len(y_input):
            raise Exception("Data and label array lenght do not match.")
        if epoch_num < 1:
            raise Exception("Invalid number of epochs")
        if batch_size < 1:
            raise Exception("Invalid number of batches")

        self.input_layer_size = len(X_input[0])
        self.output_layer_size = max(y_input) + 1

        # Shuffle training data
        data_original = list(zip(X_input, y_input))
        data = random.sample(data_original, len(data_original))

        # Divide training data into batches
        batches = list(divide_into_batches(data, batch_size))

        # Main training loop
        print("Training")
        for epoch_index in range(epoch_num):
            # Print out current epoch, calculate accuracy
            # NOTE: If get_accuracy won't get along well with large
            # datasets, make it do the evaluation on a smaller amount of data
            print("Epoch: %d/%d, accuracy: %.4f" % (epoch_index, epoch_num, self.get_accuracy(data)))
            for batch in batches:

                # Define empty nabla_w and nabla_b arrays of the correct dimensions (copied from the __init__ function)
                nabla_b = []
                for layer_size in self.sizes[1:]:
                    nabla_b.append(np.zeros(layer_size))
                nabla_w = [np.zeros([y, x])
                           for x, y in zip(self.sizes[:-1], self.sizes[1:])]

                # For each sample in a batch, calculate delta_nabla and add it to nabla
                for x, y_hat in batch:
                    delta_nabla_w, delta_nabla_b = self.backpropagation(x, y_hat)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                # Correct weights and biases according to nabla
                self.correct_weights(nabla_b, nabla_w, beta, len(batch))

    def get_accuracy(self, data):
        """Return the accuracy of the network for a set of data.
        This data should be zipped together - list(zip(X,y))
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

    def backpropagation(self, x, y_hat_index):
        """Perform backpropagation:
        1. Input x: Set the corresponding activation a1 for the input layer.
        2. Feedforward: For each l=2,3,…,L compute zl=wlal−1+bl and al=σ(zl).
        3. Output error δL: Compute the vector δL = ∇_a C ⊙ σ′(z_L). This delta is an additional variable in the computation.
        4. Backpropagate the error: For each l=L−1,L−2,…,2 compute δ_l=((w_(l+1))' δ_(l+1) ⊙ σ′(z_l).
        5. Output: The gradient of the cost function is given by ∂C∂wjkl=akl−1δjl and ∂C∂bjl=δjl.

        Parameters x and y_hat_index represent a single sample along with the expected output (the index of the triggered output neuron).
        """
        # Define empty nabla_w and nabla_b arrays of the correct dimensions (copied from the __init__ function)
        nb = []
        for layer_size in self.sizes[1:]:
            nb.append(np.zeros(layer_size))
        nw = [np.zeros([y, x])
              for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        # Perform the feedforward operation on input data.
        # This populates the self.layers and self.layers_z attributes
        y = self.feedforward(x)

        # Calculate the cost derivative
        y_hat = np.zeros(self.output_layer_size)
        y_hat[y_hat_index] = 1
        cost_derivative = y - y_hat

        # Calculate the delta for the output layer
        delta = [np.zeros(layer_size) for layer_size in self.sizes[1:]]
        delta[-1] = np.multiply(cost_derivative, sigmoid_prime(self.layers_z[-1]))
        # Calculate the nabla_n and nabla_w for the output layer
        nb[-1] = delta[-1]
        nw[-1] = np.outer(delta[-1], self.layers[-2])  # Outer product of two vectors - a matrix

        for layer_i in reversed(range(1, len(self.layers)-1)):
            # Calculate the delta for the other layers - backpropagate the error along the layers
            delta[layer_i-1] = np.multiply(
                np.dot(
                    np.transpose(self.weights[layer_i]),
                    delta[layer_i]),
                sigmoid_prime(self.layers_z[layer_i]))
            # Calculate the nabla_n and nabla_w for other layers
            nb[layer_i-1] = delta[layer_i-1]
            nw[layer_i-1] = np.outer(delta[layer_i-1], self.layers[layer_i-1])  # Outer product of two vectors

        return nw, nb

    def correct_weights(self, nabla_b, nabla_w, beta, batch_len):
        # self.weights = [w - (beta / batch_len) * nw
        #                 for w, nw in zip(self.weights, nabla_w)]
        # self.biases = [b - (beta / batch_len) * nb
        #                for b, nb in zip(self.biases, nabla_b)]
        for layer_i in range(len(self.weights)):
            for neuron_i in range(len(self.weights[layer_i])):
                # Apply bias correction
                b = self.biases[layer_i][neuron_i]
                nb = nabla_b[layer_i][neuron_i]
                self.biases[layer_i][neuron_i] = b - (beta / batch_len) * nb
                for weight_i in range(len(self.weights[layer_i][neuron_i])):
                    # Apply weight correction
                    w = self.weights[layer_i][neuron_i][weight_i]
                    nw = nabla_w[layer_i][neuron_i][weight_i]
                    self.weights[layer_i][neuron_i][weight_i] = w - (beta / batch_len) * nw

        return


def sigmoid(z):
    """The sigmoid function"""
    # Prevent overflow warnings
    # if z > 100:
    #     z = 100
    # elif z < -100:
    #     z = 100
    # Calculate the value
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """The derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))

def divide_into_batches(data, batch_size):
    """Yield successive batch_size-sized chunks from data"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

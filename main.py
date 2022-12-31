"""
A program to demonstrate a three layer perceptron and backpropagation.

Author:
Kristjan Šoln
"""
import network
import numpy as np


def task1():
    # The XOR problem

    # Train dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
    y = np.array([0, 1, 1, 0])  # Validation data

    # Create the model object
    perc_task1 = network.Perceptron([2, 4, 4, 2])

    # Train the model
    perc_task1.train(X, y, 3000, 4, 0.5)

    # Test the model again
    print("Testing the model")
    print("Accuracy:", perc_task1.get_accuracy(list(zip(X, y))))


if __name__ == '__main__':
    task1()

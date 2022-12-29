"""
A program to demonstrate a three layer perceptron and backpropagation.

Author:
Kristjan Šoln
"""
import numpy as np

if __name__ == '__main__':
    # The XOR problem

    # Test data
    X = np.array([[0, 0], [0, ], [1, 0], [1, 1]])  # Input data
    y_hat = np.array([0, 1, 1, 0])  # Validation data

    

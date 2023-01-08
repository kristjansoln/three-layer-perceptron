"""
"A program that demonstrates backpropagation and how it is used in a multilayer perceptron."

Kristjan Šoln
"""
import csv
import itertools
import time

import network
import numpy as np


def task1():
    """Task 1 - The XOR problem"""

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

    return


def task2():
    """TASK 2 - The Isolet dataset"""

    # Preparing the dataset
    with open("data/isolet1+2+3+4.data", "r") as f:
        content = f.read()
    lines = content.split("\n")[:-1]

    X = []
    y = []
    for line in lines:
        data = line.split(", ")
        y.append(data.pop().strip("."))
        X.append(data)

    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    y = y.astype(int) - 1

    # Create the perceptron
    input_layer_size = len(X[0])
    output_layer_size = max(y) + 1
    perc_task2 = network.Perceptron([input_layer_size, 30, 30, output_layer_size])

    # Train the model
    perc_task2.train(X, y, 300, 128, 0.5)

    # TESTING: Prepare the test dataset
    with open("data/isolet5.data", "r") as f:
        content = f.read()
    lines = content.split("\n")[:-1]

    X_test = []
    y_test = []
    for line in lines:
        data = line.split(", ")
        y_test.append(data.pop().strip("."))
        X_test.append(data)

    X_test = np.array(X_test)
    X_test = X_test.astype(float)
    y_test = np.array(y_test)
    y_test = y_test.astype(int) - 1

    # Run the model on the test dataset
    print("Testing the model")
    print("Accuracy:", perc_task2.get_accuracy(list(zip(X_test, y_test))))

    return


def parameter_sweep():
    # Define parameter range
    epochs = 300  # 500
    layers_range = [3, 4]
    width_range = [10, 30, 80]
    beta_range = [0.5, 1, 2]

    # Preparing the train dataset
    with open("data/isolet1+2+3+4.data", "r") as f:
        content = f.read()
    lines = content.split("\n")[:-1]

    X = []
    y = []
    for line in lines:
        data = line.split(", ")
        y.append(data.pop().strip("."))
        X.append(data)

    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)
    y = y.astype(int) - 1

    # Preparing the test dataset
    with open("data/isolet5.data", "r") as f:
        content = f.read()
    lines = content.split("\n")[:-1]

    X_test = []
    y_test = []
    for line in lines:
        data = line.split(", ")
        y_test.append(data.pop().strip("."))
        X_test.append(data)

    X_test = np.array(X_test)
    X_test = X_test.astype(float)
    y_test = np.array(y_test)
    y_test = y_test.astype(int) - 1

    input_layer_size = len(X[0])
    output_layer_size = max(y) + 1

    del lines

    # Prepare the CSV file
    results = ['layers', 'width', 'beta', 't[min]', 'train. accuracy', 'val. accuracy']
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results)

    # for layers in layers_range:
    #     for width in width_range:
    #         for beta in beta_range:
    it = -1
    it_len = len(list(itertools.product(layers_range, width_range, beta_range)))
    for layers, width, beta in itertools.product(layers_range, width_range, beta_range):
        # Create the perceptron
        if layers == 3:
            perc = network.Perceptron([input_layer_size, 30, 30, output_layer_size])
        else:  # layers == 4:
            perc = network.Perceptron([input_layer_size, 30, 30, 30, output_layer_size])

        # Train the model
        it += 1
        start_time = time.time()
        print("It" + str(it) + "/" + str(it_len), ": Training with parameters: layers =", layers, " width =", width, "beta =", beta)
        perc.train(X, y, epochs, 128, beta, X_test, y_test)

        # Run the model on the test dataset
        t = round((time.time() - start_time)/60, 1)
        results = [layers, width, beta, t]
        results.extend(perc.training_accuracy)
        results.extend(perc.validation_accuracy)
        print("Total time[min]:", t)

        # Write to csv
        with open('results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results)

        del perc
    return


if __name__ == '__main__':
    parameter_sweep()

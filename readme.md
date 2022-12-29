# THREE LAYER PERCEPTRON

## Task description
Write a program to simulate a three-layer perceptron. The learning procedure should be based on the gradient backpropagation learning procedure (RV, p. 400). The program should allow testing the network with any given number of neurons of the second intermediate hidden layer. The number of neurons of the first layer is overwritten by the dimension of the samples. The number of neurons of the output layer should be equal to the number of classes.
### Task 1
First, test the program and check the correctness of the calculations on a simple example of a two-class training set - the bpxor.txt file in the supplementary material (the so-called XOR problem, which is not linearly separable).
### Task 2
Then test the program on the Isolet training set (file isolet1+2+3+4.data.zip in the supplementary material) and the test part of the collection (file isolet5.data.zip in the supplementary material). The collection is described in more detail in the file isolet.txt in the supplementary material. The collection contains samples of individual pronunciations of 26 English letters. The number of classes is therefore equal to 26 (from A to Z). The number of features in the samples is 617. The last feature indicates the class of the sample (1st to 26th). Perform several experiments varying the number of neurons of the hidden layer, the learning factor and the persistence factor, respectively.


import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import gzip
import struct

sys.path.append('../../../../study/machine learning/deep learning/backpropagation') #appending address to bring backpropagration class
from numpy_backpropagation import backpropagate

from mnist import MNIST
mndata = MNIST('./python-mnist/bin/data/MNIST/raw')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
input = np.array([train_images])[0]
output = np.array([train_labels])[0]

#creating a mnist backpropagate class
class mnist_backpropagate(backpropagate):
    def __init__(self, data, layers):
        super().__init__(data, layers)
        output = np.zeros((self.output.shape[0], 9))
        for i in range(self.output.shape[0]):
            new_output_vector = np.zeros((9))
            new_output_vector[self.output[i][0]- 1] = 1
            output[i] = new_output_vector
        self.output = output
        self.weights['weight{}'.format(len(self.layers) - 2)] = np.zeros((self.layers[-2], 9))

data = [i for i in range(2)]
data[0] = input
data[1] = output.reshape(output.shape[0], 1)
layers = [1,2,4,2,1,4]
test = mnist_backpropagate(data, layers = layers)
test.train(20, 0.01)
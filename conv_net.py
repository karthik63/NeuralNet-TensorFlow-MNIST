import tensorflow as tf
import numpy as np
import inputs


class NeuralNet:

    def __init__(self, x, y, z):
        self.X_train_set, self.Y_train_set_n, self.X_valid_set, self.Y_valid_set_n, self.X_test_set, self.Y_test_set_n,\
            self.Y_train_set_v, self.Y_valid_set_v, self.Y_test_set_v = inputs.get_inputs('mnist.pkl.gz')

        self.n_layers = x
        self.n_perceptrons_per_layer = y
        self.batch_size = z
        self.weights = [] * self.n_layers

        self.weights[1] = {'weights':tf.random_normal(784, self.n_perceptrons_per_layer[1]),
                           'biases':tf.random_normal(self.n_perceptrons_per_layer[1])}
        self.weights[self.n_layers - 1] = {'weights':tf.random_normal(self.n_perceptrons_per_layer[self.n_layers - 2], 10),
                                           'biases':tf.random_normal(10)}

        for i in range(2, self.n_layers - 1):
            self.weights[i] = {'weights':tf.random_normal((self.n_perceptrons_per_layer[i - 1]), self.n_perceptrons_per_layer[i])
                               'biases':tf.random_normal(self.n_perceptrons_per_layer[i])}

        self.layer_activations = [] * self.n_layers

    def predict(self, X):

        self.layer_activations[0] = X

        for i in range(1, self.n_layers):
            self.layer_activations[i] = tf.add(tf.matmul(self.layer_activations[i - 1], self.weights[i]['weights']),
                                               self.weights[i]['biases'])

        return self.layer_activations[self.n_layers - 1]

    def train(self, X, Y):

        for i in ran
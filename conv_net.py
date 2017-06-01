import tensorflow as tf
import numpy as np
import inputs

tf.set_random_seed(1234)

class NeuralNet:

    def __init__(self, x, y, z, w):
        self.X_train_set, self.Y_train_set_n, self.X_valid_set, self.Y_valid_set_n, self.X_test_set, self.Y_test_set_n,\
            self.Y_train_set_v, self.Y_valid_set_v, self.Y_test_set_v = inputs.get_inputs('mnist.pkl.gz')

        self.n_training_examples = self.X_train_set.shape[0]

        self.max_epochs = w
        self.n_layers = x
        self.n_perceptrons_per_layer = y
        self.batch_size = z
        self.weights = [None] * self.n_layers

        self.weights[1] = {'weights': tf.Variable(tf.random_normal((784, self.n_perceptrons_per_layer[1]))),
                           'biases': tf.Variable(tf.random_normal((1, self.n_perceptrons_per_layer[1])))}

        self.weights[self.n_layers - 1] = {'weights': tf.Variable(tf.random_normal((self.n_perceptrons_per_layer[self.n_layers - 2],
                                                                        10))),
                                           'biases': tf.Variable(tf.random_normal((1, 10)))}

        for i in range(2, self.n_layers - 1):
            self.weights[i] = {'weights': tf.Variable(tf.random_normal((self.n_perceptrons_per_layer[i - 1]), self.n_perceptrons_per_layer[i])),
                               'biases': tf.Variable(tf.random_normal(self.n_perceptrons_per_layer[i]))}

        self.layer_activations = [None] * self.n_layers

    def predict(self, X):

        self.layer_activations[0] = X

        for i in range(1, self.n_layers):
            self.layer_activations[i] = tf.nn.relu(tf.add(tf.matmul(self.layer_activations[i - 1], self.weights[i]['weights']),
                                               self.weights[i]['biases']))

        return self.layer_activations[self.n_layers - 1]

    def find_loss_per_batch(self, X, Y):

        prediction = self.predict(X)
        batch_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=Y))

        return batch_loss

    def train(self):

        X = tf.placeholder(dtype='float32', shape=(None, 784), name='X_batch')
        Y = tf.placeholder(dtype='float32', shape=(None, 10), name='Y_batch')

        batch_loss = self.find_loss_per_batch(X, Y)

        optimiser = tf.train.AdamOptimizer(learning_rate=.001).minimize(batch_loss)

        n_updates_per_epoch = int(self.n_training_examples / self.batch_size)

        with tf.Session() as sess:

            for i in range(self.max_epochs):
                epoch_loss = 0
                for j in range(n_updates_per_epoch):

                    current_batch_X = self.X_train_set[j * self.batch_size : j * self.batch_size + self.batch_size]
                    current_batch_Y = self.Y_train_set_v[j * self.batch_size : j * self.batch_size + self.batch_size]

                    sess.run(tf.global_variables_initializer())
                    _, temp = sess.run([optimiser, batch_loss], feed_dict={X: current_batch_X, Y: current_batch_Y})
                    epoch_loss += temp
                print('epoch loss is ' + str(epoch_loss))

    def find_accuracy(self):
        with tf.Session() as sess:
            a = sess.eval(self.predict(self.X_test_set))
        b = self.Y_test_set_v

        c = a == b

        print((sum(c)/b.shape[0]) * 100)

nn1 = NeuralNet(3, [784, 10, 10], 100, 10)

nn1.train()
nn1.find_accuracy()
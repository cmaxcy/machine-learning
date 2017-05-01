import tensorflow as tf
import numpy as np
import random
import math


class PrePost:
    """
        Functions with the purpose of assisting in Machine Learning pre and post processing.
        Prefix indicates domain in which function operates: np - Numpy, tf - TensorFlow.
        Mappings simplified with Input -> Output description.

        Note:
            - X examples are often stored as an (m * n) np array where m is the number of training examples and
            n is the number of features
            - Y labels are often stored as an (m * k) np array where m is the number of examples training examples and
            k is the number of labels (k = 1 for regression and binary classification)
    """

    @staticmethod
    def tf_square_error(tf_logits, tf_labels):
        """
        Calculates element-wise squared difference of tensors passed

        t1, t2 -> (t1 - t2) ** 2

        :param tf_logits: (m * k) tensor of prediction values
        :param tf_labels: (m * k) tensor of true values
        :return: the element-wise square difference of each corresponding term
        """
        return tf.square(tf_logits - tf_labels)

    @staticmethod
    def tf_cross_entropy(tf_logits, tf_labels):
        """
        Calculates element-wise cross entropy on tensors passed

        t1, t2 -> cross entropy

        :param tf_logits: (m * k) tensor of prediction values
        :param tf_labels: (m * k) tensor of true values
        :return: the element-wise cross entropy of each corresponding term
        """
        return tf_labels * -tf.log(tf_logits + 1e-10) - (1 - tf_labels) * tf.log(1 - tf_logits + 1e-10)

    @staticmethod
    def np_sigmoid(np_x):
        """
        Applies the sigmoid function element-wise to the np array passed
        Note: works on scalars

        np array -> np array

        :param np_x: n-dimensional np array (or matrix)
        :return: result of sigmoid application
        """
        return 1 / (1 + np.exp(-1 * np_x))

    # TODO: test on tensors with alternative dimensions (tested 2D)
    @staticmethod
    def tf_extend_ones(tf_tensor, axis):
        """
        Add row or column of 1's to tensor passed

        tensor -> extended tensor

        :param tf_tensor: target (rank 2) tensor
        :param axis: axis in which extension should occur axis=0 adds row, axis=1 adds column
        :return: extended tensor
        """

        # store shape of tensor passed
        tf_dims = tf.shape(tf_tensor)

        if axis == 0:
            tf_row = tf.ones(shape=[1, tf_dims[1]], dtype=tf_tensor.dtype)
            return tf.concat([tf_row, tf_tensor], 0)
        elif axis == 1:
            tf_col = tf.ones(shape=[tf_dims[0], 1], dtype=tf_tensor.dtype)
            return tf.concat([tf_col, tf_tensor], 1)
        else:
            print("ERROR: unrecognized axis passed to tf_extend_ones")
            return None

    @staticmethod
    def np_extend_ones(np_matrix, axis):
        """
        Adds row or column of 1's to matrix passed

        matrix -> extended matrix

        :param np_matrix: target matrix
        :param axis: axis in which extension should occur axis=0 adds row, axis=1 adds column
        :return new_matrix: extended matrix
        """

        matrix_copy = np_matrix

        height, width = np_matrix.shape

        if axis == 0:
            new_matrix = np.ones(shape=(height + 1, width))
            new_matrix[1:, :] = matrix_copy
        elif axis == 1:
            new_matrix = np.ones(shape=(height, width + 1))
            new_matrix[:, 1:] = matrix_copy
        else:
            print("ERROR: unrecognized axis passed to np_extend_ones")
            return None

        return new_matrix

    @staticmethod
    def np_feat_normal(np_x_examples):
        """
        Normalizes features of examples passed
        (X - mean) / std

        Data -> normalized Data, normalization info

        :param np_x_examples: (m * n) matrix containing x examples
        :return normald_mat, feature_means, feature_stds:
                the normalized (m * n) matrix along with the feature mean and std
        """

        feature_means = np.mean(np_x_examples, axis=0)
        feature_stds = np.std(np_x_examples, axis=0) + 1e-10  # avoid 0 div

        normald_mat = (np_x_examples - feature_means) / feature_stds

        return normald_mat, feature_means, feature_stds

    @staticmethod
    def np_tt_split(np_x_examples, np_y_examples, train_portion=0.70):
        """
        Randomly splits data into testing and training based on portion passed

        Data -> train Data, test Data

        :param np_x_examples: m * n matrix containing all X examples
        :param np_y_examples: m * k matrix containing all Y labels
        :param train_portion: percentage of data to be trained
        :return np_x_train, np_x_test, np_y_train, np_y_test: data split by training portion
        """

        m, n = np_x_examples.shape
        _, k = np_y_examples.shape

        # random indices for data to occupy
        rand_range = list(range(m))
        random.shuffle(rand_range)

        train_examp_count = math.floor(train_portion * m)
        test_examp_count = m - train_examp_count

        np_x_train = np.zeros((train_examp_count, n))
        np_x_test = np.zeros((test_examp_count, n))
        np_y_train = np.zeros((train_examp_count, k))
        np_y_test = np.zeros((test_examp_count, k))

        for i in range(train_examp_count):
            np_x_train[i, :] = np_x_examples[rand_range[i], :]
            np_y_train[i, :] = np_y_examples[rand_range[i], :]

        for i in range(test_examp_count):
            np_x_test[i, :] = np_x_examples[rand_range[i + train_examp_count], :]
            np_y_test[i, :] = np_y_examples[rand_range[i + train_examp_count], :]

        return np_x_train, np_x_test, np_y_train, np_y_test

    @staticmethod
    def batch_split(np_x_examples, np_y_examples, batch_size):
        """
        Splits X and Y into list of tuple batches (batch_size * n matrix, batch_size * k matrix)
        Last batch will be of size m % batch_size

        Data -> [b1, b2, ... bn]
        where bi = (x, y)

        :param np_x_examples: m * n matrix containing X examples
        :param np_y_examples: m * k matrix containing Y labels
        :param batch_size: desired sizes of batches
        :return batch_list: list of tuples containing the batches of X's and Y's
        """

        m, n = np_x_examples.shape

        batch_list = []

        for i in range(int(m / batch_size)):
            this_x = np_x_examples[i * batch_size: (i + 1) * batch_size, :]
            this_y = np_y_examples[i * batch_size: (i + 1) * batch_size, :]

            batch_list.append((this_x, this_y))

        # remainder batch
        if (m / batch_size) % 1 != 0:
            last_x = np_x_examples[(int(m / batch_size)) * batch_size:, :]
            last_y = np_y_examples[(int(m / batch_size)) * batch_size:, :]
            batch_list.append((last_x, last_y))

        return batch_list


class TrainTest:
    """
        Functions with the purpose of training and testing statistical models on data.
        All data passed and returned should be an np array of rank 2, or a list of such arrays (no tf's).
        Each function uses its own session and all TensorFlow operations are internal.
    """

    # TODO: consider altering propagation to remove double transpose (change TRAIN method)
    @staticmethod
    def forward_prop(np_x_examples, list_np_theta, tf_activation_f=tf.sigmoid, layer_activation_guide=None):
        """
        Runs examples on Network passed. Outputs prediction for regression and predicted likelihoods for
        classification

        examples, THETA -> predictions

        :param np_x_examples: (m * n) array of example data
        :param np_theta_list: List of np layer arrays
        :param tf_activation_f: activation function to be used between layers
        :param layer_activation_guide: Outlines layers in which activation function should be applied
               (defaults to every layer)
        :return: (m * k) array of predictions
        """

        m, n = np_x_examples.shape

        tf_x = tf.placeholder('float', [None, n], name="tf_x")

        # determines on which layers to apply activation function (defaults to every layer)
        if layer_activation_guide is None:
            layer_activation_guide = [True] * (len(list_np_theta))

        # convert np thetas to tf
        list_tf_theta = []
        for this_np_theta in list_np_theta:
            list_tf_theta.append(tf.Variable(this_np_theta))

        # forward prop, using activation function when specified
        output = tf.transpose(tf_x)
        for i in range(len(list_tf_theta)):
            output = PrePost.tf_extend_ones(output, 0)
            if layer_activation_guide[i]:
                output = tf_activation_f(tf.matmul(list_tf_theta[i], output))
            else:
                output = tf.matmul(list_tf_theta[i], output)

        output = tf.transpose(output)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(output, feed_dict={tf_x: np_x_examples})

    # TODO: consider altering propagation to remove double transpose (change TRAIN method)
    @staticmethod
    def train_deep_NN(np_x_examples, np_y_examples, hl_architecture, num_epochs=10, batch_size=10, reg_param=0.0,
                      show_epoch_each=1, tf_activation_f=tf.sigmoid, optimizer=tf.train.AdamOptimizer(),
                      tf_error_f=PrePost.tf_cross_entropy, layer_activation_guide=None):
        """
        Trains a feed-forward neural network on the data passed
        Network can have a non-negative number of hidden layers of variables size (0 hidden layers is linear model)
        Optimizer, activation function, and cost function are all able to be specified

        examples, labels, NN specifications -> [Theta(0), Theta(1), ... Theta(h)]

        where h is the number of hidden layers
        Theta(0) maps input data to first hidden layer
        Theta(1) first hl to second hl.... and Theta(h) last hl to output

        :param np_x_examples: X training data (m * n) matrix
        :param np_y_examples: Y training data (m * k) matrix
        :param hl_architecture: List of hidden layer sizes
        :param num_epochs: number of times to cycle through batches
        :param batch_size: size of batches to train on
        :param reg_param: regularization parameter
        :param show_epoch_each: frequency to show training status and error
        :param tf_activation_f: activation function to be used from layer to layer
        :param optimizer: method of parameter optimization
        :param tf_error_f: un-regularized cost function
        :param layer_activation_guide: specifies when activation function should be applied (default every layer)
        :return: list of learned weight matrices
        """

        m, n = np_x_examples.shape
        _, k = np_y_examples.shape

        # data divided into batch tuples
        batch_tups = PrePost.batch_split(np_x_examples, np_y_examples, batch_size)

        # construct layer neuron counts from input/output size and hl size
        net_arch = hl_architecture
        net_arch.insert(0, n)
        net_arch.append(k)

        # each Theta is a 2-dimensional tensor mapping layer to layer
        # dimensions are (layer(i + 1) * layer(i))
        Thetas = []
        for i in range(len(net_arch) - 1):
            Thetas.append(tf.Variable(1e-10 * tf.random_normal([hl_architecture[i + 1], hl_architecture[i] + 1])))

        # tensors will hold examples and results
        tf_x = tf.placeholder('float', [None, n], name="tf_x")
        tf_y = tf.placeholder('float', [None, k], name="tf_y")

        # determines on which layers to apply activation function (defaults to every layer)
        if layer_activation_guide is None:
            layer_activation_guide = [True] * (len(net_arch) - 1)

        # forward prop, using activation function when specified
        output = tf.transpose(tf_x)
        for i in range(len(Thetas)):
            output = PrePost.tf_extend_ones(output, 0)
            if layer_activation_guide[i]:
                output = tf_activation_f(tf.matmul(Thetas[i], output))
            else:
                output = tf.matmul(Thetas[i], output)

        output = tf.transpose(output)

        # calculate un-regularized error using passed error function
        tf_unreg_error = (1 / m) * tf.reduce_sum(tf_error_f(tf_logits=output, tf_labels=tf_y))

        # calculate regularized error (skip for un-regularized model)
        tf_reg_error = 0
        if reg_param != 0:
            for this_theta in Thetas:
                tf_reg_error += tf.reduce_sum(tf.square(this_theta))
            tf_reg_error *= (reg_param / (2 * m))

        tf_error = tf_unreg_error + tf_reg_error
        tf_train_op = optimizer.minimize(tf_error)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                epoch_loss = 0

                for this_batch in batch_tups:
                    this_x, this_y = this_batch

                    _, c = sess.run([tf_train_op, tf_error], feed_dict={tf_x: this_x, tf_y: this_y})

                    epoch_loss += c

                if (epoch + 1) % show_epoch_each == 0:
                    print("Completed", str(epoch + 1), "out of", str(num_epochs), "epochs. Error:", epoch_loss)

            return sess.run(Thetas)

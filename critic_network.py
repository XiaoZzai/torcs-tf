import tensorflow as tf
import numpy as np
import math


LAYER1_SIZE = 200
LAYER2_SIZE = 200
LEARNING_RATE = 1e-3
TAU = 1e-3
L2 = 0.01

class CriticNetwork:
    def __init__(self, sess, state_dim, action_dim):
        self.time_step = 0
        self.sess = sess

        self.state_input, self.action_input, self.q_value_output, self.net = self.create_q_network()

        self.target_state_input, self.target_action_input, self.target_q_value_output, self.target_update = \
            self.create_target_q_network(self.net)

        self.create_training_method()

        # initialization 
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        with tf.variable_scope("img_critic_traing_method") as scope:
            self.y_input = tf.placeholder("float", [None, 1])
            weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
            self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self):

        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        with tf.variable_scope("img_critic") as scope:
            s = tf.placeholder(tf.float32, [None, 64, 64, 4])
            a = tf.placeholder(tf.float32, [None, 1])

            # Convolution
            W_conv1, b_conv1 = self._conv_variable([8, 8, 4, 32])   # stride=4
            W_conv2, b_conv2 = self._conv_variable([4, 4, 32, 32])  # stride=2
            h_conv1 = tf.nn.relu6(self._conv2d(s, W_conv1, 4) + b_conv1)
            h_conv2 = tf.nn.relu6(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)

            # Flatten
            flatten = int(h_conv2.shape[1] * h_conv2.shape[2] * h_conv2.shape[3])
            h_conv2_flat = tf.reshape(h_conv2, [-1, flatten])

            # Concat
            # print(h_conv3_flat)
            # print(a)
            h_conv2_concat = tf.concat([h_conv2_flat, a], axis=1)
            # print(h_conv3_concat)

            # Dense
            W_fc1, b_fc1 = self._fc_variable([flatten+1, layer1_size])
            W_fc2, b_fc2 = self._fc_variable([layer2_size, layer1_size])
            W_v, b_v = self._fc_variable([layer2_size, 1])

            h_fc1 = tf.nn.relu6(tf.matmul(h_conv2_concat, W_fc1) + b_fc1)
            h_fc2 = tf.nn.relu6(tf.matmul(h_fc1, W_fc2) + b_fc2)
            v = tf.nn.relu6(tf.matmul(h_fc2, W_v) + b_v)

        return s, a, v, [W_conv1, b_conv1, W_conv2, b_conv2,
                          W_fc1, b_fc1, W_fc2, b_fc2, W_v, b_v]

    def create_target_q_network(self, net):
        with tf.variable_scope("img_critic_target") as scope:

            ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
            target_update = ema.apply(net)
            target_net = [ema.average(x) for x in net]

            s = tf.placeholder(tf.float32, [None, 64, 64, 4])
            a = tf.placeholder(tf.float32, [None, 1])

            # Convolution
            h_conv1 = tf.nn.relu6(self._conv2d(s, target_net[0], 4) + target_net[1])
            h_conv2 = tf.nn.relu6(self._conv2d(h_conv1, target_net[2], 2) + target_net[3])

            # Flatten
            flatten = int(h_conv2.shape[1] * h_conv2.shape[2] * h_conv2.shape[3])
            h_conv2_flat = tf.reshape(h_conv2, [-1, flatten])

            # Concat
            h_conv2_concat = tf.concat([h_conv2_flat, a], axis=1)

            # Dense
            h_fc1 = tf.nn.relu6(tf.matmul(h_conv2_concat, target_net[4]) + target_net[5])
            h_fc2 = tf.nn.relu6(tf.matmul(h_fc1, target_net[6]) + target_net[7])
            v = tf.nn.relu6(tf.matmul(h_fc2, target_net[8]) + target_net[9])

        return s, a, v, target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        # print(action_batch.shape)
        # print(state_batch.shape)
        # print(self.state_input)
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={
            self.y_input : y_batch,
            self.state_input : state_batch,
            self.action_input : action_batch
        })
        return cost

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input : state_batch,
            self.action_input : action_batch
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input : state_batch,
            self.target_action_input : action_batch
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input : state_batch,
            self.action_input : action_batch
        })

    def _fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels  = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        # d = 3*1e-4
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

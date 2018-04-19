import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 200
LAYER2_SIZE = 200
LEARNING_RATE = 1e-4
TAU = 1e-3
# BATCH_SIZE = 16

class ActorNetwork:
    def __init__(self,sess,state_dim,action_dim):

        self.sess = sess
        self.state_dim  = state_dim
        self.action_dim = action_dim
        
        self.state_input, self.action_output, self.net = self.create_network()
        self.target_state_input, self.target_action_output, self.target_update, self.target_net = self.create_target_network(self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        with tf.variable_scope("img_actor_traing_method") as scope:
            self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
            self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
            '''        
            for i, grad in enumerate(self.parameters_gradients):
                if grad is not None:
                    self.parameters_gradients[i] = tf.clip_by_value(grad, -2.0,2.0)
	        '''
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(list(zip(self.parameters_gradients, self.net)))
    
    def create_network(self):

        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        with tf.variable_scope("img_actor") as scope:
            s = tf.placeholder(tf.float32, [None, 64, 64, 4])

            # Convolution
            W_conv1, b_conv1 = self._conv_variable([8, 8, 4, 32])   # stride=4
            W_conv2, b_conv2 = self._conv_variable([4, 4, 32, 32])  # stride=2
            h_conv1 = tf.nn.relu6(self._conv2d(s, W_conv1, 4) + b_conv1)
            h_conv2 = tf.nn.relu6(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)

            # Flatten
            flatten = int(h_conv2.shape[1] * h_conv2.shape[2] * h_conv2.shape[3])
            h_conv2_flat = tf.reshape(h_conv2, [-1, flatten])

            # Dense
            W_fc1, b_fc1 = self._fc_variable([flatten, layer1_size])
            W_fc2, b_fc2 = self._fc_variable([layer2_size, layer1_size])
            W_steer, b_steer = self._fc_variable([layer2_size, 1])

            h_fc1 = tf.nn.relu6(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
            h_fc2 = tf.nn.relu6(tf.matmul(h_fc1, W_fc2) + b_fc2)
            steer = tf.nn.relu6(tf.matmul(h_fc2, W_steer) + b_steer)
        return s, steer, [W_conv1, b_conv1, W_conv2, b_conv2,
                          W_fc1, b_fc1, W_fc2, b_fc2, W_steer, b_steer]

    def create_target_network(self, net):
        with tf.variable_scope("img_actor_target") as scope:
            ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
            target_update = ema.apply(net)
            target_net = [ema.average(x) for x in net]

            s = tf.placeholder(tf.float32, [None, 64, 64, 4])

            h_conv1 = tf.nn.relu6(self._conv2d(s, target_net[0], 4) + target_net[1])
            h_conv2 = tf.nn.relu6(self._conv2d(h_conv1, target_net[2], 2) + target_net[3])

            # Flatten
            flatten = int(h_conv2.shape[1] * h_conv2.shape[2] * h_conv2.shape[3])
            h_conv2_flat = tf.reshape(h_conv2, [-1, flatten])

            # Dense
            h_fc1 = tf.nn.relu6(tf.matmul(h_conv2_flat, target_net[4]) + target_net[5])
            h_fc2 = tf.nn.relu6(tf.matmul(h_fc1, target_net[6]) + target_net[7])
            steer = tf.nn.relu6(tf.matmul(h_fc2, target_net[8]) + target_net[9])
        return s, steer, target_update, target_net

    def update_target(self):
        self.sess.run(self. target_update)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input : q_gradient_batch,
            self.state_input : state_batch
            })

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input : state_batch
            })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input : [state]
            })[0]


    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input:state_batch
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


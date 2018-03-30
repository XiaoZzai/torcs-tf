
import tensorflow as tf 
import numpy as np
import math

# According to
# According to original paper
LAYER1_SIZE = 200
LAYER2_SIZE = 200
LEARNING_RATE = 1e-4
TAU = 0.001
L2 = 0.01

class CriticNetwork:
    def __init__(self, sess, action_dim, img_dim):
        self.time_step = 0
        self.sess = sess

        self.img_input, self.action_input, self.q_value_output, self.net = self.create_q_network(action_dim, img_dim)

        self.target_img_input, self.target_action_input, self.target_q_value_output, self.target_update = \
            self.create_target_q_network( action_dim, self.net, img_dim)

        self.create_training_method()

        # initialization
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):

        with tf.name_scope("critic_method"):
            self.y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
            self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, action_dim, img_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        # Image state input
        with tf.name_scope("critic_network"):
            img_input = tf.placeholder(dtype=tf.float32, shape=[None, img_dim[0], img_dim[1], img_dim[2]], name="img_input")
            init_num = 1/math.sqrt(int(img_input.shape[1] * img_input.shape[2] * img_input.shape[3]))
            img_w1 = tf.Variable(tf.random_uniform(([5, 5, 9, 32]), -init_num, init_num), name="img_conv1_w")
            img_b1 = tf.Variable(tf.random_uniform([32], -init_num, init_num), name="img_conv1_b")
            img_layer1 = tf.nn.relu(tf.nn.conv2d(img_input, img_w1, [1, 1, 1, 1], "VALID") + img_b1, name="img_conv1")

            init_num = 1 / math.sqrt(int(img_layer1.shape[1] * img_layer1.shape[2] * img_layer1.shape[3]))
            img_w2 = tf.Variable(tf.random_uniform(([5, 5, 32, 32]), -init_num, init_num), name="img_conv2_w")
            img_b2 = tf.Variable(tf.random_uniform([32], -init_num, init_num), name="img_conv2_b")
            img_layer2 = tf.nn.relu(tf.nn.conv2d(img_layer1, img_w2, [1, 1, 1, 1], "VALID") + img_b2, name="img_conv2")

            init_num = 1 / math.sqrt(int(img_layer2.shape[1] * img_layer2.shape[2] * img_layer2.shape[3]))
            img_w3 = tf.Variable(tf.random_uniform(([3, 3, 32, 32]), -init_num, init_num), name="img_conv3_w")
            img_b3 = tf.Variable(tf.random_uniform([32], 1e-4, 1e-4), name="img_conv3_b")
            img_layer3 = tf.nn.relu(tf.nn.conv2d(img_layer2, img_w3, [1, 1, 1, 1], "VALID") + img_b3, name="img_conv3")

            flatten = int(img_layer3.shape[1] * img_layer3.shape[2] * img_layer3.shape[3])
            img_layer4 = tf.reshape(img_layer3, [-1, flatten], name="img_flatten")
            init_num = 1 / math.sqrt(flatten)
            img_w4 = tf.Variable(tf.random_uniform([flatten, layer1_size], -init_num, init_num), name="img_fc1_w")
            img_b4 = tf.Variable(tf.random_uniform([layer1_size], -init_num, init_num), name="img_fc1_b")
            img_layer5 = tf.nn.relu(tf.matmul(img_layer4, img_w4) + img_b4, name="img_fc1")

            action_input = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name="action_input")

            layer1 = tf.concat([img_layer5, action_input], 1, name="concat")
            init_num = 1 / math.sqrt(layer1_size + action_dim)
            w = tf.Variable(tf.random_uniform([layer1_size + action_dim, layer2_size], -init_num, init_num), name="common_w")
            b = tf.Variable(tf.random_uniform([layer2_size], -init_num, init_num), name="common_b")
            layer2 = tf.nn.relu(tf.matmul(layer1, w) + b, name="fc1")

            q_w = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-4, 3e-4), name="q_value_w")
            q_b = tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name="q_value_b")
            q_value_output = tf.identity(tf.matmul(layer2, q_w) + q_b, name="q_value")

        return img_input, action_input, q_value_output, \
               [img_w1, img_b1, img_w2, img_b2, img_w3, img_b3, img_w4, img_b4, w, b, q_w, q_b]

    def create_target_q_network(self, action_dim, net, img_dim):

        with tf.name_scope("critic_target_network"):

            ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
            target_update = ema.apply(net) # real operation to update
            target_net = [ema.average(x) for x in net] # just get value of shadow_value

            img_input = tf.placeholder(dtype=tf.float32, shape=[None, img_dim[0], img_dim[1], img_dim[2]])
            img_layer1 = tf.nn.relu(tf.nn.conv2d(img_input, target_net[0], [1, 1, 1, 1], "VALID") + target_net[1], name="img_conv1")
            img_layer2 = tf.nn.relu(tf.nn.conv2d(img_layer1, target_net[2], [1, 1, 1, 1], "VALID") + target_net[3], name="img_conv2")
            img_layer3 = tf.nn.relu(tf.nn.conv2d(img_layer2, target_net[4], [1, 1, 1, 1], "VALID") + target_net[5], name="img_conv3")
            flatten = int(img_layer3.shape[1] * img_layer3.shape[2] * img_layer3.shape[3])
            img_layer4 = tf.reshape(img_layer3, [-1, flatten], name="img_flatten")
            img_layer5 = tf.nn.relu(tf.matmul(img_layer4, target_net[6]) + target_net[7], name="img_fc1")
            action_input = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name="action_input")

            layer1 = tf.concat([img_layer5, action_input], 1, name="concat")
            layer2 = tf.nn.relu(tf.matmul(layer1, target_net[8]) + target_net[9], name="fc1")

            q_value_output = tf.identity(tf.matmul(layer2, target_net[10]) + target_net[11], name="q_value")

        return img_input, action_input, q_value_output, target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, action_batch, img_batch):
        self.time_step += 1
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={
            self.y_input : y_batch,
            self.action_input : action_batch,
            self.img_input : img_batch
        })
        return cost

    def gradients(self, action_batch, img_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.action_input : action_batch,
            self.img_input : img_batch
        })[0]

    def target_q(self, action_batch, img_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_action_input : action_batch,
            self.target_img_input : img_batch
        })

    def q_value(self, action_batch, img_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.action_input : action_batch,
            self.img_input : img_batch
        })

    # f fan-in size
    def variable(self, shape, f, name):
        return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)), name=name)
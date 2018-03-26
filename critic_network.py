
import tensorflow as tf 
import numpy as np
import math


LAYER1_SIZE = 300
LAYER2_SIZE = 600
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.0001

class CriticNetwork:
    def __init__(self, sess, state_dim, action_dim, img_dim):
        self.time_step = 0
        self.sess = sess

        self.img_input, self.state_input, self.action_input, self.q_value_output, self.net = self.create_q_network(state_dim, action_dim, img_dim)

        self.target_img_input, self.target_state_input, self.target_action_input, self.target_q_value_output, self.target_update = \
            self.create_target_q_network(state_dim, action_dim, self.net, img_dim)

        self.create_training_method()

        # initialization 
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, state_dim, action_dim, img_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        # Image state input
        img_input = tf.placeholder(dtype=tf.float32, shape=[None, img_dim[0], img_dim[1], img_dim[2]])

        img_w1 = tf.Variable(tf.random_uniform(([5, 5, 3, 16]), -1e-4, 1e-4))
        img_b1 = tf.Variable(tf.random_uniform([16], 1e-4, 1e-4))
        # 60*60*16
        img_layer1 = tf.nn.relu(tf.nn.conv2d(img_input, img_w1, [1, 1, 1, 1], "VALID") + img_b1)
        # 30*30*16
        img_layer2 = tf.nn.max_pool(img_layer1, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

        img_w2 = tf.Variable(tf.random_uniform(([5, 5, 16, 32]), -1e-4, 1e-4))
        img_b2 = tf.Variable(tf.random_uniform([32], 1e-4, 1e-4))
        # 26*26*32
        img_layer3 = tf.nn.relu(tf.nn.conv2d(img_layer2, img_w2, [1, 1, 1, 1], "VALID") + img_b2)
        # 13*13*32
        img_layer4 = tf.nn.max_pool(img_layer3, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

        img_w3 = tf.Variable(tf.random_uniform(([4, 4, 32, 64]), -1e-4, 1e-4))
        img_b3 = tf.Variable(tf.random_uniform([64], 1e-4, 1e-4))
        # 10*10*64
        img_layer5 = tf.nn.relu(tf.nn.conv2d(img_layer4, img_w3, [1, 1, 1, 1], "VALID") + img_b3)
        # 5*5*64
        img_layer6 = tf.nn.max_pool(img_layer5, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

        img_layer7 = tf.reshape(img_layer6, [-1, 5*5*64])

        img_w4 = tf.Variable(tf.random_uniform([5*5*64, layer1_size], -1e-4, 1e-4))
        img_b4 = tf.Variable(tf.random_uniform([layer1_size], -1e-4, 1e-4))

        img_layer8 = tf.nn.relu(tf.matmul(img_layer7, img_w4) + img_b4)

        state_input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        action_input = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])

        state_w1 = self.variable([state_dim, layer1_size], state_dim)
        state_b1 = self.variable([layer1_size], state_dim)

        state_w2 = self.variable([layer1_size, layer2_size], layer1_size + action_dim)
        action_w2 = self.variable([action_dim, layer2_size], layer1_size + action_dim)
        b2 = self.variable([layer2_size], layer1_size + action_dim)

        w3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input, state_w1) + state_b1)
        layer2 = tf.add(layer1, img_layer8)
        layer3 = tf.nn.relu(tf.matmul(layer2, state_w2) + tf.matmul(action_input, action_w2) + b2)
        q_value_output = tf.identity(tf.matmul(layer3, w3) + b3)

        return img_input, state_input, action_input, q_value_output, \
               [state_w1, state_b1, state_w2, action_w2, b2, w3, b3, img_w1, img_b1, img_w2, img_b2, img_w3, img_b3, img_w4, img_b4]

    def create_target_q_network(self, state_dim, action_dim, net, img_dim):
        
        state_input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        action_input = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        img_input = tf.placeholder(dtype=tf.float32, shape=[None, img_dim[0], img_dim[1], img_dim[2]])
        img_layer1 = tf.nn.relu(tf.nn.conv2d(img_input, net[7], [1, 1, 1, 1], "VALID") + net[8])
        img_layer2 = tf.nn.max_pool(img_layer1, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        img_layer3 = tf.nn.relu(tf.nn.conv2d(img_layer2, net[9], [1, 1, 1, 1], "VALID") + net[10])
        img_layer4 = tf.nn.max_pool(img_layer3, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        img_layer5 = tf.nn.relu(tf.nn.conv2d(img_layer4, net[11], [1, 1, 1, 1], "VALID") + net[12])
        img_layer6 = tf.nn.max_pool(img_layer5, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        img_layer7 = tf.reshape(img_layer6, [-1, 5*5*64])
        img_layer8 = tf.nn.relu(tf.matmul(img_layer7, net[13]) + net[14])

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = layer1 + img_layer8
        layer3 = tf.nn.relu(tf.matmul(layer2, target_net[2]) + tf.matmul(action_input, target_net[3]) + target_net[4])
        q_value_output = tf.identity(tf.matmul(layer3, target_net[5]) + target_net[6])

        return img_input, state_input, action_input, q_value_output, target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch, img_batch):
        self.time_step += 1
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={
            self.y_input : y_batch,
            self.state_input : state_batch,
            self.action_input : action_batch,
            self.img_input : img_batch
        })
        return cost

    def gradients(self, state_batch, action_batch, img_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input : state_batch,
            self.action_input : action_batch,
            self.img_input : img_batch
        })[0]

    def target_q(self, state_batch, action_batch, img_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input : state_batch,
            self.target_action_input : action_batch,
            self.target_img_input : img_batch
        })

    def q_value(self, state_batch, action_batch, img_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input : state_batch,
            self.action_input : action_batch,
            self.img_input : img_batch
        })

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)))
    '''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
    '''

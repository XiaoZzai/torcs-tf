import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 300
LAYER2_SIZE = 600
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 32
class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,action_dim):

        self.sess = sess
        self.state_dim  = state_dim
        self.action_dim = action_dim
        
        self.state_input, self.action_output, self.net = self.create_network(state_dim, action_dim)
        self.target_state_input, self.target_action_output, self.target_update, self.target_net = self.create_target_network(state_dim, action_dim, self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        #self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        '''        
        for i, grad in enumerate(self.parameters_gradients):
            if grad is not None:
                self.parameters_gradients[i] = tf.clip_by_value(grad, -2.0,2.0)
	    '''
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(list(zip(self.parameters_gradients, self.net)))
    
    def create_network(self, state_dim, action_dim, img_dim):

        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        # Image state input
        img_input = tf.placeholder(dtype=tf.uint8, shape=[None, img_dim[0], img_dim[1], img_dim[2]])
        img_w1 = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4))
        img_w1 = tf.nn.conv2d(img_input, , strides=[1, 1, 1, 1], padding='SAME')

        # Scalar state input
        state_input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        state_w1 = self.variable([state_dim, layer1_size], state_dim)
        state_b1 = self.variable([layer1_size], state_dim)
        state_w2 = self.variable([layer1_size, layer2_size], layer1_size)
        state_b2 = self.variable([layer2_size], layer1_size)

        layer1 = tf.nn.relu(tf.matmul(state_input, state_w1) + state_b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, state_w2) + state_b2)

        steer_w = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4))
        steer_b = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4))
        steer = tf.tanh(tf.matmul(layer2, steer_w) + steer_b)

        # accel_w = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4))
        # accel_b = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4))
        # accel = tf.sigmoid(tf.matmul(layer2, accel_w) + accel_b)

        # brake_w = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4))
        # brake_b = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4))
        # brake = tf.sigmoid(tf.matmul(layer2, brake_w) + brake_b)
        
        # action_output = tf.concat([steer, accel, brake], 1)
        action_output = steer
        return state_input,action_output,[state_w1, state_b1, state_w2, state_b2, steer_w, steer_b]

    def create_target_network(self, state_dim, action_dim, net, use_bn=False):
        state_input = tf.placeholder("float", [None, state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])

        steer = tf.tanh(tf.matmul(layer2, target_net[4]) + target_net[5])
        # accel = tf.sigmoid(tf.matmul(layer2,target_net[6]) + target_net[7])
        # brake = tf.sigmoid(tf.matmul(layer2,target_net[8]) + target_net[9])
        # action_output = tf.concat([steer, accel, brake], 1)

        action_output = steer
        return state_input, action_output, target_update, target_net

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

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)))
    '''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
            
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

    '''



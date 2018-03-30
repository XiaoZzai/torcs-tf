import tensorflow as tf
import math

# Hyper Parameters
# According to original paper
LAYER1_SIZE = 200
LAYER2_SIZE = 200
LEARNING_RATE = 1e-3
TAU = 0.001
BATCH_SIZE = 32

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self, sess, action_dim, img_dim):

        self.sess = sess
        self.action_dim = action_dim
        
        self.img_input, self.action_output, self.net = self.create_network(action_dim, img_dim)
        self.target_img_input, self.target_action_output, \
            self.target_update, self.target_net = self.create_target_network(action_dim, self.net, img_dim)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        with tf.name_scope("actor_method"):
            self.q_gradient_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
            self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(list(zip(self.parameters_gradients, self.net)))
    
    def create_network(self, action_dim, img_dim):

        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        # tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)
        with tf.name_scope("actor_network"):
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

            init_num = 1 / math.sqrt(layer1_size)
            img_w5 = tf.Variable(tf.random_uniform([layer1_size, layer2_size], -init_num, init_num), name="img_fc2_w")
            img_b5 = tf.Variable(tf.random_uniform([layer2_size], -init_num, init_num), name="img_fc2_b")
            img_layer6 = tf.nn.relu(tf.matmul(img_layer5, img_w5) + img_b5, name="img_fc2")

            steer_w = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-4, 3e-4), name="action_steer_w")
            steer_b = tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name="action_steer_b")
            steer = tf.tanh(tf.matmul(img_layer6, steer_w) + steer_b, name="action_steer")

            action_output = steer

        return img_input, action_output, \
               [img_w1, img_b1, img_w2, img_b2, img_w3, img_b3, img_w4, img_b4, img_w5, img_b5, steer_w, steer_b]

    def create_target_network(self, action_dim, net, img_dim):

        with tf.name_scope("actor_target_network"):
            ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
            target_update = ema.apply(net)
            target_net = [ema.average(x) for x in net]

            img_input = tf.placeholder(dtype=tf.float32, shape=[None, img_dim[0], img_dim[1], img_dim[2]])
            img_layer1 = tf.nn.relu(tf.nn.conv2d(img_input, target_net[0], [1, 1, 1, 1], "VALID") + target_net[1], name="img_conv1")
            img_layer2 = tf.nn.relu(tf.nn.conv2d(img_layer1, target_net[2], [1, 1, 1, 1], "VALID") + target_net[3], name="img_conv2")
            img_layer3 = tf.nn.relu(tf.nn.conv2d(img_layer2, target_net[4], [1, 1, 1, 1], "VALID") + target_net[5], name="img_conv3")
            flatten = int(img_layer3.shape[1] * img_layer3.shape[2] * img_layer3.shape[3])
            img_layer4 = tf.reshape(img_layer3, [-1, flatten], name="img_flatten")
            img_layer5 = tf.nn.relu(tf.matmul(img_layer4, target_net[6]) + target_net[7], name="img_fc1")
            img_layer6 = tf.nn.relu(tf.matmul(img_layer5, target_net[8]) + target_net[9], name="img_fc2")
            steer = tf.tanh(tf.matmul(img_layer6, target_net[10]) + target_net[11], name="action_steer")

            action_output = steer

        return img_input, action_output, target_update, target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, img_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input : q_gradient_batch,
            self.img_input : img_batch
            })

    def actions(self, img_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.img_input : img_batch
            })

    def action(self, img):
        return self.sess.run(self.action_output, feed_dict={
            self.img_input : [img]
            })[0]


    def target_actions(self, img_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_img_input : img_batch
            })

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)))
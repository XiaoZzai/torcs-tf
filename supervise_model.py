import tensorflow as tf
import numpy as np

class Supervise:
    def __init__(self, sess, state_dim, img_dim, models_dir):
        self.sess = sess
        self.state_dim = state_dim
        self.img_dim = img_dim
        self.models_dir = models_dir

        self.img_input, self.state_input, self.steer = self.create_model(state_dim, img_dim)
        self.create_method()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def create_model(self, state_dim, img_dim):
        img_input = tf.placeholder(dtype=tf.float32, shape=[None, img_dim[0], img_dim[1], img_dim[2]])
        img_layer1 = tf.layers.conv2d(inputs=img_input, filters=16, kernel_size=[5, 5], padding="VALID", activation=tf.nn.relu)
        img_layer2 = tf.layers.max_pooling2d(inputs=img_layer1, pool_size=[3, 3], strides=3)
        img_layer3 = tf.layers.conv2d(inputs=img_layer2, filters=32, kernel_size=[5, 5], padding="VALID", activation=tf.nn.relu)
        img_layer4 = tf.layers.max_pooling2d(inputs=img_layer3, pool_size=[3, 3], strides=3)
        img_layer5 = tf.layers.conv2d(inputs=img_layer4, filters=64, kernel_size=[3, 3], padding="VALID", activation=tf.nn.relu)
        img_layer6 = tf.layers.max_pooling2d(inputs=img_layer5, pool_size=[3, 3], strides=3)
        # img_layer7 = tf.reshape(img_layer6, [-1, 14*10*64])
        # print(img_layer6.shape)
        img_layer7 = tf.reshape(img_layer6, [-1, img_layer6.shape[1] * img_layer6.shape[2] * img_layer6.shape[3]])

        state_input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        layer1 = tf.concat([img_layer7, state_input], 1)
        layer2 = tf.layers.dense(layer1, 300, activation=tf.nn.relu)
        layer3 = tf.layers.dense(layer2, 600, activation=tf.nn.relu)
        # layer4 = tf.layers.dense(layer3, 128, activation=tf.nn.relu)
        steer = tf.layers.dense(layer3, 1, activation=tf.nn.tanh)

        return img_input, state_input, steer

    def create_method(self):
        self.steer_ground_truth = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.steer - self.steer_ground_truth)))
        self.cost = tf.reduce_sum(tf.abs(self.steer_ground_truth - self.steer))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    def validate(self, state_batch, steer_ground_truth_batch, img_batch):
        predicts = self.sess.run(self.steer, feed_dict={
            self.state_input : state_batch,
            self.img_input : img_batch
        })
        return np.sum(np.square(predicts - steer_ground_truth_batch))


    def train(self, state_batch, steer_ground_truth_batch, img_batch):
        _, cost = self.sess.run([self.optimizer, self.cost], feed_dict={
            self.state_input : state_batch,
            self.steer_ground_truth : steer_ground_truth_batch,
            self.img_input : img_batch
        })
        return cost

    def action(self, state, img):
        return self.sess.run(self.steer, feed_dict={
            self.state_input : [state],
            self.img_input : [img]
        })[0]

    def save_network(self, step):
        self.saver.save(self.sess, self.models_dir + 'torcs-network-supervise.ckpt', global_step=step)

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.models_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

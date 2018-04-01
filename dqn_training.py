#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
from gym_torcs import TorcsEnv
import random
import numpy as np
from collections import deque

GAME = 'torcs' # the name of the game being played for log files
STEERS = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
ACTIONS = len(STEERS) # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 50000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.3 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def index_to_action(index):
    return [STEERS[index], 0.10, 0]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    # input layer
    s = tf.placeholder("float", [None, 64, 64, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    flatten = int(h_conv3.shape[1] * h_conv3.shape[2] * h_conv3.shape[3])
    W_fc1 = weight_variable([flatten, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, flatten])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # store the previous observations in replay memory
    D = deque()

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("dqn_saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    i = 0
    MAX_STEPS = 2000000
    MAX_STEPS_EP = 3000
    env = TorcsEnv(vision=True, throttle=True, text_mode=False, track_no=15, random_track=False, track_range=(5, 8))
    while t < MAX_STEPS:
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        x_t = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        i_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        t_ep = 0
        while (t < MAX_STEPS) and (t_ep < MAX_STEPS_EP):

            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s : [i_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 3
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                else:
                    action_index = np.argmax(readout_t)

            a_t[action_index] = 1

            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observe next state and reward
            ob, r_t, terminal, info = env.step(index_to_action(action_index))
            x_t1 = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = x_t1.reshape(64, 64, 1)
            i_t1 = np.append(x_t1, i_t[:, :, :3], axis=2)
            # s_t1 = None
            # store the transition in D
            D.append((i_t, a_t, r_t, i_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch
                })

            # update the old values
            i_t = i_t1
            t += 1
            t_ep += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'dqn_saved_networks/' + GAME + '-dqn', global_step = t)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state,
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
                "/ Q_MAX %e" % np.max(readout_t))

            if terminal == True:
                break

def main():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

if __name__ == "__main__":
    main()

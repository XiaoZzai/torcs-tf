#!/usr/bin/python

import gym
import tensorflow as tf
import numpy as np
from utils import ornstein_uhlenbeck_process
import math
from critic_network import CriticNetwork 
from actor_network import ActorNetwork
from ReplayBuffer import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 30000
REPLAY_START_SIZE = 100
BATCH_SIZE = 16
GAMMA = 0.99


class ddpg:
    def __init__(self, env_name, sess, action_dim, models_dir, img_dim):

        self.name = 'DDPG'
        self.env_name = env_name
        self.action_dim = action_dim
        self.img_dim = img_dim
        self.models_dir = models_dir
        
        # Ensure action bound is symmetric
        self.time_step = 0 
        self.sess = sess

        self.actor_network = ActorNetwork(self.sess, self.action_dim, self.img_dim)
        self.critic_network = CriticNetwork(self.sess, self.action_dim, self.img_dim)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        self.saver = tf.train.Saver()


    def train(self):
        minibatch = self.replay_buffer.getBatch(BATCH_SIZE)
        img_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_img_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        # Calculate y_batch
        next_action_batch = self.actor_network.target_actions(next_img_batch)
        q_value_batch = self.critic_network.target_q(next_action_batch, next_img_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])

        critic_cost = self.critic_network.train(y_batch, action_batch, img_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(img_batch)

        q_gradient_batch = self.critic_network.gradients(action_batch_for_gradients, img_batch)

        self.actor_network.train(q_gradient_batch, img_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()
        return critic_cost

    def save_network(self, step):
        self.saver.save(self.sess, self.models_dir + self.env_name + '-network-ddpg.ckpt', global_step=step)

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.models_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def action(self, img):
        action = self.actor_network.action(img)

        action[0] = np.clip( action[0], -1 , 1 )
        # action[1] = np.clip( action[1], 0 , 1 )
        # action[2] = np.clip( action[2], 0 , 1 )

        return action

    def noise_action(self, epsilon, img):

        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(img)
        noise_t = np.zeros(self.action_dim)

        noise_t[0] = epsilon * ornstein_uhlenbeck_process(action[0],  0.0 , 0.60, 0.80)
        # noise_t[1] = epsilon * ornstein_uhlenbeck_process(action[1],  0.5 , 1.00, 0.10)
        # noise_t[2] = epsilon * ornstein_uhlenbeck_process(action[2], -0.1 , 1.00, 0.05)
        action = action + noise_t
        action[0] = np.clip( action[0], -1 , 1)
        # action[1] = np.clip( action[1], 0 , 1)
        # action[2] = np.clip( action[2], 0 , 1)
        return action
    
    def perceive(self, img, action, reward, next_img, done):
        if not (math.isnan(reward)):
            self.replay_buffer.add(img, action, reward, next_img, done)

        self.time_step =  self.time_step + 1

        # Return critic cost
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            return self.train()
        else:
            return 0



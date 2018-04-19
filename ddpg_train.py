import numpy as np
import os
import time
import tensorflow as tf
import traceback
import signal
np.random.seed(2018)

from utils import formatted_timestamp
from torcs_wrapper import TorcsWrapper
from ddpg import ddpg
from guide_ddpg import guide_ddpg
from my_config import *
import cv2

def main():
    MAX_EPISODE = max_episode
    MAX_STEPS_EP = max_steps_ep
    epsilon   = epsilon_start

    # Creating necessary directories
    experiment_name = "guide0"
    experiment_dir  = "experiment-%s/" % experiment_name
    models_dir = experiment_dir + "model/"
    logs_train_dir = experiment_dir + "logs-train/"
    if os.path.exists(experiment_dir) == False:
        os.mkdir(experiment_dir)
    if os.path.exists(logs_train_dir) == False:
        os.mkdir(logs_train_dir)
    if os.path.exists(models_dir) == False:
        os.mkdir(models_dir)

    description = 'Using the 4 frames as input, output (steer)' + '\n\n' + \
                    'Training with guide_ddpg and warmup in tracks [-2, 0, 1]' + '\n\n' \
                    'throttle = 0.20' + '\n\n' \
                    'brake = 0' + '\n\n' \
                    'sp*np.cos(obs["angle"]) - np.abs(sp*np.sin(obs["angle"])) - sp * np.abs(obs["trackPos"])  \
                    - sp * np.abs(action_torcs["steer"]) * 2 ' + '\n\n' + \
                    'env = TorcsWraper(noisy=True)' + '\n\n' \
                    'abs(trackPos) > 0.9 is out of track' + '\n\n'

    with open(experiment_dir + "README.md", 'w') as file:
        file.write(description)
        file.write("\n\n")
        file.write(formatted_timestamp())

    action_dim = 1
    state_dim  = 25
    env_name   = 'torcs'

    sess = tf.InteractiveSession()
    guide_agent = guide_ddpg(env_name, sess, 25,  1, "model/")

    agent = ddpg("img" + env_name, sess, state_dim, action_dim, models_dir)
    agent.load_network()
    # agent = guide_agent

    guide_agent.load_network()

    vision = False
    env = TorcsWrapper(noisy=True)

    # rewards_every_steps = np.zeros([MAX_STEPS])
    # actions_every_steps = np.zeros([MAX_STEPS, action_dim])

    # sess.run(tf.initialize_all_variables())

    # Using tensorboard to visualize data
    with tf.name_scope('summary'):
        critic_cost = tf.placeholder(dtype=tf.float32)
        actor_action = tf.placeholder(dtype=tf.float32)
        reward = tf.placeholder(dtype=tf.float32)
        # state = tf.placeholder(dtype=tf.float32, shape=(state_dim, ))
        tf.summary.scalar("critic_cost", critic_cost)
        tf.summary.scalar('actor_action', actor_action)
        tf.summary.scalar('reward', reward)
        # tf.summary.histogram('state', state)
        merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    print("Training Start.")
    # print('Press Ctrl+C to stop')
    # signal.signal(signal.SIGINT, signal_handler)

    reward_every_episode = np.zeros(MAX_EPISODE)
    start_time = time.time()
    delta_epsilon = (epsilon_start - epsilon_end) / (MAX_EPISODE / 2)
    try:
        for episode in range(MAX_EPISODE):
            track_no = np.random.choice([0, 1])
            s_t = env.reset(track_offset=track_no)

            total_reward = 0
            warmup_step = np.random.randint(10 * int(episode / 100), 400 + 10 * int(episode / 100))
            warmup_step = warmup_step % min(1000, MAX_STEPS_EP / 2)
            epsilon -= delta_epsilon
            epsilon = max(epsilon, epsilon_end)
            print((" Episode : " + str(episode) + " Replay Buffer " + str(agent.replay_buffer.count()) +
                   " WarmUp Step " + str(warmup_step) + " Track " + str(track_no) +
                   " Epsilon " + str(epsilon)))
            ep_r = 0
            for step in range(MAX_STEPS_EP):
                if step < warmup_step:
                    a_t = guide_agent.action(s_t[1])
                else:
                    a_t = agent.noise_action(s_t, epsilon)

                a_t = a_t[0]
                s_t1, r_t, done, info = env.step(a_t)
                cost = agent.perceive(s_t[0], a_t, r_t, s_t1[0], done)

                if step >= warmup_step:
                    ep_r += r_t

                # summary = sess.run([merged_summary], feed_dict={
                #     critic_cost : cost,
                #     actor_action : a_t,
                #     reward : r_t,
                #     state : s_t
                # })

                # writer.add_summary(summary[0], step)

                total_reward += r_t
                s_t = s_t1

                print("Ep", episode, "Total steps ", step, "Reward ", r_t, " Actions ", a_t, " Epsilon ", epsilon, "Step ep ", step)

                # rewards_every_steps[step] = r_t
                # actions_every_steps[step] = a_t

                if done:
                    break

            reward_every_episode[episode] = ep_r

            if np.mod(episode + 1, 200) == 0:
                print("Now we save model with step = ", episode)
                agent.save_network(episode + 1)

            print(("TOTAL REWARD @ " + str(episode) + "-th Episode  : Reward " + str(total_reward)))
            print("")
            # print('Now saving data. Please wait')
            # agent.save_network( + 1)

    except:
        traceback.print_exc()
        # with open((logs_train_dir + "exception"), 'w') as file:
        #     file.write(str(traceback.format_exc()))

    finally:
        env.end()
        end_time = time.time()

        print("Total time = %s " % (end_time - start_time))

        # np.save(logs_train_dir + "reward.npy", rewards_every_steps)
        # np.save(logs_train_dir + "action.npy", actions_every_steps)

        # with open(logs_train_dir + "log", 'w') as file:
        #     file.write("epsilon_start = %d\n" % epsilon_start)
        #     file.write("total_episode = %d\n" % MAX_EPISODE)
        #     file.write("total_step = %d\n" % step)
        #     file.write("total_time = %s (s)\n" % str(end_time - start_time))
        #
        print("Finish.")

if __name__ == "__main__":
    main()


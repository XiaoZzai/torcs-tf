import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import traceback
import cv2
import guide_ddpg
np.random.seed(2018)

from utils import formatted_timestamp
from gym_torcs import TorcsEnv
from ddpg import ddpg
from my_config import *

print( total_explore )
print( max_steps )
print( epsilon_start )
print( total_observe )

def main():

    OBSERVE = total_observe
    EXPLORE   = total_explore
    MAX_STEPS = max_steps
    MAX_STEPS_EP = max_steps_ep
    INIT_EPSILON = epsilon_start
    FINAL_EPSILON = epsilon_end

    epsilon   = epsilon_start

    # Creating necessary directories
    experiment_name = "img-4"
    experiment_dir  = "experiment-%s/" % experiment_name
    models_dir = experiment_dir + "model/"
    logs_train_dir = experiment_dir + "logs-train/"
    if os.path.exists(experiment_dir) == False:
        os.mkdir(experiment_dir)
    if os.path.exists(logs_train_dir) == False:
        os.mkdir(logs_train_dir)
    if os.path.exists(models_dir) == False:
        os.mkdir(models_dir)

    description = 'Only using raw pixels(4 frames) as input, output (steer)' + '\n' + \
                    'Training from scratch with actor mimic' + '\n\n' \
                    'Img dim = [64, 64, 4]' + '\n\n' \
                    'throttle = 0.14' + '\n\n' \
                    'brake = 0' + '\n\n' \
                    'sp*np.cos(obs["angle"]) - np.abs(sp*np.sin(obs["angle"])) - sp * np.abs(obs["trackPos"])  \
                    - sp * np.abs(action_torcs["steer"]) * 4' + '\n\n' + \
                    'env = TorcsEnv(vision=False, throttle=True, text_mode=False, track_no=15, random_track=False, track_range=(5, 8))' + '\n\n' \
                    'abs(trackPos) > 1 is out of track' + '\n\n' \

    with open(experiment_dir + "README.md", 'w') as file:
        file.write(description)
        file.write("\n\n")
        file.write(formatted_timestamp())

    action_dim = 1
    img_dim = [64, 64, 4]
    env_name   = 'torcs'

    guide_agent = guide_ddpg.ddpg('torcs', tf.InteractiveSession(), 30, 1, "experiment-tensorboard-4/model/")
    guide_agent.load_network()

    sess = tf.InteractiveSession()
    agent = ddpg(env_name, sess, action_dim, models_dir, img_dim)
    agent.load_network()

    vision = True
    env = TorcsEnv(vision=vision, throttle=True, text_mode=False, track_no=15, random_track=False, track_range=(5, 8))

    rewards_every_steps = np.zeros([MAX_STEPS])
    actions_every_steps = np.zeros([MAX_STEPS, action_dim])

    # sess.run(tf.initialize_all_variables())

    # Using tensorboard to visualize data
    with tf.name_scope('summary'):
        critic_cost = tf.placeholder(dtype=tf.float32)
        actor_action = tf.placeholder(dtype=tf.float32)
        reward = tf.placeholder(dtype=tf.float32)
        # img = tf.placeholder(dtype=tf.float32, shape=(1, img_dim[0], img_dim[1], img_dim[2]))
        tf.summary.scalar("critic_cost", critic_cost)
        tf.summary.scalar('actor_action', actor_action)
        tf.summary.scalar('reward', reward)
        # tf.summary.image("img", img)
        merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    print("Training Start.")
    start_time = time.time()
    i = 0
    step = 0
    try:
        while step < MAX_STEPS:


            if np.mod(i, 3) == 0:
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            # Early episode annealing for out of track driving and small progress
            # During early training phases - out of track and slow driving is allowed as humans do ( Margin of error )
            # As one learns to drive the constraints become stricter

            # random_number = random.random()
            # eps_early = max(epsilon,0.10)
            # if (random_number < (1.0-eps_early)) and (train_indicator == 1):
            #     early_stop = 1
            # else:
            #     early_stop = 0
            # print(("Episode : " + str(i) + " Replay Buffer " + str(agent.replay_buffer.count())))

            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm,
                              0.0))

            x_t = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY)
            ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
            x_t = x_t / 255
            i_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

            total_reward = 0
            step_ep = 0
            while (step < MAX_STEPS) and (step_ep < MAX_STEPS_EP):

                if (step >= OBSERVE) and (step < EXPLORE + OBSERVE):
                    epsilon -= (INIT_EPSILON - FINAL_EPSILON) / EXPLORE

                if (step < OBSERVE * 2 / 3) or (np.random.random() < epsilon):
                    a_t = guide_agent.action(s_t)
                else:
                    a_t = agent.noise_action(epsilon, i_t)

                ob, r_t, done, info = env.step([a_t[0], 0.14, 0])

                s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm,
                                  a_t[0]))

                x_t1 = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY)
                _, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
                x_t1 = x_t1.reshape(img_dim[0], img_dim[1], 1)
                x_t1 = x_t1 / 255

                # cv2.imshow("img", x_t1)
                # cv2.waitKey(0)

                i_t1 = np.append(x_t1, i_t[:, :, :3], axis=2)
                cost = agent.perceive(i_t, a_t, r_t, i_t1, done)
                summary = sess.run(merged_summary, feed_dict={
                    critic_cost : cost,
                    actor_action : a_t[0],
                    reward : r_t,
                    # img : i_t.reshape(1, img_dim[0], img_dim[1], img_dim[2])
                })

                writer.add_summary(summary, step)

                total_reward += r_t
                s_t = s_t1
                i_t = i_t1

                print("Ep", i, "Total steps", step, "Reward", r_t, " Actions ", a_t, " Epsilon ", epsilon, "Step ep", step_ep)

                rewards_every_steps[step] = r_t
                actions_every_steps[step] = a_t
                step += 1
                step_ep += 1


                if np.mod(step + 1, 10000) == 0:
                        print("Now we save model with step = ", step)
                        agent.save_network(step + 1)

                if done:
                    break

            print(("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward)))
            print(("Total Step: " + str(step)))
            print("")
            i += 1

    except:
        traceback.print_exc()
        with open((logs_train_dir + "exception"), 'w') as file:
            file.write(str(traceback.format_exc()))

    finally:
        env.end()
        end_time = time.time()

        np.save(logs_train_dir + "reward.npy", rewards_every_steps)
        np.save(logs_train_dir + "action.npy", actions_every_steps)

        with open(logs_train_dir + "log", 'w') as file:
            file.write("epsilon_start = %d\n" % epsilon_start)
            file.write("total_explore = %d\n" % total_explore)
            file.write("total_episode = %d\n" % i)
            file.write("total_step = %d\n" % step)
            file.write("total_time = %s (s)\n" % str(end_time - start_time))

        print("Finish.")

if __name__ == "__main__":
    main()


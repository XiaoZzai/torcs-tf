import numpy as np
import os
import time
import tensorflow as tf
import traceback
np.random.seed(2018)

from utils import formatted_timestamp
from gym_torcs import TorcsEnv
from ddpg import ddpg
from my_config import *
import cv2

print( total_explore )
print( max_steps )
print( epsilon_start )

def main():

    EXPLORE   = total_explore
    MAX_STEPS = max_steps
    MAX_STEPS_EP = max_steps_ep
    epsilon   = epsilon_start

    # Creating necessary directories
    experiment_name = "image-fc-0"
    experiment_dir  = "experiment-%s/" % experiment_name
    models_dir = experiment_dir + "model/"
    logs_train_dir = experiment_dir + "logs-train/"
    if os.path.exists(experiment_dir) == False:
        os.mkdir(experiment_dir)
    if os.path.exists(logs_train_dir) == False:
        os.mkdir(logs_train_dir)
    if os.path.exists(models_dir) == False:
        os.mkdir(models_dir)

    description = 'Using flattened image and speeds as input, output (steer)' + '\n\n' + \
                    'Training from scratch' + '\n\n' \
                    'throttle = 0.14' + '\n\n' \
                    'brake = 0' + '\n\n' \
                    'sp*np.cos(obs["angle"]) - np.abs(sp*np.sin(obs["angle"])) - sp * np.abs(obs["trackPos"])  \
                    - sp * np.abs(action_torcs["steer"]) * 2' + '\n\n' + \
                    'env = TorcsEnv(vision=False, noisy=True)' + '\n\n' \
                    'abs(trackPos) > 0.9 is out of track' + '\n\n'

    with open(experiment_dir + "README.md", 'w') as file:
        file.write(description)
        file.write("\n\n")
        file.write(formatted_timestamp())

    action_dim = 1
    state_dim  = 4099
    env_name   = 'torcs'

    sess = tf.InteractiveSession()
    agent = ddpg(env_name, sess, state_dim, action_dim, models_dir)
    agent.load_network()

    vision = True
    env = TorcsEnv(vision=vision, noisy=True)

    rewards_every_steps = np.zeros([MAX_STEPS])
    actions_every_steps = np.zeros([MAX_STEPS, action_dim])

    # Using tensorboard to visualize data
    with tf.name_scope('summary'):
        critic_cost = tf.placeholder(dtype=tf.float32)
        actor_action = tf.placeholder(dtype=tf.float32)
        reward = tf.placeholder(dtype=tf.float32)
        state = tf.placeholder(dtype=tf.float32, shape=(state_dim, ))
        tf.summary.scalar("critic_cost", critic_cost)
        tf.summary.scalar('actor_action', actor_action)
        tf.summary.scalar('reward', reward)
        tf.summary.histogram('state', state)
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

            print(("Episode : " + str(i) + " Replay Buffer " + str(agent.replay_buffer.count())))

            # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, 0.0))
            x_t = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 255.0
            s_t = np.hstack((x_t.reshape(4096), ob.speedX, ob.speedY, ob.speedZ))
            total_reward = 0
            step_ep = 0
            while (step < MAX_STEPS) and (step_ep < MAX_STEPS_EP):
                # print(x_t)
                cv2.imshow("img", x_t)
                cv2.waitKey(10)

                # Take noisy actions during training
                epsilon -= 1.0 / EXPLORE
                epsilon = np.clip(epsilon, 0.05, 0.30)
                a_t = agent.noise_action(s_t, epsilon)
                ob, r_t, done, info = env.step([a_t[0], 0.14, 0])

                x_t1 = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 255.0
                s_t1 = np.hstack((x_t1.reshape(4096), ob.speedX, ob.speedY, ob.speedZ))

                # s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, a_t[0]))

                cost = agent.perceive(s_t, a_t, r_t, s_t1, done)
                summary = sess.run([merged_summary], feed_dict={
                    critic_cost : cost,
                    actor_action : a_t[0],
                    reward : r_t,
                    state : s_t
                })

                writer.add_summary(summary[0], step)

                total_reward += r_t
                s_t = s_t1

                print("Ep", i, "Total steps", step, "Reward", r_t, " Actions ", a_t, " Epsilon ", epsilon, "Step ep", step_ep)

                rewards_every_steps[step] = r_t
                actions_every_steps[step] = a_t
                step += 1
                step_ep += 1

                if done:
                    break

                if np.mod(step + 1, 10000) == 0:
                        print("Saving model at step = ", step)
                        agent.save_network(step + 1)

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


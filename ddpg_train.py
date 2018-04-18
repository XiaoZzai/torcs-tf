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
from my_config import *

print( total_explore )
print( max_steps )
print( epsilon_start )

stop_requested = False
def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True

def main():

    EXPLORE   = total_explore
    MAX_STEPS = max_steps
    MAX_STEPS_EP = max_steps_ep
    epsilon   = epsilon_start

    # Creating necessary directories
    experiment_name = "img-0"
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
                    'Training from scratch' + '\n\n' \
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
    state_dim  = 22
    env_name   = 'torcs'

    sess = tf.InteractiveSession()
    agent = ddpg(env_name, sess, state_dim, action_dim, models_dir)
    agent.load_network()

    vision = False
    env = TorcsWrapper(noisy=True)

    rewards_every_steps = np.zeros([MAX_STEPS])
    actions_every_steps = np.zeros([MAX_STEPS, action_dim])

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
    print('Press Ctrl+C to stop')
    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()
    i = 0
    step = 0
    try:
        while (step < MAX_STEPS) and (stop_requested == False):
            # if ((np.mod(i, 10) == 0 ) and (i>20)):
            #     train_indicator= 0
            # else:
            #     train_indicator=is_training

            track_no = np.random.choice([0, 1])
            s_t = env.reset(track_offset=track_no)

            # Early episode annealing for out of track driving and small progress
            # During early training phases - out of track and slow driving is allowed as humans do ( Margin of error )
            # As one learns to drive the constraints become stricter

            # random_number = random.random()
            # eps_early = max(epsilon,0.10)
            # if (random_number < (1.0-eps_early)) and (train_indicator == 1):
            #     early_stop = 1
            # else:
            #     early_stop = 0
            print(("Episode : " + str(i) + " Replay Buffer " + str(agent.replay_buffer.count())))

            # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm,
            #                  0.0))
            # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, 0.0))

            # s_t = np.hstack((ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, 0.0))
            # s_t = np.hstack((x_t, x_t, x_t, x_t))

            total_reward = 0
            step_ep = 0
            while (step < MAX_STEPS) and (step_ep < MAX_STEPS_EP) and (stop_requested == False):
                # Take noisy actions during training
                # epsilon -= 1.0 / EXPLORE
                if step < EXPLORE / 3:
                    epsilon = 0.4
                elif step < EXPLORE * 2 / 3:
                    epsilon = 0.12
                else:
                    epsilon = 0.05
                a_t = agent.noise_action(s_t, epsilon)
                a_t = a_t[0]
                s_t1, r_t, done, info = env.step(a_t)
                cost = agent.perceive(s_t, a_t, r_t, s_t1, done)
                summary = sess.run([merged_summary], feed_dict={
                    critic_cost : cost,
                    actor_action : a_t,
                    reward : r_t,
                    # state : s_t
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

                if np.mod(step + 1, 100000) == 0:
                        print("Now we save model with step = ", step)
                        agent.save_network(step + 1)

            print(("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward)))
            print(("Total Step: " + str(step)))
            print("")
            i += 1

        # signal.pause()
        print('Now saving data. Please wait')
        agent.save_network(step + 1)

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


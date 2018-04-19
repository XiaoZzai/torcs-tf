import numpy as np
import os
import time
import tensorflow as tf
import traceback
np.random.seed(2018)

from torcs_wrapper import TorcsWrapper
from guide_ddpg import guide_ddpg

def main():

    MAX_EP = 1
    MAX_STEPS_EP = 2000

    # Creating necessary directories
    test_track_no = 6
    experiment_name = "noisy-2"
    # experiment_dir  = "experiment-%s/" % experiment_name
    # models_dir = experiment_dir + "model/"
    # logs_test_dir = experiment_dir + "logs-test-track-no-%d/" % test_track_no

    if os.path.exists("model/") == False:
        print("%s dosen't exists" % "model/")
        return

    # if os.path.exists(models_dir) == False:
    #     print("%s dosen't exists" % models_dir)
    #     return
    #
    # if os.path.exists(logs_test_dir) == False:
    #     os.mkdir(logs_test_dir)

    action_dim = 1
    state_dim  = 25
    env_name   = 'torcs'

    sess = tf.InteractiveSession()
    agent = guide_ddpg(env_name, sess, state_dim, action_dim, "model")
    agent.load_network()

    vision = False
    env = TorcsWrapper(noisy=False)

    # rewards_every_steps = np.zeros([MAX_EP, MAX_STEPS_EP])
    # actions_every_steps = np.zeros([MAX_EP, MAX_STEPS_EP, action_dim])

    # Using tensorboard to visualize data
    with tf.name_scope('summary'):
        actor_action = tf.placeholder(dtype=tf.float32)
        reward = tf.placeholder(dtype=tf.float32)
        # state = tf.placeholder(dtype=tf.float32, shape=(state_dim, ))
        tf.summary.scalar('actor_action', actor_action)
        tf.summary.scalar('reward', reward)
        # tf.summary.histogram('state', state)
        merged_summary = tf.summary.merge_all()

    # writer = tf.summary.FileWriter(logs_test_dir, sess.graph)

    print("Testing Start.")
    start_time = time.time()
    step = 0
    try:
        for i in range(MAX_EP):
            s_t = env.reset(0)

            print(("Episode : " + str(i) + " Replay Buffer " + str(agent.replay_buffer.count())))

            total_reward = 0
            step_ep = 0
            while (step_ep < MAX_STEPS_EP):
                a_t = agent.action(s_t[1])
                s_t1, r_t, done, info = env.step(a_t[0])
                summary = sess.run([merged_summary], feed_dict={
                    actor_action : a_t[0],
                    reward : r_t,
                    # state : s_t
                })

                # writer.add_summary(summary[0], step)

                total_reward += r_t
                s_t = s_t1

                print("Ep", i, "Total steps", step, "Reward", r_t, " Actions ", a_t, "Step ep", step_ep)

                # rewards_every_steps[step] = r_t
                # actions_every_steps[step] = a_t
                step += 1
                step_ep += 1

                if done:
                    break

            print(("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward)))
            print(("Total Step: " + str(step)))
            print("")

    except:
        traceback.print_exc()
        # with open((logs_test_dir + "exception"), 'w') as file:
        #     file.write(str(traceback.format_exc()))

    finally:
        env.end()
        end_time = time.time()

        # np.save(logs_test_dir + "reward.npy", rewards_every_steps)
        # np.save(logs_test_dir + "action.npy", actions_every_steps)

        # with open(logs_test_dir + "log", 'w') as file:
        #     file.write("total_episode = %d\n" % MAX_EP)
        #     file.write("max_steps_ep = %d\n" % MAX_STEPS_EP)
        #     file.write("total_step = %d\n" % step)
        #     file.write("total_time = %s (s)\n" % str(end_time - start_time))
        #
        print("Finish.")

if __name__ == "__main__":
    main()
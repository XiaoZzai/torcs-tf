import numpy as np
import os
import time
import traceback
np.random.seed(2018)

from utils import formatted_timestamp
from gym_torcs import TorcsEnv
from ddpg import ddpg
from my_config import *

print( is_training )
print( total_explore )
print( max_steps )
print( epsilon_start )

def main(train_indicator=is_training):  # 1 means Train, 0 means simply Run

    experiment_name = "reward-tf3"
    experiment_dir  = "experiment-%s/" % experiment_name

    if os.path.exists(experiment_dir) == False:
        os.mkdir(experiment_dir)

    description = """reward = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos']) / 8 """ \
                    """- sp * np.abs(action_torcs['steer']) * 4"""

    with open(experiment_dir + "README.md", 'w') as file:
        file.write(description)
        file.write("\n\n")
        file.write(formatted_timestamp())

    action_dim = 1
    state_dim  = 30
    env_name   = 'torcs'

    agent = ddpg(env_name, state_dim, action_dim, experiment_dir)
    agent.load_network()

    vision = False
    env = TorcsEnv(vision=vision, throttle=True, text_mode=False, track_no=0, random_track=True, track_range=(0, 4))
    
    EXPLORE   = total_explore
    MAX_STEPS = max_steps
    MAX_STEPS_EP = max_steps_ep
    epsilon   = epsilon_start

    step = 0
    best_reward = -100000

    rewards_every_steps = np.zeros([MAX_STEPS])
    actions_every_steps = np.zeros([MAX_STEPS, action_dim])

    print("Training Start.")

    start_time = time.time()
    try:
        i = 0

        while step < MAX_STEPS:
            # if ((np.mod(i, 10) == 0 ) and (i>20)):
            #     train_indicator= 0
            # else:
            #     train_indicator=is_training

            # restart because of memory leak bug in torcs
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
            print(("Episode : " + str(i) + " Replay Buffer " + str(agent.replay_buffer.count())))

            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm,
                             0.0))

            # Counting the total reward and total steps in the current episode
            total_reward = 0.
            step_ep = 0

            # episode starts
            while (step < MAX_STEPS) or (step_ep < MAX_STEPS_EP):
                # Take noisy actions during training
                if (train_indicator):
                    epsilon -= 1.0 / EXPLORE
                    epsilon = max(epsilon, 0.0)
                    a_t = agent.noise_action(s_t, epsilon)
                else:
                    a_t = agent.action(s_t)

                #ob, r_t, done, info = env.step(a_t[0], early_stop)

                ob, r_t, done, info = env.step([a_t[0], 0.2, 0])
                s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm,
                                  a_t[0]))

                if (train_indicator):
                    agent.perceive(s_t, a_t, r_t, s_t1, done)

                # Cheking for nan rewards
                # if ( math.isnan( r_t )):
                #     r_t = 0.0
                #     for bad_r in range( 50 ):
                #         print( 'Bad Reward Found' )

                total_reward += r_t
                s_t = s_t1

                print("Ep", i, "Steps", step_ep, "Reward", r_t, " Actions ", a_t, " Epsilon ", epsilon)

                rewards_every_steps[step] = r_t
                actions_every_steps[step] = a_t
                step += 1
                step_ep += 1

                if done:
                    break

                if np.mod(step + 1, 50000) == 0:
                    if train_indicator == 1:
                        print("Now we save model with step = ", step)
                        agent.save_network(step + 1)

            # episode ends
            # if total_reward >= best_reward :
            #     if train_indicator == 1:
            #         print(("Now we save model with reward " + str(total_reward) + " previous best reward was " + str(best_reward)))
            #         best_reward = total_reward
            #         agent.save_network("best-reward")

            print(("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward)))
            print(("Total Step: " + str(step)))
            print("")
            i += 1

    except:
        traceback.print_exc()
        with open((experiment_dir + "exception"), 'w') as file:
            file.write(str(traceback.format_exc()))

    finally:
        env.end()
        end_time = time.time()

        np.save(experiment_dir + "reward-train.npy", rewards_every_steps)
        np.save(experiment_dir + "action-train.npy", actions_every_steps)

        with open(experiment_dir + "log-train", 'w') as file:
            file.write("epsilon start = %d" % epsilon_start)
            file.write("total explore = %d" % total_explore)
            file.write("total step = %d\n" % step)
            file.write("total time = %s (s)\n" % str(end_time - start_time))

        print("Finish.")

if __name__ == "__main__":
    main()


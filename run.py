import numpy as np
import os
import time
import traceback
np.random.seed(2018)

from gym_torcs import TorcsEnv
from ddpg import ddpg

def main():

    experiment_name = "reward-tf2"
    experiment_dir  = "experiment-%s/" % experiment_name

    action_dim = 1
    state_dim  = 30
    env_name   = 'torcs'

    vision = False
    env = TorcsEnv(vision=vision, throttle=True, text_mode=False, track_no=5, random_track=False, track_range=(0, 5))
    agent = ddpg(env_name, state_dim, action_dim, experiment_dir)

    MAX_STEPS = 2000
    step = 0

    rewards_every_steps = np.zeros([MAX_STEPS])
    actions_every_steps = np.zeros([MAX_STEPS, action_dim])

    agent.load_network()

    print("Testing Start.")
    start_time = time.time()
    try:
        i = 0
        while step < MAX_STEPS:
            # restart because of memory leak bug in torcs
            if np.mod(i, 3) == 0:
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            total_reward = 0.
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm,
                             0.0))

            while step < MAX_STEPS:
                a_t = agent.action(s_t)
                ob, r_t, done, info = env.step([a_t[0], 0.2, 0])
                s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, a_t[0]))
                total_reward += r_t
                s_t = s_t1
                rewards_every_steps[step] = r_t
                actions_every_steps[step] = a_t

                print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t )

                step += 1
                if done:
                    break

            i += 1

            print(("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward)))
            print(("Total Step: " + str(step)))
            print("")

    except:
        traceback.print_exc()
        with open((experiment_dir + "exception-test"), 'w') as file:
            file.write(str(traceback.format_exc()))

    finally:
        env.end()
        end_time = time.time()

        np.save(experiment_dir + "reward-test.npy", rewards_every_steps)
        np.save(experiment_dir + "action-test.npy", actions_every_steps)

        with open(experiment_dir + "log-test", 'w') as file:
            file.write("total step = %d\n" % step)
            file.write("total time = %s (s)\n" % str(end_time - start_time))

        print("Finish.")

if __name__ == "__main__":
    main()


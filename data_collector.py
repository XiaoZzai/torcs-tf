import numpy as np
import os
import time
import tensorflow as tf
import traceback
import cv2
np.random.seed(2018)

from gym_torcs import TorcsEnv
from ddpg import ddpg

def main():

    # Creating necessary directories
    collect_track_no = 5
    experiment_name = "tensorboard-4"
    experiment_dir  = "experiment-%s/" % experiment_name
    models_dir = experiment_dir + "model/"
    datas_dir = experiment_dir + "datas-track-no-%d/" % collect_track_no

    if os.path.exists(experiment_dir) == False:
        print("%s dosen't exists" % experiment_dir)
        return

    if os.path.exists(models_dir) == False:
        print("%s dosen't exists" % models_dir)
        return

    if os.path.exists(datas_dir) == False:
        os.mkdir(datas_dir)

    action_dim = 1
    state_dim  = 30
    env_name   = 'torcs'

    sess = tf.InteractiveSession()
    agent = ddpg(env_name, sess, state_dim, action_dim, models_dir)
    agent.load_network()

    vision = True
    env = TorcsEnv(vision=vision, throttle=True, text_mode=False, track_no=collect_track_no, random_track=False, track_range=(0, 3))

    print("Collecting Start.")
    max_data_entry_count = 10000
    data_entry_count = 0
    start_time = time.time()
    i = 0
    step = 0
    try:
        file = open(datas_dir + 'state-action-scalar', 'w')
        while data_entry_count < max_data_entry_count:
            if np.mod(i, 3) == 0:
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ,
                             ob.wheelSpinVel/100.0, ob.rpm, 0.0))
            pre_a_t = 0.0
            while data_entry_count < max_data_entry_count:
                a_t = agent.action(s_t)

                ob, r_t, done, info = env.step([a_t[0], 0.16, 0])

                print("Step", step, "Action", a_t, "Reward", r_t)

                s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ,
                                  ob.wheelSpinVel/100.0, ob.rpm, a_t[0]))

                image = ob.img
                # print(image)
                if step > 20:
                    cv2.imwrite(datas_dir + ("%d-%d.jpg" % (collect_track_no, data_entry_count)), image)
                    file.write("%f %f %f %f %f\n" % (ob.speedX, ob.speedY, ob.speedZ, pre_a_t, a_t[0]))
                    data_entry_count += 1

                s_t = s_t1
                step += 1
                pre_a_t = a_t[0]

                if done:
                    break

            print(("TOTAL REWARD @ " + str(i) + "Collect", data_entry_count))
            print(("Total Step: " + str(step)))
            print("")

    except:
        traceback.print_exc()
        with open((datas_dir + "exception"), 'w') as file:
            file.write(str(traceback.format_exc()))

    finally:

        file.close()
        
        env.end()
        end_time = time.time()

        with open(datas_dir + "log", 'w') as file:
            file.write("total_step = %d\n" % step)
            file.write("total_time = %s (s)\n" % str(end_time - start_time))

        print("Finish.")

if __name__ == "__main__":
    main()
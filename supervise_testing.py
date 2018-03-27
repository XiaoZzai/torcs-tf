import tensorflow as tf
import cv2
import os
import numpy as np
from supervise_model import Supervise
from gym_torcs import TorcsEnv
def main():
    # Creating necessary directories
    track_no = 5
    experiment_name = "tensorboard-4"
    experiment_dir  = "experiment-%s/" % experiment_name
    datas_dir = experiment_dir + "datas-track-no-%d/" % track_no
    models_dir = datas_dir + "model/"

    if os.path.exists(experiment_dir) == False:
        print("%s dosen't exists" % experiment_dir)
        return

    if os.path.exists(datas_dir) == False:
        print("%s dosen't exists" % datas_dir)
        return

    if os.path.exists(models_dir) == False:
        print("%s dosen't exists" % models_dir)
        return

    state_dim = 4
    img_dim = [304, 412, 3]
    sess = tf.InteractiveSession()
    agent = Supervise(sess, state_dim, img_dim, models_dir)
    agent.load_network()

    MAX_STEP = 10000
    step = 0
    vision = True
    env = TorcsEnv(vision=vision, throttle=True, text_mode=False, track_no=track_no, random_track=False, track_range=(5, 8))
    for i in range(1):
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s_t = np.hstack((ob.speedX, ob.speedY, ob.speedZ, 0.0))
        i_t = ob.img
        # print(i_t)

        while step < MAX_STEP:
            action = agent.action(s_t, i_t)
            ob, reward, done, info = env.step([action, 0.16, 0])
            s_t = np.hstack((ob.speedX, ob.speedY, ob.speedZ, action))
            i_t = ob.img

            print("Step", step, "Action", action, "Reward", reward)
            if done == True:
                break

    env.end()

if __name__ == "__main__":
    main()
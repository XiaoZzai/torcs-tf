import tensorflow as tf
import cv2
import os
import numpy as np
from supervise_model import Supervise

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
        os.mkdir(models_dir)

    state_dim = 4
    img_dim = [64, 64, 3]
    sess = tf.InteractiveSession()
    agent = Supervise(sess, state_dim, img_dim, models_dir)
    writter = tf.summary.FileWriter(models_dir, sess.graph)
    with tf.name_scope('summary'):
        cost_scalar = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("cost", cost_scalar)
    merged_summary = tf.summary.merge_all()
    agent.load_network()

    states = np.zeros([10000, state_dim])
    imgs = np.zeros([10000, img_dim[0], img_dim[1], img_dim[2]])
    steer_ground_truths = np.zeros([10000, 1])

    file = open(datas_dir + "state-action-scalar", 'r')
    for i in range(10000):
        img = cv2.imread(datas_dir + "%d-%d.jpg" % (track_no, i))
        imgs[i] = img / 255.0
        line = file.readline()
        data = line.split(' ')
        for j in range(state_dim):
            states[i][j] = float(data[j])
        steer_ground_truths[i][0] = float(data[4])

    file.close()

    MAX_STEP = 10000
    BATCH_SIZE = 32
    for step in range(MAX_STEP):
        indics = np.random.randint(0, 10000, BATCH_SIZE)

        state_batch = states[indics]
        img_batch = imgs[indics]
        steer_ground_truth_batch = steer_ground_truths[indics]
        # cv2.imshow("img", img_batch[0])
        # cv2.waitKeyEx(0)
        cost = agent.train(state_batch, steer_ground_truth_batch, img_batch)

        summary = sess.run(merged_summary, feed_dict={
            cost_scalar : cost
        })
        writter.add_summary(summary, step)

        if (step+1) % 2000 == 0:
            print("Saving model")
            agent.save_network(step)

        print("Step", step, "Loss", cost)

if __name__ == "__main__":
    main()
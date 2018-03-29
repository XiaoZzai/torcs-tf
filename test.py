import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

v1 = tf.Variable(30, dtype=tf.float32)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average = ema.apply([v1])

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    print(sess.run([v1, ema.average(v1)]))
    sess.run(tf.assign(v1, 50))
    # print(sess.run([v1, ema.average(v1)]))
    print(sess.run(maintain_average))
    # sess.run(tf.assign(v1, 5000))
    # print(sess.run([v1, ema.average(v1)]))
    print(sess.run(maintain_average))

    print(sess.run([v1, ema.average(v1)]))
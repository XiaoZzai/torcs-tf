import numpy as np
import datetime


def ornstein_uhlenbeck_process(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)


def formatted_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
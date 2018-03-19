#!/usr/bin/python3
from gym_torcs import TorcsEnv

env = TorcsEnv(vision=False, throttle=True, text_mode=False, track_no=0, random_track=False, track_range=(0, 3))
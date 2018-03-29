Using the (angle, track, trackPos, speedX, speedY, speedZ, wheelSpinVel, rpm, steer) as input, output (steer)
Training based on experiment-tensorboard-2

throttle = 0.16

brake = 0

reward = sp*np.cos(obs["angle"]) - np.abs(sp*np.sin(obs["angle"])) - sp * np.abs(obs["trackPos"]) / 2 - sp * np.abs(action_torcs["steer""]) * 2 - np.abs(sp*(action_torcs["steer"]-self.last_steer)) * 2

env = TorcsEnv(vision=False, throttle=True, text_mode=False, track_no=6, random_track=False, track_range=(5, 8))

abs(trackPos) > 0.9 is out of track

2018-03-20 13:17:05

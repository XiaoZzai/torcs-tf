from torcs import Torcs
import cv2
import gym
import numpy as np

class TorcsWrapper:
    def __init__(self, port=3101, noisy=False):
        self.episode = 0
        self.step_per_episode = 0
        self.step_total = 0

        self.env = Torcs(vision=True, port=port, noisy=noisy)

        # self.s_t = None

        # Discrete action space
        # self.steers = [-0.50, 0, 0.50]
        # self.action_space = gym.spaces.Discrete(len(self.steers))

    def reset(self, track_offset=0):

        # print("****************** Torcs Reseting !", self.episode)

        relaunch = False

        if self.episode % 3 == 0:
            relaunch = True

        self.episode += 1
        self.step_per_episode = 0

        ob = self.env.reset(relaunch=relaunch, track_offset=track_offset)
        # print(ob.img)
        # cv2.resize(ob.img, (320, 240))[:, 40:280]
        # print(np.asarray(ob.img).shape)
        img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 255.0
        # img = img.reshape(64, 64, 1)
        self.s_t = np.stack((img, img, img, img), axis=2)
        self.dist_start = ob.distFromStart
        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, 0.0))
        return self.s_t

    def step(self, steer_action):
        self.step_total += 1
        self.step_per_episode += 1
        ob, reward, done, _  = self.env.step(steer_action)
        # print(np.asarray(ob.img).shape)

        img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 255.0
        img = img.reshape(64, 64, 1)
        self.s_t = np.append(self.s_t[:, :, 1:], img, axis=2)

        # speed = np.sqrt(ob.speedX**2 + ob.speedY**2 + ob.speedZ**2)
        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, speed))
        return self.s_t, reward, done, ob.distFromStart - self.dist_start

    def end(self):
        self.env.end()

if __name__ == "__main__":
    env = TorcsWrapper()
    img = env.reset(1)
    while True:
        cv2.imshow("img", img[:, :, 3])
        cv2.waitKey(10)
        # print(img.shape)
        img, _, done, info = env.step(0.1)
        if done == True:
            break
        else:
            print(info)
    env.end()
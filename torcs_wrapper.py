from torcs import Torcs
import cv2
import gym
import numpy as np

class TorcsWrapper:
    def __init__(self, port=3101, noisy=False, action_repeatition=0, throttle=0.14):
        self.episode = 0
        self.step_per_episode = 0
        self.step_total = 0
        self.action_repeatition = action_repeatition

        self.env = Torcs(vision=True, port=port, noisy=noisy, throttle=0.14)

        # self.s_t = None

        # Discrete action space
        # self.steers = [-0.50, 0, 0.50]
        # self.action_space = gym.spaces.Discrete(len(self.steers))

    def reset(self, track_offset=0):
        relaunch = False

        if self.episode % 3 == 0:
            relaunch = True

        self.episode += 1
        self.step_per_episode = 0

        self.last_steer = 0.0
        if self.action_repeatition > 1:
            imgs = np.zeros([64, 64, self.action_repeatition])
            ob = self.env.reset(relaunch=relaunch, track_offset=track_offset)
            img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
            imgs[:, :, 0] = img
            for i in range(self.action_repeatition - 1):
                ob, reward, done, _ = self.env.step(self.last_steer)
                img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
                imgs[:, :, i+1] = img
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, self.last_steer))
            return [imgs, s_t]
        else:
            ob = self.env.reset(relaunch=relaunch, track_offset=track_offset)
            # print(ob.img)
            # cv2.resize(ob.img, (320, 240))[:, 40:280]
            # print(np.asarray(ob.img).shape)
            img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
            # img = img.reshape(64, 64, 1)
            self.s_t = np.stack((img, img, img, img), axis=2)
            self.dist_start = ob.distFromStart
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, self.last_steer))
            return [self.s_t, s_t]

    def step(self, steer_action):
        self.step_total += 1
        self.step_per_episode += 1

        if self.action_repeatition > 1:
            imgs = np.zeros([64, 64, 4])
            for i in range(self.action_repeatition):
                ob, reward, done, _ = self.env.step(steer_action)
                img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
                imgs[:, :, i] = img
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, self.last_steer))
            self.last_steer = steer_action
            return [imgs, s_t], reward, done, None
        else:
            ob, reward, done, _  = self.env.step(steer_action)
            # print(np.asarray(ob.img).shape)

            img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
            img = img.reshape(64, 64, 1)
            self.s_t = np.append(self.s_t[:, :, 1:], img, axis=2)

            # speed = np.sqrt(ob.speedX**2 + ob.speedY**2 + ob.speedZ**2)
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, self.last_steer))
            self.last_steer = steer_action
            return [self.s_t, s_t], reward, done, None # ob.distFromStart - self.dist_start

    def end(self):
        self.env.end()

if __name__ == "__main__":
    env = TorcsWrapper(action_repeatition=4)
    img = env.reset(1)
    while True:
        cv2.imshow("img", img[0][:, :, 1])
        cv2.waitKey(1)
        # print(img[0].shape)
        print(img[1])
        img, _, done, info = env.step(0.1)
        if done == True:
            break
        # else:
        #     print(info)
    env.end()
from gym import spaces
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time
import subprocess
import signal
import cv2

# Torcs env only for steer controlling
class Torcs:

    def __init__(self, vision=False, screenshot=False, noisy=False, port=3101):
        self.vision = vision
        self.screenshot = screenshot # If obtain image through screenshot or not
        self.port  = port
        self.noisy = noisy

        # self.observation_space = spaces.Box(low=-1, high=1, shape=[])
        # self.action_space = spaces.Box(low=-1, high=1, shape=[1]) # steer

        self.client = None
        self.torcs_proc = None

        self.terminal_judge_start = 30       # If after 100 timestep still no progress, terminated
        self.termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
        self.default_speed = 100

        self.initial_reset = True
        self.initial_run = True

        self.last_steer = 0.0

    def reset(self, relaunch=False, track_offset=0):
        self.time_step = 0

        if relaunch is True:
            self._reset_torcs(track_offset)
        else:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

        self.initial_reset = False
        self.client = snakeoil3.Client(p=self.port, vision=self.vision, torcs_restart_func=self._reset_torcs, track_offset=track_offset)
        self.client.MAX_STEPS = np.inf

        self.client.get_servers_input()
        raw_ob = self.client.S.d
        self.observation = self._make_observaton(raw_ob)

        self.last_u = None
        self.last_steer = 0.0
        return self.observation

    def step(self, steer):

        action = [steer, 0.16, 0]

        client = self.client
        this_action = self._agent_to_torcs(action)
        action_torcs = client.R.d

        action_torcs['steer'] = this_action['steer']
        action_torcs['accel'] = this_action['accel']
        action_torcs['brake'] = this_action['brake']
        action_torcs['gear'] = 1

        # Save the previous full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        client.respond_to_server()
        client.get_servers_input()

        raw_obs = client.S.d
        self.observation = self._make_observaton(raw_obs)

        # reward
        track = np.array(raw_obs['track'])
        trackPos = np.array(raw_obs['trackPos'])
        sp = np.array(raw_obs['speedX'])
        # damage = np.array(obs['damage'])

        reward = sp*np.cos(raw_obs["angle"]) - sp * np.abs(raw_obs['trackPos']) / 2 \
                    - sp * np.abs(action_torcs['steer']) * 4
        self.last_steer = action_torcs['steer']

        done = False

        # collision detection
        # if obs['damage'] - obs_pre['damage'] > 0:
        #     print("Collision")
        #     reward = -200
        #     done = True

        if abs(trackPos) > 0.9:
            print("Out of track")
            reward = -200
            done = True

        if self.terminal_judge_start < self.time_step:
           if sp < self.termination_limit_progress :
                print("No progress")
                reward = -200
                done = True

        if np.cos(raw_obs['angle']) < 0: # Episode is terminated if the agent runs backward
            print("Reversing")
            reward = -200
            done = True

        self.time_step += 1

        return self.observation, reward, done, {}

    def end(self):
        if self.screenshot == True:
            os.system("rm .tmp.png")
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)

    def _reset_torcs(self, track_offset):
        # kill existing torcs
        if self.torcs_proc != None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            self.torcs_proc = None
            time.sleep(1)

        # start new instance
        window_title = str(self.port)
        command = 'torcs -nofuel -nodamage -nolaptime -title %s -p %d' % (window_title, self.port)
        if (self.vision == True) and (self.screenshot == False):
            command += ' -vision'
        if self.noisy == True:
            command += ' -noisy'
        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        time.sleep(1)

        # select track
        command = "sh autostart.sh %d %d" % (self.port, track_offset)
        os.system(command)
        time.sleep(1)

    def _agent_to_torcs(self, action):
        torcs_action = {'steer' : action[0],
                        'accel' : action[1],
                        'brake' : action[2]
                        }

        return torcs_action

    def _obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)

        img = np.zeros(shape=[64, 64, 3], dtype=np.uint8)
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        # print(img.shape)
        return img

    def _make_observaton(self, raw_obs):
        names = ['speedX', 'speedY', 'speedZ', 'angle', 'damage', 'opponents', 'rpm', 'track', 'trackPos',
                 'wheelSpinVel', 'img', "distFromStart"]

        if self.vision == True:
            if self.screenshot == True:
                command = "sh screenshot.sh %d" % self.port
                os.system(command)
                image_rgb = cv2.imread(".tmp.png")
            else:
                image_rgb = self._obs_vision_to_image_rgb(raw_obs[names[-2]])
                image_rgb = cv2.flip(image_rgb, 0)
        else:
            image_rgb = None

        angle = np.array(raw_obs['angle'], dtype=np.float32) / 3.1415926
        track = np.array(raw_obs['track'], dtype=np.float32) / 200.
        trackPos = np.array(raw_obs['trackPos'], dtype=np.float32) / 1.

        if self.noisy == True:
            if np.random.random() < 0.05:
                angle = angle * (1 + np.random.randint(-20, 20) / 100)
            else:
                angle = angle * (1 + np.random.randint(-8, 8) / 100)

            if np.random.random() < 0.05:
                track = track * (1 + np.random.randint(-20, 20, 19) / 100)
            else:
                track = track * (1 + np.random.randint(-8, 8, 19) / 100)

            if np.random.random() < 0.05:
                trackPos = trackPos * (1 + np.random.randint(-20, 20) / 100)
            else:
                trackPos = trackPos * (1 + np.random.randint(-8, 8) / 100)

        angle = np.clip(angle, -1, 1)
        track = np.clip(track, 0, 1)
        trackPos = np.clip(trackPos, -1, 1)

        Observation = col.namedtuple('Observaion', names)
        return Observation(speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                           speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                           speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                           angle=angle,
                           damage=np.array(raw_obs['damage'], dtype=np.float32),
                           opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                           rpm=np.array(raw_obs['rpm'], dtype=np.float32) / 10000,
                           track=track,
                           trackPos=trackPos,
                           wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                           img=image_rgb,
                           distFromStart=raw_obs["distFromStart"])
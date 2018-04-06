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

class TorcsEnv:

    def __init__(self, vision=False, noisy=False, port=3101):
        self.vision = vision
        self.port  = port
        self.noisy = noisy

        self.client = None
        self.torcs_proc = None

        self.terminal_judge_start = 30       # If after 100 timestep still no progress, terminated
        self.termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
        self.default_speed = 100

        self.initial_reset = True
        self.initial_run = True

        self.last_steer = 0.0

    def reset_torcs(self, track_offset):
        # kill existing torcs
        if self.torcs_proc != None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            self.torcs_proc = None
            time.sleep(1)

        # start new instance
        window_title = str(self.port)
        command = 'torcs -nofuel -nodamage -nolaptime -title %s -p %d' % (window_title, self.port)
        if self.vision == True:
            command += ' -vision'
        if self.noisy == True:
            command += ' -noisy'
        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        time.sleep(1)

        # select track
        command = "sh autostart.sh %d %d" % (self.port, track_offset)
        os.system(command)
        time.sleep(1)

    def step(self, action):

        client = self.client
        this_action = self.agent_to_torcs(action)
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
        self.observation = self.make_observaton(raw_obs)

        # reward
        track = np.array(raw_obs['track'])
        trackPos = np.array(raw_obs['trackPos'])
        sp = np.array(raw_obs['speedX'])
        # damage = np.array(obs['damage'])

        # please compute upper bound and lower bound
        # progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos']) \
        #             - sp * np.abs(action_torcs['steer']) * 2 - np.abs(sp*(action_torcs['steer']-self.last_steer)) * 5

        progress = sp*np.cos(raw_obs["angle"]) - np.abs(sp*np.sin(raw_obs["angle"])) - sp * np.abs(raw_obs['trackPos'])  \
                    - sp * np.abs(action_torcs['steer']) * 3
        # progress = -np.abs(action_torcs['steer'])
        reward = progress
        # reward = 1
        self.last_steer = action_torcs['steer']

        done = False

        # collision detection
        # if obs['damage'] - obs_pre['damage'] > 0:
        #     print("Collision")
        #     reward = -200
        #     done = True

        if abs(trackPos) > 0.9:
            print("Out of track")
            reward = -10
            done = True

        if self.terminal_judge_start < self.time_step:
           if sp < self.termination_limit_progress :
                print("No progress")
                reward = -10
                done = True

        if np.cos(raw_obs['angle']) < 0: # Episode is terminated if the agent runs backward
            print("Reversing")
            reward = -10
            done = True

        self.time_step += 1

        return self.observation, reward, done, {}

    def reset(self, relaunch=False, track_offset=0):
        self.time_step = 0

        if relaunch is True:
            self.reset_torcs(track_offset)
        else:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

        print("### TORCS is RELAUNCHED ###")
        self.initial_reset = False
        self.client = snakeoil3.Client(p=self.port, vision=self.vision, torcs_restart_func=self.reset_torcs, track_offset=track_offset)
        self.client.MAX_STEPS = np.inf

        self.client.get_servers_input()
        raw_ob = self.client.S.d
        self.observation = self.make_observaton(raw_ob)

        self.last_u = None
        self.last_steer = 0.0
        return self.observation

    def end(self):
        # os.system("rm .tmp.png")
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)

    def agent_to_torcs(self, action):
        torcs_action = {'steer' : action[0],
                        'accel' : action[1],
                        'brake' : action[2]
                        }

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
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
        return img

    def make_observaton(self, raw_obs):
        names = ['focus', 'speedX', 'speedY', 'speedZ', 'angle', 'damage', 'opponents', 'rpm', 'track', 'trackPos',
                 'wheelSpinVel', 'img']

        if self.vision is True:
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[-1]])
            image_rgb = cv2.flip(image_rgb, 0)
            # command = "sh screenshot.sh %d" % self.port
            # os.system(command)
            # image_rgb = cv2.imread(".tmp.png")
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

        Observation = col.namedtuple('Observaion', names)
        return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                           speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                           speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                           speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                           angle=angle,
                           damage=np.array(raw_obs['damage'], dtype=np.float32),
                           opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                           rpm=np.array(raw_obs['rpm'], dtype=np.float32) / 10000,
                           track=track,
                           trackPos=trackPos,
                           wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                           img=image_rgb)
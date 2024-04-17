'''Gym Interface for Franka'''
import threading
import numpy as np
import gym
from gym import core, spaces
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import cv2
import copy
import time
import requests
import queue

from franka_env.envs.capture.rs_capture import RSCapture
from franka_env.envs.capture.video_capture import VideoCapture

class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.stop_signal = False
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break
            pair1 = np.concatenate([img_array['wrist_1_full'], img_array['wrist_2_full']], axis=0)
            pair2 = np.concatenate([img_array['side_2_full'], img_array['side_1_full']], axis=0)
            concatenated = np.concatenate([pair1, pair2], axis=1)
            cv2.imshow('wrist', concatenated/255.)
            cv2.waitKey(1)
# Create queue and worker thread at the beginning of your script

class FrankaFMB(gym.Env):
    def __init__(self, 
                 randomReset=np.zeros(6), 
                 hz = 10,
                 img_dim=(480, 640), # H x W
                 start_gripper=0,
                 ):
        

        self.resetpos = np.zeros(7)
        self.resetpos[:3] = np.array([0.5, 0.1, 0.2])
        self.reset_yaw=np.pi/2
        self.resetpos[3:] = self.euler_2_quat(np.pi, 0, self.reset_yaw )
        self.nextpos=self.resetpos
        self.currpos = self.resetpos[:].copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6,7))
        self.currgrip = start_gripper
        self.lastsent = time.time()
        self.randomreset = randomReset
        self.actionnoise = 0
        self.hz = hz

        ## NUC
        self.ip = '127.0.0.1'
        self.url = 'http://'+self.ip+':5000/'
        
        # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.35, -0.3, 0.02)),
            np.array((0.82, 0.3, 0.4)),
            dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((2*np.pi/3, -np.pi/3, -np.pi/2)),
            np.array((np.pi, np.pi/3, 5*np.pi/6)),
            dtype=np.float64
            )
        ## Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.array((-0.06, -0.06, -0.06, -0.25, -0.25, -0.25, 0-1e-8)),
            np.array((0.06, 0.06, 0.06, 0.25 , 0.25, 0.25, 1+1e-8))
        )
        self.img_dim = img_dim
        self.observation_space = spaces.Dict({
                                'side_1': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'side_1_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 1), dtype=np.uint16),
                                'side_1_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),

                                'side_2': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'side_2_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint16),
                                'side_2_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),
                                
                                'wrist_1': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'wrist_1_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint16),
                                'wrist_1_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),
                                
                                'wrist_2': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'wrist_2_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint16),
                                'wrist_2_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),
                                
                                
                                'tcp_pose': spaces.Box(-np.inf, np.inf, shape=(7,)),
                                'tcp_vel': spaces.Box(-np.inf, np.inf, shape=(6,)),
                                'gripper_pose': spaces.Box(-1, 1, shape=(1,), dtype=np.int8),
                                'q': spaces.Box(-np.inf, np.inf, shape=(7,)),
                                'dq': spaces.Box(-np.inf, np.inf, shape=(7,)),
                                'tcp_force': spaces.Box(-np.inf, np.inf, shape=(3,)),
                                'tcp_torque': spaces.Box(-np.inf, np.inf, shape=(3,)),
                                'jacobian': spaces.Box(-np.inf, np.inf, shape=((6,7))),
                                'gripper_dist': spaces.Box(-np.inf, np.inf, shape=(1,)),
                            })

        self.cap_wrist_1 = VideoCapture(RSCapture(name='wrist_1', serial_number='127122270350', depth=True))
        self.cap_wrist_2 = VideoCapture(RSCapture(name='wrist_2', serial_number='128422271851', depth=True))
        self.cap_side_1 = VideoCapture(RSCapture(name='side_1', serial_number='128422270679', depth=True))
        self.cap_side_2 = VideoCapture(RSCapture(name='side_2', serial_number='127122270146', depth=True))   
        self.cap = {'side_1': self.cap_side_1,
                    'side_2': self.cap_side_2,
                    'wrist_1': self.cap_wrist_1,
                    'wrist_2': self.cap_wrist_2,}
        print("Initialized Franka")
        if start_gripper==0:
            requests.post(self.url + 'open')

        self.img_queue = queue.Queue()
        self.displayer = ImageDisplayer(self.img_queue)
        self.displayer.start()


    def recover(self):
        requests.post(self.url + 'clearerr')
        
    def _send_pos_command(self, pos):
        self.recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + 'pose', json=data)

    def update_currpos(self):
        ps = requests.post(self.url + 'getstate').json()
        self.currpos[:] = np.array(ps['pose'])
        self.currvel[:] = np.array(ps['vel'])
        self.currforce[:] = np.array(ps['force'])
        self.currtorque[:] = np.array(ps['torque'])
        self.currjacobian[:] = np.reshape(np.array(ps['jacobian']), (6,7))
        self.q[:] = np.array(ps['q'])
        self.dq[:] = np.array(ps['dq'])
        self.gripper_dist = np.array(ps['gripper'])




    def set_gripper(self, position):
        # print("CALLED GRIPPER", position)
        if position != self.currgrip:
            if position == 1:
                st = 'close'
                self.currgrip = 1
            else:
                st = 'open'
                self.currgrip = 0
        else:
            return

        ### IMPORTANT, IF FRANKA GRIPPER GETS OPEN/CLOSE COMMANDS TOO QUICKLY IT WILL FREEZE
        delta = time.time() - self.lastsent
        time.sleep(max(0, 1 - delta))

        requests.post(self.url + st)
        if st == 'close':
            time.sleep(1.2)
        else:
            time.sleep(0.6)
        self.lastsent = time.time()


    def clip_safety_box(self, pose):
        pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        euler = Rotation.from_quat(pose[3:]).as_euler('xyz')
        old_sign = np.sign(euler[0])
        euler[0] = np.clip(euler[0]*old_sign, self.rpy_bounding_box.low[0], self.rpy_bounding_box.high[0]) * old_sign
        euler[1:] = np.clip(euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:])   
        pose[3:] = Rotation.from_euler('xyz', euler).as_quat()
        return pose
    
    
    def move_to_pos(self, pos):
        start_time = time.time()
        if len(pos[3:]) == 3:
            trans = pos[:3]
            quat = self.euler_2_quat(pos[3], pos[4], pos[5])
            pos = np.concatenate([trans, quat])
        self._send_pos_command(self.clip_safety_box(pos))
        dl = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dl))
        self.update_currpos()
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.update_currpos()
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.actionnoise > 0:
            a = action[:3] + np.random.uniform(-self.actionnoise, self.actionnoise, (3,))
        else:
            a = action[:3]

        self.nextpos = copy.deepcopy(self.currpos[:])
        self.nextpos[:3] = self.nextpos[:3] + a
        self.nextpos[3:] = (Rotation.from_euler('xyz', action[3:6]) * Rotation.from_quat(self.currpos[3:])).as_quat() 

        self.nextpos = self.clip_safety_box(self.nextpos)
        self._send_pos_command(self.nextpos)
        self.set_gripper(action[-1])

        self.curr_path_length +=1
        done = False
        dl = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dl))

        self.update_currpos()
        obs = self._get_obs()
        return obs, 0, done, {}



    def _get_im(self):
        images = {}
        for key, cap in self.cap.items():
            try:
                rgb, depth = cap.read()
                images[key] = cv2.resize(rgb, (256, 256))
                images[key + "_full"] = rgb
                images[f"{key}_depth"] = depth
            except queue.Empty:
                input(f'{key} camera frozen. Check connect, then press enter to relaunch...')
                cap.close()
                if key == 'side_1':
                    cap = RSCapture(name='side_1', serial_number='128422270679', depth=True)
                elif key == 'side_2':
                    cap = RSCapture(name='side_2', serial_number='127122270146', depth=True)
                elif key == 'wrist_1':
                    cap = RSCapture(name='wrist_1', serial_number='127122270350', depth=True)
                elif key == 'wrist_2':
                    cap = RSCapture(name='wrist_2', serial_number='128422271851', depth=True)
                else:
                    raise KeyError
                self.cap[key] = VideoCapture(cap)
                return self._get_im()

        self.img_queue.put(images)
        return images
   
    def _get_state(self):
        state_observation = {
            'tcp_pose': self.currpos,
            'tcp_vel': self.currvel,
            'gripper_pose': self.currgrip,
            'q': self.q,
            'dq': self.dq,
            'tcp_force': self.currforce,
            'tcp_torque': self.currtorque,
            'jacobian': self.currjacobian,
            'gripper_dist': self.gripper_dist,
        }
        return state_observation

    def _get_obs(self):
        images = self._get_im()
        state_observation = self._get_state()

        return copy.deepcopy(images) | copy.deepcopy(state_observation)

    def go_to_rest(self, jpos=False):
        count = 0
        self.update_currpos()
        restp_new = copy.deepcopy(self.currpos)
        restp_new[2] = 0.3
        dp = restp_new - self.currpos
        count_1 = 0
        while ((np.linalg.norm(dp[:3])>0.03 or np.linalg.norm(dp[3:])>0.04)) and count_1 < 50:
            if np.linalg.norm(dp[3:]) > 0.2:
                dp[3:] = 0.2*dp[3:]/np.linalg.norm(dp[3:])
            if np.linalg.norm(dp[:3]) > 0.07:
                dp[:3] = 0.07*dp[:3]/np.linalg.norm(dp[:3])
            self._send_pos_command(self.currpos + dp)
            time.sleep(0.1)
            self.update_currpos()
            dp = restp_new - self.currpos
            count_1 += 1

        if jpos:
            self.go_to_rest(jpos=False)
            print("JOINT RESET")
            requests.post(self.url + 'jointreset')
        else:    
            print("RESET")
            self.update_currpos()
            restp = copy.deepcopy(self.resetpos[:])
            # resetroll, resetpitch, restyaw = self.quat_2_euler(restp[3:])
            restp[:3] += np.random.uniform(-self.randomreset[:3], self.randomreset[:3])
            # restyaw += np.random.uniform(-self.randomreset[3], self.randomreset[3])
            # resetroll += np.random.uniform(-self.randomreset[4], self.randomreset[4])
            # resetpitch += np.random.uniform(-self.randomreset[5], self.randomreset[5])
            rand_rpy = np.random.uniform(-self.randomreset[3:], self.randomreset[3:])
            rand_quat = Rotation.from_euler('xyz', rand_rpy)
            restp[3:] = (Rotation.from_quat(restp[3:]) * rand_quat).as_quat()

            restp_new = copy.deepcopy(restp)
            restp_new[2] = 0.3
            dp = restp_new - self.currpos
            while count < 200 and (np.linalg.norm(dp[:3])>0.01 or np.linalg.norm(dp[3:])>0.04):
                if np.linalg.norm(dp[3:]) > 0.2:
                    dp[3:] = 0.2*dp[3:]/np.linalg.norm(dp[3:])
                if np.linalg.norm(dp[:3]) > 0.09:
                    dp[:3] = 0.09*dp[:3]/np.linalg.norm(dp[:3])
                self._send_pos_command(self.currpos + dp)
                time.sleep(0.1)
                self.update_currpos()
                dp = restp_new - self.currpos
                count += 1

            dp = restp - self.currpos
            count = 0
            while count < 200 and (np.linalg.norm(dp[:3])>0.01 or np.linalg.norm(dp[3:])>0.04):
                if np.linalg.norm(dp[3:]) > 0.2:
                    dp[3:] = 0.2*dp[3:]/np.linalg.norm(dp[3:])
                if np.linalg.norm(dp[:3]) > 0.07:
                    dp[:3] = 0.07*dp[:3]/np.linalg.norm(dp[:3])
                self._send_pos_command(self.currpos + dp)
                time.sleep(0.1)
                self.update_currpos()
                dp = restp - self.currpos
                count += 1
        return count<50

    def reset(self, jpos=None, gripper=0, require_input=True):
        requests.post(self.url+ 'precision_mode')
        self.set_gripper(gripper)
        self.update_currpos()
        if jpos == None:
            jpos = (np.abs(self.q[0])>0.3)

        success = self.go_to_rest(jpos=jpos)
        self.curr_path_length = 0
        self.recover()
        if jpos==True:
            self.go_to_rest(jpos=False)
            self.recover()

        if require_input:
            input('Reset Environment, Press Enter Once Complete: ')
        self.update_currpos()
        # self.last_quat = self.currpos[3:]
        o = self._get_obs()
        requests.post(self.url+ 'compliance_mode')

        return o
    
    def precision_mode(self):
        requests.post(self.url+ 'precision_mode')

    def compliance_mode(self):
        requests.post(self.url+ 'compliance_mode')


    def insertion_mode(self):
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((np.pi - 0.3, -0.3, -np.pi/2)),
            np.array((np.pi+0.3, 0.3, 5*np.pi/6)),
            dtype=np.float64
            )
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.3, -0.35, 0.06)),
            np.array((0.82, 0.3, 0.4)),
            dtype=np.float64
        )
        
    def freespace_mode(self):
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((2*np.pi/3, -np.pi/3, -np.pi/2)),
            np.array((np.pi, np.pi/3, 5*np.pi/6)),
            dtype=np.float64
            )
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.35, -0.3, 0.02)),
            np.array((0.82, 0.3, 0.4)),
            dtype=np.float64
        )

    def grasp_mode(self):
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((np.pi - 0.05, -0.05, -np.pi/2)),
            np.array((np.pi, 0.05, 5*np.pi/6)),
            dtype=np.float64
            )
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.32, -0.3, 0.02)),
            np.array((0.82, 0.3, 0.4)),
            dtype=np.float64
        )

    def quat_2_euler(self, quat):
        # calculates and returns: yaw, pitch, roll from given quaternion
        if not isinstance(quat, Quaternion):
            quat = Quaternion(quat)
        yaw, pitch, roll = quat.yaw_pitch_roll
        return yaw + np.pi, pitch, roll


    def euler_2_quat(self, yaw=np.pi/2, pitch=0.0, roll=np.pi):
        yaw = np.pi - yaw
        yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0], [0, 0, 1.0]])
        pitch_matrix = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0, np.cos(pitch)]])
        roll_matrix = np.array([[1.0, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
        return Quaternion(matrix=rot_mat).elements


    def close(self):
        [cap.close() for cap in self.cap.items]
        self.displayer.join()
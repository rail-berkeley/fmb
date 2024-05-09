import os
from functools import partial
import cloudpickle as pickle
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
import cv2

import jax
import jax.numpy as jnp

import absl.app
import absl.flags
from pathlib import Path

from .jax_utils import JaxRNG, next_rng, wrap_function_with_rng
from .model import TanhGaussianResNetMixedPolicy, TanhGaussianResNetPolicy
from .utils import define_flags_with_default, set_random_seed
from flax.training.train_state import TrainState
import robot_infra
import gym

should_reset = False
NUM_PEGS = 0
FLAGS_DEF = define_flags_with_default(
    load_checkpoint='',
    model_key='train_state',
    primitive='',
    horizon=100,
    gpu=True,
    deterministic=False,
    seed=1234,
)
FLAGS = absl.flags.FLAGS
def on_press(key):
    global should_reset
    try:
        if str(key) == 'Key.esc':
            should_reset = True
    except AttributeError:
        pass

def rotate_observation(obs):
    obs = copy.deepcopy(obs)
    state = obs['tcp_pose']
    velocity = obs['tcp_vel']
    wrench = np.concatenate((obs['tcp_force'], obs['tcp_torque']), axis=0)

    #Rotation matrix
    r = R.from_quat(state[3:7])
    r = r.as_matrix()

    #Translation vector
    p = state[:3]

    #Translation hat
    p_hat = np.array([[0, -p[2], p[1]],
    [p[2], 0, -p[0]],
    [-p[1], p[0], 0]])

    #Adjoint
    adjoint = np.zeros((6,6))
    adjoint[:3, :3] = r
    adjoint[3:, :3] = p_hat@r
    adjoint[3:, 3:] = r
    adjoint_inv = np.linalg.inv(adjoint)
    #Velocity in base frame
    velocity = adjoint_inv @ velocity

    #Force in base frame
    wrench = adjoint_inv @ wrench
    
    obs['tcp_vel'] = velocity
    obs['tcp_force'] = wrench[:3]
    obs['state_observation']['tcp_torque'] = wrench[3:]
    return obs

def rotate_action(obs, action):
    obs = copy.deepcopy(obs)
    state = obs['state_observation']['tcp_pose']

    #Rotation matrix
    r = R.from_quat(state[3:7])
    r = r.as_matrix()

    #Translation vector
    p = state[:3]

    #Translation hat
    p_hat = np.array([[0, -p[2], p[1]],
    [p[2], 0, -p[0]],
    [-p[1], p[0], 0]])

    #Adjoint
    adjoint = np.zeros((6,6))
    adjoint[:3, :3] = r
    adjoint[3:, :3] = p_hat@r
    adjoint[3:, 3:] = r

    #Action in base frame
    action = adjoint @ action

    return action

def rollout(
        env,
        agent,
        preprocess_obs_for_policy_fn,
        max_path_length=np.inf,
        o=None,
        primitive='grasp',
        tcp_frame=False,
        reset=True,
):

    observations = []
    actions = []
    path_length = 0
    last_action = np.zeros(7)
    if reset:
        o = env.reset(gripper= 1 if primitive=='insert' else 0)
    else:
        o = o
    if primitive=='place_on_fixture' or primitive=='rotate':
        o, _, _, _ = env.step(np.array([0,0,0,0,0,0,1]))
    global should_reset, NUM_PEGS
    should_reset = False
    if primitive=='insert':
        env.insertion_mode()

    if NUM_PEGS == 0:
        peg_vec = None
    else:
        peg_vec = np.zeros((1, NUM_PEGS))
        try:
            peg_id = int(input("Enter peg id: "))
        except: 
            return dict(
                observations=observations,
                actions=actions,
            )        
        peg_vec[0, peg_id-1] = 1

    while path_length < max_path_length:
        if should_reset:
            break
        robot_state, image = preprocess_obs_for_policy_fn(o, last_action, tcp_frame=tcp_frame, observations=observations)
        #Get action from policy
        action, agent_info = agent(robot_state, image, peg_vec)
        if (action.size == 7 or action.size == 6) and tcp_frame:
            action = action.at[:6].set(rotate_action(o, a[:6]))
        elif tcp_frame:
            raise NotImplementedError

        if action.size % 7 == 0:
            action = action.reshape((-1,7))
        elif action.size % 6 == 0:
            action = action.reshape((-1,6))
        else:
            raise NotImplementedError

        for a in action:
            if a[6] > 0.5:
                a =a.at[6].set(1)
            else:
                a = a.at[6].set(0)
            a = a.at[:6].set(a[:6] * env.action_space.high[:6])
            next_o, _, _, _= env.step(a)
            observations.append(o)
            actions.append(a)
            last_action = a
            path_length += 1
            o = next_o
    env.freespace_mode()
    return dict(
        observations=observations,
        actions=actions,
    )

def general_preprocessor(obs, last_action=None, tcp_frame=False, image_keys=[], state_keys=[], use_last_action=False, num_frame_stack=1, observations=[]):
    if tcp_frame:
        obs = rotate_observation(obs)
    images = []
    for key in image_keys:
        previous_frames=[]
        if "_depth" not in key:
            processed_img = cv2.resize(obs[key], (256,256)).reshape((1,256,256,3)).astype(np.float32) / 255.0
        elif "_depth" in key:
            processed_img = cv2.resize(obs[key], (256,256)).reshape((1,256,256,1)).astype(np.float32) / 65535.0
        previous_frames.append(processed_img)
        
        for i in range(1, num_frame_stack):
            if i <= len(observations):
                if "_depth" not in key:
                    processed_img = cv2.resize(observations[-i][key], (256,256)).reshape((1,256,256,3)).astype(np.float32) / 255.0
                elif "_depth" in key:
                    processed_img = cv2.resize(observations[-i][key], (256,256)).reshape((1,256,256,1)).astype(np.float32) / 65535.0
            previous_frames.append(processed_img)
        stacked_frames = np.concatenate(previous_frames[::-1], axis=-1)
        images.append(stacked_frames)

    states = []
    for key in state_keys:
        if key == 'tcp_vel':
            states.append(obs[key].reshape((1,-1)))
        else:
            states.append(obs[key].reshape((1,-1)))
    if use_last_action:
        states.append(last_action[:6].reshape((1,-1)))
    
    state_observation = np.concatenate(states, axis=1).reshape((1,-1))

    return state_observation, images

def simulate_policy(args, env, policy, processor, primitive, tcp_frame, N=20, traj_length=50):
    for i in range(N):
        paths = []
        print("Number: ", i+1)
        paths.append(rollout(
            env,
            policy,
            max_path_length=traj_length,
            preprocess_obs_for_policy_fn=processor,
            primitive=primitive,
            tcp_frame=tcp_frame,
        ))
        if primitive in ['grasp', 'regrasp']:
            observations = []
            actions = []
            o=paths[-1]['observations'][-1]
            observations.append(o)
            o, _, _, _ = env.step(np.array([0,0,0,0,0,0,1]))
            actions.append(np.array([0,0,0,0,0,0,1]))
            for _ in range(2):
                observations.append(o)
                a = np.zeros(7)
                a[2] = 0.03
                a[-1] = 1
                o, _, _, _= env.step(a)
                actions.append(a)
            traj = dict(
                observations=observations,
                actions=actions,
            )

            for key in paths[-1].keys():
                paths[-1][key] += traj[key]
    
        # Save videos
        saved_model_path = FLAGS.load_checkpoint 
        video_dir = os.path.join(os.path.dirname(saved_model_path), 'videos')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        video_path = Path(os.path.join(video_dir, f'{FLAGS.model_key}_0.mp4'))
        i=0
        while video_path.exists():
            i+=1
            video_path = Path(os.path.join(video_dir, f'{FLAGS.model_key}_{i}.mp4'))
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (640*2, 480*2))
        for i in range(len(paths)):
            for s in range(len(paths[i]['observations'])):
                images = paths[i]['observations'][s]
                frame1 = np.concatenate([images['side_1_full'][..., :3], images['side_2_full'][..., :3]], axis=1).astype(np.uint8)
                frame2 = np.concatenate([images['wrist_1_full'][..., :3], images['wrist_2_full'][..., :3]], axis=1).astype(np.uint8)
                frame = np.vstack((frame1, frame2))
                writer.write(frame)
            for _ in range(10):
                writer.write(np.zeros((480*2, 1280, 3), dtype=np.uint8))
        writer.release()


def main(argv):
    global NUM_PEGS
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.load_checkpoint != ''
    set_random_seed(FLAGS.seed)
    with open(FLAGS.load_checkpoint, 'rb') as fin:
        checkpoint_data = pickle.load(fin)
    checkpoint_policy_config = {
        k[7:]: v for k, v in checkpoint_data['variant'].items()
        if k.startswith('policy.')
    }
    image_keys = checkpoint_data['variant']['dataset_image_keys'].split(':')
    num_action_chunk = checkpoint_data['variant']['num_action_chunk'] if 'num_action_chunk' in checkpoint_data['variant'].keys() else 1

    checkpoint_policy_config['resnet_type'] = tuple([(checkpoint_data['variant']['resnet_type']+'Depth' if 'depth' in key else checkpoint_data['variant']['resnet_type']) for key in image_keys])
    if checkpoint_data['variant']['train_gripper']:
        policy_config = TanhGaussianResNetMixedPolicy.get_default_config()
        policy_config.update_from_flattened_dict(checkpoint_policy_config)
        policy = TanhGaussianResNetMixedPolicy(output_dim=7*num_action_chunk, config_updates=policy_config,)
    else:
        policy_config = TanhGaussianResNetPolicy.get_default_config()
        policy_config.update_from_flattened_dict(checkpoint_policy_config)
        policy = TanhGaussianResNetPolicy(output_dim=6*num_action_chunk, config_updates=policy_config,
                                               model=checkpoint_data['variant']['encoder'] if 'encoder' in checkpoint_data['variant'] else 'ResNet',
                                          )
    train_gripper = checkpoint_data['variant']['train_gripper']
    params = checkpoint_data[FLAGS.model_key]
    if type(params) == TrainState:
        params = params.params

    @wrap_function_with_rng(next_rng())
    @jax.jit
    def forward_policy(rng, robot_state, images, peg_vec=None):
        rng_generator=JaxRNG(rng)
        action, log_prob = policy.apply(
            params, robot_state, images, peg_vec, None,
            deterministic=FLAGS.deterministic,
            rngs=rng_generator(policy.rng_keys())
        )
        action = action.reshape(-1)
        if not train_gripper:
            action = jnp.append(action, 1)
        return action, {}
    NUM_PEGS = checkpoint_data['variant']['num_pegs']
    
    state_dim = {'tcp_pose': 7, 'tcp_vel': 6, 'gripper_pose': 1, 'q':7, 'dq':7, 'tcp_force':3, 'tcp_torque':3}

    image_keys = checkpoint_data['variant']['dataset_image_keys'].split(':')
    state_keys = checkpoint_data['variant']['state_keys'].split(':')
    use_last_action = checkpoint_data['variant']['last_action']
    action_dim = 6 if use_last_action else 0
    tcp_frame = checkpoint_data['variant']['tcp_frame']
    train_gripper = checkpoint_data['variant']['train_gripper']
    num_frame_stack = checkpoint_data['variant']['num_frame_stack'] if 'num_frame_stack' in checkpoint_data['variant'].keys() else 1
    rgb_keys = [item for item in image_keys if not item.endswith('_depth')]
    depth_keys = [item for item in image_keys if item.endswith('_depth')]
    print('Frame: ', 'tcp_frame' if tcp_frame else 'base_frame')
    print('State: ', policy_config.state_injection)
    print('States: ', state_keys)
    print('Views: ', image_keys)
    print('Last action: ', use_last_action)
    print('Train gripper: ', train_gripper)
    print(f'Numer of frames stacked: {num_frame_stack}')
    print(f'Action chunk: {num_action_chunk}')
    robot_state = jnp.zeros((1, sum([state_dim[key] for key in state_keys]) + action_dim))
    robot_state = robot_state.at[0, robot_state.shape[1]].set(1)
    images = [jnp.zeros((1, 256, 256, 3*num_frame_stack)) for _ in range(len(rgb_keys))] + [jnp.zeros((1, 256, 256, 1*num_frame_stack)) for _ in range(len(depth_keys))]

    # Trigger jit compilation
    forward_policy(robot_state, images, np.zeros((1, NUM_PEGS)) if NUM_PEGS > 0 else None)

    processor = partial(general_preprocessor, image_keys=image_keys, state_keys=state_keys, use_last_action=use_last_action, num_frame_stack=num_frame_stack)
    
    primitive = FLAGS.primitive
    if primitive == 'grasp' or primitive=='long_horizon':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=0) 
        env.resetpos[:3] = np.array([0.45, 0.1, 0.22])
        env.reset_yaw=np.pi/2
        env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)
    elif primitive == 'place_on_fixture':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=1) 
        env.resetpos[:3] = np.array([0.45, 0.1, 0.03])
        env.reset_yaw=np.pi/2
        env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)
    elif primitive=='rotate':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=1) 
        env.resetpos[:3] = np.array([0.45, 0.1, 0.15])
        env.reset_yaw=np.pi/2
        env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)
    elif primitive == 'regrasp':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=0) 
        env.resetpos[:3] = np.array([0.6054150172032604,0.06086102586621338,0.1046158263847778])
        env.resetpos[3:] = np.array([0.6299743988034661,0.6971103396175138,-0.23715609242845734,0.24683277552754387])
    elif primitive == 'insert':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=1) 
        env.resetpos[:3] = np.array([0.5,-0.18,0.33])
        env.resetpos[3:] = np.array([0.7232287667289446,0.6900075685348428,-0.010295688497306624,-0.026901768319765842])
    else:
        raise ValueError("Unrecongized primitive name")

    traj_length ={
        'grasp':60,
        'insert': 100,
        'long_horizon': 300,
        'regrasp': 60,
        'place_on_fixture': 100,
        'rotate': 200,
    }

    simulate_policy(FLAGS, env, forward_policy, processor=processor, primitive=primitive, tcp_frame=tcp_frame, N=100, traj_length=traj_length[primitive])


if __name__ == "__main__":
    absl.app.run(main)
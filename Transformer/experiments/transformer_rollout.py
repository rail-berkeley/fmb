import copy
import os
from functools import partial
import time
from pynput import keyboard
import numpy as np
import cv2
from flax.training import checkpoints
from orca.data.rlds.fmb_primitive_peg_map import FMB_PRIMITIVE_TO_ID_DICT

import jax
import jax.numpy as jnp
import wandb

import absl
import absl.app
from absl import flags
from pathlib import Path

import sys;

sys.path.append('/home/panda/code/franka-env')
import franka_env
import gym

import jax
# import matplotlib.pyplot as plt
from orca.model import create_model_def_fmb
from functools import partial
from orca.train_utils import create_train_state
import optax
import functools as ft


should_reset = False

FLAGS = absl.flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("primitive", None, "Name of primitive", required=True)
flags.DEFINE_integer("peg", None, "Name of peg", required=True)
flags.DEFINE_string("wandb_run_name", None, "Name of wandb run", required=True)
flags.DEFINE_integer("obs_horizon", 1, "Observation history length")
flags.DEFINE_integer("im_size", 256, "Image size")
flags.DEFINE_list("act_mean", None, "Mean values for action normalization")
flags.DEFINE_list("act_std", None, "Standard deviation values for action normalization")
ACT_MEAN = [float(val) for val in FLAGS.act_mean]
ACT_STD = [float(val) for val in FLAGS.act_std]

def unnormalize_action(action, mean, std):
    return action * std + mean


@partial(jax.jit, static_argnames="argmax")
def sample_actions(observations, state, rng, argmax=False, temperature=1.0):
    observations = jax.tree_map(lambda x: x[None], observations)
    observations['peg_id'] = observations['peg_id'][:, 0]
    observations['primitive_id'] = observations['primitive_id'][:, 0]
    actions = state.apply_fn(
        {"params": state.params},
        observations,
        train=False,
        argmax=argmax,
        rng=rng,
        temperature=temperature,
        method="predict_action",
    )
    return actions[0]


def load_checkpoint(path, wandb_run_name):
    # load information from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)

    example_actions = np.zeros((1, 7), dtype=np.float32)
    example_obs = {
        "image_wrist_1": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3)) / 127.5 - 1.0,
        "image_wrist_2": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3)) / 127.5 - 1.0,
        "image_side_1": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3)) / 127.5 - 1.0,
        "image_side_2": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3)) / 127.5 - 1.0,
        "primitive_id": np.zeros((1,), dtype=np.uint8),
        "peg_id": np.zeros((1,), dtype=np.uint8),
        "ee_ft": np.zeros((1, 6), dtype=np.float32),
        "ee_vel": np.zeros((1, 6), dtype=np.float32),
        "ee_pose": np.zeros((1, 7), dtype=np.float32),
    }

    example_batch = {
        "observations": example_obs,
        "actions": example_actions,
    }

    # create train_state from wandb config
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)
    # breakpoint()
    model_def = create_model_def_fmb(
        action_dim=example_batch["actions"].shape[-1],
        time_sequence_length=1,
        **run.config["model"],
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=run.config["optimizer"]["learning_rate"],
        warmup_steps=run.config["optimizer"]["warmup_steps"],
        decay_steps=run.config["optimizer"]["decay_steps"],
        end_value=0.0,
    )
    tx = optax.adam(lr_schedule)
    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observations"],
            example_batch["actions"],
        ),
    )

    try:
        train_state = checkpoints.restore_checkpoint(path, train_state)
    except:
        ckpt_state = checkpoints.restore_checkpoint(path, None)
        token_0 = ckpt_state['params']['observation_tokenizers_0']
        token_1 = ckpt_state['params']['observation_tokenizers_1']

        ckpt_state['params']['observation_tokenizers_0'] = token_1
        ckpt_state['params']['observation_tokenizers_1'] = token_0

        train_state = train_state.replace(params=ckpt_state['params'])

    action_mean = np.array(ACT_MEAN)
    action_std = np.array(ACT_STD)

    return train_state, action_mean, action_std


def on_press(key):
    global should_reset
    try:
        if str(key) == 'Key.esc':
            should_reset = True
    except AttributeError:
        pass


def rollout(
        env,
        agent,
        preprocess_obs_for_policy_fn,
        max_path_length=np.inf,
        o=None,
        primitive='grasp',
        reset=True,
):
    observations = []
    actions = []
    path_length = 0
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

    while path_length < max_path_length:
        if should_reset:
            break
        # Get action from policy
        agent_obs = preprocess_obs_for_policy_fn(o,
        )
        action, agent_info = agent(agent_obs)
        action[:6] = action[:6] * env.action_space.high[:6]

        next_o, _, _, _ = env.step(action)
        observations.append(o)
        actions.append(action)
        last_action = action
        path_length += 1
        o = next_o
    env.freespace_mode()
    return dict(
        observations=observations,
        actions=actions,
    )


def general_preprocessor(obs, primitive_id, peg_id):
    return {
        "image_wrist_1": obs['wrist_1'] / 127.5 - 1.0,
        "image_wrist_2": obs['wrist_2'] / 127.5 - 1.0,
        "image_side_1": obs['side_1'] / 127.5 - 1.0,
        "image_side_2": obs['side_2'] / 127.5 - 1.0,
        "ee_pose": obs['tcp_pose'],
        "ee_vel": obs['tcp_vel'],
        "ee_ft": np.concatenate((obs['tcp_force'], obs['tcp_torque']), axis=-1),
        "primitive_id": primitive_id,
        "peg_id": peg_id,
    }


def simulate_policy(args, env, policy, processor, primitive, N=20, traj_length=50):
    paths = []
    for i in range(N):
        print("Number: ", i + 1)
        paths.append(rollout(
            env,
            policy,
            max_path_length=traj_length,
            preprocess_obs_for_policy_fn=processor,
            primitive=primitive,
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
        saved_model_path = FLAGS.checkpoint_path
        video_dir = os.path.join(saved_model_path, 'videos')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        video_path = Path(os.path.join(video_dir, f'video_0.mp4'))
        i = 0
        while video_path.exists():
            i += 1
            video_path = Path(os.path.join(video_dir, f'video_{i}.mp4'))
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (640 * 2, 480 * 2))
        for i in range(len(paths)):
            for s in range(len(paths[i]['observations'])):
                images = paths[i]['observations'][s]
                frame1 = np.concatenate([images['side_1_full'][..., :3], images['side_2_full'][..., :3]], axis=1).astype(
                    np.uint8)
                frame2 = np.concatenate([images['wrist_1_full'][..., :3], images['wrist_2_full'][..., :3]], axis=1).astype(
                    np.uint8)
                frame = np.vstack((frame1, frame2))
                writer.write(frame)
            for _ in range(10):
                writer.write(np.zeros((480 * 2, 1280, 3), dtype=np.uint8))
        writer.release()
        paths = []


def main(argv):
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    rng = jax.random.PRNGKey(42)
    key, rng = jax.random.split(rng)
    train_state, action_mean, action_std = load_checkpoint(FLAGS.checkpoint_path, FLAGS.wandb_run_name)

    def forward_policy(obs):
        actions = sample_actions(obs, train_state, rng=key, argmax=False, temperature=1)
        actions = unnormalize_action(actions, action_mean, action_std)
        actions = np.array(actions)
        actions[-1] = 1 if actions[-1] > 0.5 else 0
        return actions, {}

    processor = ft.partial(
        general_preprocessor,
        primitive_id=np.array([FMB_PRIMITIVE_TO_ID_DICT[FLAGS.primitive]], dtype=np.uint8),
        peg_id=np.array([FLAGS.peg], dtype=np.uint8),
    )

    primitive = FLAGS.primitive
    primitive_id = FMB_PRIMITIVE_TO_ID_DICT[primitive]
    primitive_id = np.array([primitive_id], dtype=np.uint8)
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
        env.resetpos[:3] = np.array([0.5854150172032604,0.06086102586621338,0.1046158263847778])
        env.resetpos[3:] = np.array([0.6299743988034661,0.6971103396175138,-0.23715609242845734,0.24683277552754387])
    elif primitive == 'insert':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=1) 
        env.resetpos[:3] = np.array([0.5,-0.18,0.33])
        env.resetpos[3:] = np.array([0.7232287667289446,0.6900075685348428,-0.010295688497306624,-0.026901768319765842])
    else:
        raise ValueError("Unrecongized primitive name")

    traj_length ={
        'grasp':75,
        'insert': 100,
        'long_horizon': 300,
        'regrasp': 60,
        'place_on_fixture': 100,
        'rotate': 200,
    }

    simulate_policy(FLAGS, env, forward_policy, processor=processor, primitive=primitive, N=100,
                    traj_length=traj_length[primitive])


if __name__ == "__main__":
    absl.app.run(main)
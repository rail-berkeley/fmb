import copy
import os
from functools import partial
import time
from pynput import keyboard
import numpy as np
import cv2
from flax.training import checkpoints
from orca.data.rlds.fmb_primitive_peg_map import FMB_PRIMITIVE_TO_ID_DICT, FMB_PRIMITIVE_LIST

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


ACT_MEAN = [0.08881065249443054, -0.029072951525449753, -0.14968131482601166, 0.040181715041399, -0.006027124356478453, 0.0047734929248690605, 0.6311022639274597]
ACT_STD = [0.2633409798145294, 0.17057746648788452, 0.2521684467792511, 0.3042782247066498, 0.1229381114244461, 0.12113021314144135, 0.4826180338859558]

should_reset = False
TRAJ_LENGTH ={
    'grasp': 75,
    'insert': 100,
    'long_horizon': 300,
    'regrasp': 70,
    'place_on_fixture': 70,
    'rotate': 70,
}
FLAGS = absl.flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("primitive", None, "Name of primitive", required=True)
flags.DEFINE_integer("peg", None, "Name of peg", required=True)
flags.DEFINE_string("wandb_run_name", None, "Name of wandb run", required=True)
flags.DEFINE_integer("obs_horizon", 1, "Observation history length")
flags.DEFINE_integer("im_size", 256, "Image size")


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


    train_state = checkpoints.restore_checkpoint(path, train_state)

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
        o=None,
        tcp_frame=False,
        reset=True,
):
    global should_reset, TRAJ_LENGTH
    observations = []
    actions = []
    path_length = 0
    last_action = np.zeros(7)

    try:
        primitive_id = int(input(f"Enter primitive id: {FMB_PRIMITIVE_TO_ID_DICT}"))
        primitive = FMB_PRIMITIVE_LIST[primitive_id]
        peg_id = int(input("Enter peg id: "))
    except:
        return dict(
            observations=observations,
            actions=actions,
        )
    max_path_length = TRAJ_LENGTH[primitive]

    if primitive == 'grasp' or primitive=='long_horizon':
        env.resetpos[:3] = np.array([0.45, 0.1, 0.22])
        env.reset_yaw=np.pi/2
        env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)
        o = env.reset(gripper=0)
        if primitive == 'grasp':
            env.grasp_mode()
    elif primitive == 'place_on_fixture' or primitive=='rotate':
        o, _, _, _ = env.step(np.array([0,0,0,0,0,0,1]))
    elif primitive == 'regrasp':
        # o, _, _, _ = env.step(np.array([0,0,0,0,0,0,0]))
        env.resetpos[:3] = np.array([0.62,0.07,0.08])
        env.resetpos[3:] = np.array([0.646681636281875,0.6413527195081079,-0.2916347574378059,0.2922648092561658])
        o = env.reset(gripper=0)
    elif primitive == 'insert':
        env.resetpos[:3] = np.array([0.5,-0.18,0.33])
        env.resetpos[3:] = np.array([0.7232287667289446,0.6900075685348428,-0.010295688497306624,-0.026901768319765842])
        o = env.reset(gripper=1)
        env.insertion_mode()
    else:
        raise ValueError("Unrecongized primitive name")

    should_reset = False


    while path_length < max_path_length:
        if should_reset:
            break
        # Get action from policy
        agent_obs = preprocess_obs_for_policy_fn(o,
            primitive_id=np.array([primitive_id], dtype=np.uint8),
            peg_id=np.array([peg_id], dtype=np.uint8)
        )
        action, agent_info = agent(agent_obs)
        action[:6] = action[:6] * env.action_space.high[:6]

        next_o, _, _, _ = env.step(action)
        observations.append(o)
        actions.append(action)
        path_length += 1
        o = next_o

    if primitive in ['grasp', 'regrasp']:
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


def simulate_policy(args, env, policy, processor, N=20):
    paths = []
    for i in range(N):
        print("Number: ", i + 1)
        paths.append(rollout(
            env,
            policy,
            preprocess_obs_for_policy_fn=processor,
        ))

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
    param_count = sum(
        np.prod(x.shape) 
        for x in jax.tree_leaves(jax.tree_map(jnp.array, train_state.params))
    )
    print(f"Number of parameters: {param_count:} ({param_count / 1e6:.2f}M)")
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

    env = gym.make("Franka-FMB-v0", hz=10, start_gripper=0) 


    simulate_policy(FLAGS, env, forward_policy, processor=general_preprocessor, N=100,)


if __name__ == "__main__":
    absl.app.run(main)

import hashlib
import json
import os
import logging  
from typing import Any, Dict, List, Optional
import cv2

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from .primitive_peg_map import tf_primitive_to_primitive_id
from .dataset import BaseDataset

def _get_splits(tfds_dataset_splits):
    """Use val split from dataset if defined, otherwise use parts of train split."""
    if "val" in tfds_dataset_splits:
        return {"train": "train", "val": "val"}
    else:
        # use last 5% of training split as validation split
        return {"train": "train[:95%]", "val": "train[95%:]"}


class RLDSDataset(BaseDataset):
    """Fast parallel tf.data.Dataset-based dataloader for RLDS datasets.

    Args:
        image_obs_key: Key used to extract image from raw dataset observation.
        tfds_data_dir: Optional. Directory to load tf_datasets from. Defaults to ~/tensorflow_datasets
    """

    def __init__(
        self,
        *args,
        image_obs_key: List[str],
        primitive_key: Optional[List[str]] = None,
        peg_keys: Optional[List[int]] = None,
        tfds_data_dir: Optional[str] = None,
        num_pegs: int = 0,
        num_primitives: int = 0,
        **kwargs,
    ):
        self._image_obs_key = image_obs_key
        self._primitive_key = primitive_key
        self._peg_keys = peg_keys
        self._tfds_data_dir = tfds_data_dir
        self._num_pegs = num_pegs
        self._num_primitives = num_primitives
        super().__init__(*args, **kwargs)

    def _construct_base_dataset(self, dataset_name: str, seed: int) -> tf.data.Dataset:
        # load raw dataset of trajectories
        # skips decoding to get list of episode steps instead of RLDS default of steps as a tf.dataset
        builder = tfds.builder(dataset_name, data_dir=self._tfds_data_dir)
        dataset = builder.as_dataset(
            split=_get_splits(builder.info.splits)["train" if self.is_train else "val"],
            decoders={"steps": tfds.decode.SkipDecoding()},
            shuffle_files=self.is_train,
        )



    
        def _filter_steps_by_primitives(episode):
            #filters out steps in one episode based on queried primitive
            filtered_steps = {}
            primitive_tensor = episode['steps']['observation']['primitive']
            match = None
            for key in self._primitive_key:
                match_ = tf.equal(primitive_tensor, key)
                if match is None:
                    match = match_
                else:
                    match = tf.logical_or(match, match_)

            for key, item in episode['steps'].items():
                if not isinstance(item, dict):
                    filtered_steps[key] = tf.boolean_mask(item, match)
                else:
                    sub_dict = {}
                    for sub_key, sub_item in item.items():
                         sub_dict[sub_key] = tf.boolean_mask(sub_item, match)
                    filtered_steps[key] = sub_dict
            filtered_episode = {"steps": filtered_steps}
            return filtered_episode

        def _filter_steps_by_peg_id(episode):
            # filter steps
            # if primitive is insertion, then filter by specific peg id
            # if primitive is not insertion, then I need all pegs
            filtered_steps = {}
            match = None
            for peg in self._peg_keys:
                match_ = tf.equal(episode['steps']['observation']['peg_id'], peg)
                if match is None:
                    match = match_
                else:
                    match = tf.logical_or(match, match_)

            for key, item in episode['steps'].items():
                if not isinstance(item, dict):
                    filtered_steps[key] = tf.boolean_mask(item, match)
                else:
                    sub_dict = {}
                    for sub_key, sub_item in item.items():
                        sub_dict[sub_key] = tf.boolean_mask(sub_item, match)
                    filtered_steps[key] = sub_dict
            filtered_episode = {"steps": filtered_steps}
            return filtered_episode

        def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            # every input feature is batched, ie has leading batch dimension
            trajectory["observation"]["tcp_vel"]=trajectory["observation"]["eef_vel"]
            trajectory["observation"]["tcp_pose"]=trajectory["observation"]["eef_pose"]
            trajectory["observation"]["tcp_force"]=trajectory["observation"]["eef_force"]
            trajectory["observation"]["tcp_torque"]=trajectory["observation"]["eef_torque"]
            trajectory["observation"]["primitive_id"] = tf_primitive_to_primitive_id(trajectory["observation"]["primitive"])

            return trajectory

        def _decode_trajectory(episode: Dict[str, Any]) -> Dict[str, Any]:
            # manually decode all features since we skipped decoding during dataset construction
            steps = episode["steps"]
            
            for key in steps:                
                if key == "observation":
                    # only decode parts of observation we need for improved data loading speed
                    for img_key in self._image_obs_key:
                        steps["observation"][img_key] = builder.info.features["steps"][
                            "observation"
                        ]["image_" + img_key].decode_batch_example(
                            steps["observation"]["image_" + img_key]
                        )
                        if "depth" in img_key:
                            steps["observation"][img_key] = tf.expand_dims(steps["observation"][img_key], axis=-1)
                    steps["observation"]["state"] = builder.info.features["steps"][
                        "observation"
                    ]["state"].decode_batch_example(
                        steps["observation"]["state"]
                    )
                else:
                    steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key])
            return steps

        def _to_transition_trajectories(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            output = {}
            output["actions"] = trajectory["action"]
            for key in self._image_obs_key:
                output[f"obs/{key}"] = trajectory["observation"][key]
            if self._num_pegs > 0:
                output["shape_vec"] = tf.one_hot(trajectory["observation"]["peg_id"]-1, self._num_pegs)
            # else:
            #     output["shape_vec"] = None
            if self._num_primitives > 0:
                output["primitive_vec"] = tf.one_hot(trajectory["observation"]["primitive_id"], self._num_primitives)
            # else:
            #     output["primitive_vec"] = None
            output["obs/tcp_pose"] = trajectory["observation"]["tcp_pose"]
            output["obs/tcp_vel"] = trajectory["observation"]["tcp_vel"]
            output["obs/tcp_force"] = trajectory["observation"]["tcp_force"]
            output["obs/tcp_torque"] = trajectory["observation"]["tcp_torque"]
            return output

        if self._peg_keys is not None:
            dataset = dataset.map(_filter_steps_by_peg_id, num_parallel_calls=tf.data.AUTOTUNE)
        if self._primitive_key is not None:
            dataset = dataset.map(_filter_steps_by_primitives, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(_decode_trajectory, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            fmb_dataset_transform,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.map(
            _to_transition_trajectories, num_parallel_calls=tf.data.AUTOTUNE
        )


        return dataset

    @staticmethod
    def _get_action_proprio_stats(dataset_builder, dataset):
        # get statistics file path --> embed unique hash that catches if dataset info changed
        data_info_hash = hashlib.sha256(
            str(dataset_builder.info).encode("utf-8")
        ).hexdigest()

        path = tf.io.gfile.join(
            dataset_builder.info.data_dir, f"action_proprio_stats_{data_info_hash}.json"
        )        
        # check if stats already exist and load, otherwise compute
        if tf.io.gfile.exists(path):
            logging.info(f"Loading existing statistics for normalization from {path}.")
            print(f"Loading existing statistics for normalization from {path}.")
            with tf.io.gfile.GFile(path, "r") as f:
                action_proprio_metadata = json.load(f)

        else:
            print("Computing action/proprio statistics for normalization...")
            actions = []
            proprios = []
            for episode in tqdm.tqdm(dataset.take(1000)):
                actions.append(episode["actions"].numpy())
                proprios.append(episode["observations"]["proprio"].numpy())
            actions = np.concatenate(actions)
            proprios = np.concatenate(proprios)
            action_proprio_metadata = {
                "action": {
                    "mean": [float(e) for e in actions.mean(0)],
                    "std": [float(e) for e in actions.std(0)],
                    "max": [float(e) for e in actions.max(0)],
                    "min": [float(e) for e in actions.min(0)],
                },
                "proprio": {
                    "mean": [float(e) for e in proprios.mean(0)],
                    "std": [float(e) for e in proprios.std(0)],
                    "max": [float(e) for e in proprios.max(0)],
                    "min": [float(e) for e in proprios.min(0)],
                },
            }
            del actions
            del proprios
            with open(path, "w") as F:
                json.dump(action_proprio_metadata, F)
        return action_proprio_metadata

    def _chunk_act_obs(self, traj):
        def combine_axes(tensor):
            rank = len(tensor.shape)
            if rank == 5:
                transposed = tf.transpose(tensor, [0, 2, 3, 1, 4])
                return tf.reshape(transposed, [tf.shape(tensor)[0], tf.shape(tensor)[2], tf.shape(tensor)[3], -1])
            elif rank == 3:
                return tf.reshape(tensor, [tf.shape(tensor)[0], -1])
            else:
                raise ValueError("Unsupported tensor rank. Expected rank 3 or 5.")

        traj_len = len(traj["actions"])
        if self.act_pred_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
            )
            # pads by repeating the last action
            chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
            traj["actions"] = combine_axes(tf.gather(traj["actions"], chunk_indices))
        if self.obs_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(-self.obs_horizon + 1, 1), [traj_len, self.obs_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.obs_horizon]
            )
            # pads by repeating the first observation
            chunk_indices = tf.maximum(chunk_indices, 0)
            for key in ['obs/tcp_pose', 'obs/tcp_vel', 'obs/tcp_force', 'obs/tcp_torque'] + [f"obs/{key}" for key in self._image_obs_key]:
                traj[key] = combine_axes(tf.gather(traj[key], chunk_indices))
        return traj

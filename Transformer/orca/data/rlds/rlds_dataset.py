import hashlib
import json
import os
import logging  
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from orca.data.dataset import BaseDataset
from orca.data.rlds.rlds_dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS

def _get_splits(tfds_dataset_splits):
    """Use val split from dataset if defined, otherwise use parts of train split."""
    if "val" in tfds_dataset_splits:
        return {"train": "train", "val": "val"}
    else:
        # use last 5% of training split as validation split
        return {"train": "train[:95%]", "val": "train[95%:]"}
        # return {"train": "train[:10]", "val": "train[11:13]"} # For debugging purposes


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
        **kwargs,
    ):
        self._image_obs_key = image_obs_key
        self._primitive_key = primitive_key
        self._peg_keys = peg_keys
        print('primitive_key', primitive_key)
        print('peg_keys', peg_keys)
        self._tfds_data_dir = tfds_data_dir
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


        def _decode_trajectory(episode: Dict[str, Any]) -> Dict[str, Any]:
            # manually decode all features since we skipped decoding during dataset construction
            steps = episode["steps"]
            
            for key in steps:                
                if key == "observation":
                    # only decode parts of observation we need for improved data loading speed
                    for img_key in self._image_obs_key:
                        steps["observation"][img_key] = builder.info.features["steps"][
                            "observation"
                        ][img_key].decode_batch_example(
                            steps["observation"][img_key]
                        )
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
            # return transition dataset in convention of Bridge dataset
            observations = {}
            for key in self._image_obs_key:
                observations[key] = tf.cast(trajectory["observation"][key], tf.float32) / 127.5 - 1.0
            observations["proprio"] = trajectory["observation"]["state"]
            observations["ee_ft"] = trajectory["observation"]["ee_ft"]
            observations["primitive_id"] = tf.cast(trajectory["observation"]["primitive_id"], tf.uint8)
            # observations["primitive"] = trajectory["observation"]["primitive"]
            observations["peg_id"] = trajectory["observation"]["peg_id"]
            observations["ee_pose"] = trajectory["observation"]["ee_pose"]
            observations["ee_vel"] = trajectory["observation"]["ee_vel"]
            return {
                "observations": {**observations},
                "next_observations": {**observations}, # FMB doesn't need next obs
                **(
                    {"language": trajectory["language_instruction"]}
                    if self.load_language
                    else {}
                ),
                "actions": trajectory["action"],
                "terminals": trajectory["is_terminal"],
                "truncates": tf.math.logical_and(
                    trajectory["is_last"],
                    tf.math.logical_not(trajectory["is_terminal"]),
                ),
            }

        if self._primitive_key is not None:
            dataset = dataset.map(_filter_steps_by_primitives, num_parallel_calls=tf.data.AUTOTUNE)
        if self._peg_keys is not None:
            dataset = dataset.map(_filter_steps_by_peg_id, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(_decode_trajectory, num_parallel_calls=tf.data.AUTOTUNE)
        if dataset_name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
            # optionally apply transform function to get canonical step representation
            dataset = dataset.map(
                RLDS_TRAJECTORY_MAP_TRANSFORMS[dataset_name],
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        dataset = dataset.map(
            _to_transition_trajectories, num_parallel_calls=tf.data.AUTOTUNE
        )

        # load or compute action metadata for normalization
        self.action_proprio_metadata = self._get_action_proprio_stats(builder, dataset)

        return dataset

    @staticmethod
    def _get_action_proprio_stats(dataset_builder, dataset):
        # get statistics file path --> embed unique hash that catches if dataset info changed
        data_info_hash = hashlib.sha256(
            str(dataset_builder.info).encode("utf-8")
        ).hexdigest()
        path = tf.io.gfile.join(
            '.', f"action_proprio_stats_{data_info_hash}.json"
        )
        # path = tf.io.gfile.join(
        #     dataset_builder.info.data_dir, f"action_proprio_stats_{data_info_hash}.json"
        # )        
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
            for episode in tqdm.tqdm(dataset.take(10000)):
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


if __name__ == "__main__":
    base_data_config = dict(
        shuffle_buffer_size=200,
        prefetch_num_batches=20,
        augment=False,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
        normalization_type="normal",
    )
    image_obs_key = "image_side_2:image_wrist_1:image_wrist_2"
    image_obs_key = image_obs_key.split(":")
    # primitive_key = "grasp:insert"
    # primitive_key = primitive_key.split(":")

    ds = RLDSDataset(
        dataset_names="fmb_dataset",
        image_obs_key=image_obs_key,
        tfds_data_dir='gs://fmb-central-1/',
        image_processor="default",
        batch_size=2,
        obs_horizon=1,
        seed=0,
        **base_data_config,
    )
    for i in range(100):
        sample = next(ds.get_iterator())
        for k, v in sample.items():
            if hasattr(v, "shape"):
                print(k, v.shape, v.max(), v.min(), v.dtype)
            else:
                for kk, vv in v.items():
                    if kk == 'primitive_id':
                        print('primitive_id', vv)
                    if kk == 'peg_id':
                        print('peg_id', vv)

import fnmatch
from typing import Iterable, Iterator, List, Optional, Union

import jax
import numpy as np
import tensorflow as tf
from absl import logging
from flax.core import FrozenDict

class BaseDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader.

    Args:
        dataset_names: Single dataset name OR list of dataset names OR list of filepath-lists.
            If more than one element is provided, the data will be sampled from each
            dataset according to "sample_weights".
        seed: Random seed.
        normalization_type: The type of normalization to apply to the actions
            and proprio.
        sample_weights: If dataset_names has multiple elements, this is a
            list of weights with which to sample from each dataset.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        prefetch_num_batches: Number of batches to prefetch.
        cache: Whether to cache the dataset in memory.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        act_pred_horizon: Number of consecutive actions that will be predicted.
        obs_horizon: Number of consecutive observations that will be conditioned on.
    """

    def __init__(
        self,
        dataset_names: Union[str, List[str], List[List[str]]],
        seed: int,
        normalization_type: Optional[str] = None,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        prefetch_num_batches: int = 5,
        cache: bool = False,
        train: bool = True,
        act_pred_horizon: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        string_fields: Optional[str] = [
            "language"
        ],
        image_processor: Optional[str] = "default",
        image_shape: Optional[List[int]] = (256, 256, 3),
        skip_unlabeled: bool = False,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to Dataset: %s", kwargs)
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(dataset_names)] * len(dataset_names)
        assert len(dataset_names) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.normalization_type = normalization_type
        self.action_proprio_metadata = (
            None  # metadata for normalization, maybe computed on the fly
        )
        self.cache = cache
        self.act_pred_horizon = act_pred_horizon
        self.obs_horizon = obs_horizon
        self.is_train = train
        self.string_fields = string_fields
        self.image_processor = image_processor
        self.image_shape = image_shape

        # construct datasets
        datasets = []
        for dataset_name in dataset_names:
            datasets.append(self._construct_tf_dataset(dataset_name, seed))

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard

        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )


        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )
        
        dataset = dataset.map(
            self._process_image_fields, num_parallel_calls=tf.data.AUTOTUNE
        )

        # always prefetch at the end of the pipeline
        dataset = dataset.prefetch(prefetch_num_batches)

        self.tf_dataset = dataset

    def _construct_tf_dataset(
        self, dataset_name: Union[str, List[str]], seed: int
    ) -> tf.data.Dataset:
        # construct base tf dataset of trajectories
        dataset = self._construct_base_dataset(dataset_name, seed)

        # maybe apply action & proprio normalization
        if self.normalization_type is not None:
            dataset = dataset.map(
                self._normalize_action_proprio, num_parallel_calls=tf.data.AUTOTUNE
            )

        # maybe chunks into snippets
        dataset = dataset.map(self._chunk_act_obs, num_parallel_calls=tf.data.AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    def _construct_base_dataset(
        self, dataset_name: Union[str, List[str]], seed: int
    ) -> tf.data.Dataset:
        """Constructs basic dataset of trajectories."""
        raise NotImplementedError("This should be implemented in child class.")

    def _normalize_action_proprio(self, traj):
        if self.action_proprio_metadata is not None:
            if self.normalization_type == "normal":
                # normalize to mean 0, std 1
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["mean"]
                ) / self.action_proprio_metadata["action"]["std"]
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["mean"]
                    ) / self.action_proprio_metadata["proprio"]["std"]
            elif self.normalization_type == "bounds":
                # normalize to [0, 1]
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["min"]
                ) / (
                    self.action_proprio_metadata["action"]["max"]
                    - self.action_proprio_metadata["action"]["min"]
                )
                # clip to [0, 1]
                traj["actions"] = tf.clip_by_value(traj["actions"], 0, 1)
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    ) / (
                        self.action_proprio_metadata["proprio"]["max"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    )
                    traj[key]["proprio"] = tf.clip_by_value(traj[key]["proprio"], 0, 1)
            else:
                raise ValueError

        return traj

    def _chunk_act_obs(self, traj):
        traj_len = len(traj["actions"])
        if self.act_pred_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
            )
            # pads by repeating the last action
            chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
            traj["action_chunks"] = tf.gather(traj["actions"], chunk_indices)
        if self.obs_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(-self.obs_horizon + 1, 1), [traj_len, self.obs_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.obs_horizon]
            )
            # pads by repeating the first observation
            chunk_indices = tf.maximum(chunk_indices, 0)
            traj["obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["observations"]
            )
            traj["next_obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["next_observations"]
            )
        return traj

    def _process_image(self, image):
        if self.image_processor == "default":
            pass
        elif self.image_processor == "clip":
            # this should be exactly the same as HF's CLIPProcessor
            # but it needs to be in tf graph or it's slow
            # for some reason we need to set this shape or it won't work
            image = tf.reshape(image, [-1, *self.image_shape])
            image.set_shape([None, *self.image_shape])
            image = tf.image.resize(image, (224, 224), method="bicubic")
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = image / 255.0
            image = (image - [0.48145466, 0.4578275, 0.40821073]) / [
                0.26862954,
                0.26130258,
                0.27577711,
            ]
            image = tf.reshape(image, [-1, self.obs_horizon, 224, 224, 3])
        return image

    def _process_image_fields(self, batch):
        for key in ["observations", "next_observations"]:
            if key in batch:
                for view in self._image_obs_key:
                    batch[key][view] = self._process_image(batch[key][view])
        return batch

    def _process_strings(self, strings):
        strings = [s.decode("utf-8") for s in strings]
        return strings

    def _process_string_fields(self, batch):
        return jax.tree_util.tree_map_with_path(
            lambda kp, x: self._process_strings(x)
            if kp[-1].key in self.string_fields
            else x,
            batch,
        )

    def get_iterator(self) -> Iterator[FrozenDict]:
        # yield FrozenDicts. this can be bypassed by using
        # `dataset.tf_dataset.as_numpy_iterator()` instead
        iterator = map(FrozenDict, self.tf_dataset.as_numpy_iterator())

        # need to tokenize language instructions here already to allow for sharding (str cannot be sharded)
        # can only apply tokenizers after conversion to numpy
        iterator = map(self._process_string_fields, iterator)
        return iterator

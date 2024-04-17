import copy
from typing import Iterator, List, Union, Optional
from flax.core import FrozenDict
import numpy as np
import tensorflow as tf

class Dataset:
    """
    Fast parallel tf.data.Dataset-based dataloader for a dataset in the
    BridgeData format. This format consists of TFRecords where each example
    is one trajectory. See `PROTO_TYPE_SPEC` below for the expected format
    for each example in more detail. See `_process_trajectory` below for
    the output format.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 4,
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        train: bool = True,
        use_transitions: bool = True, 
        image_keys: Optional[List[str]] = [],
        state_keys: Optional[List[str]] = [],
        traj_length: Optional[int] = None,
        traj_end_offset: Optional[int] = None,
        num_frame_stack: Optional[int] = 1,
        num_action_chunk: Optional[int] = 1,
        add_label: Optional[bool] = False,
        peg_key: Optional[int] = None,
        primitive_key: Optional[int] = None,
        **kwargs,

    ):

        KEYS = ['obs/side_1', 'obs/side_1_depth',
                'obs/side_2',  'obs/side_2_depth',  
                'obs/wrist_1', 'obs/wrist_1_depth', 
                'obs/wrist_2', 'obs/wrist_2_depth', 
                'obs/tcp_pose', 'obs/tcp_vel', 
                'obs/gripper_pose', 
                'obs/q', 'obs/dq', 
                'obs/tcp_force', 'obs/tcp_torque', 
                'obs/jacobian', 
                'actions', 'last_actions',
                # 'peg_id', 'primitive'
                ]
        self.add_label = add_label
        self.PROTO_TYPE_SPEC = {}
        for key in KEYS:
            if key in ['obs/side_1', 'obs/side_2', 'obs/wrist_1', 'obs/wrist_2', 'peg_id']:
                self.PROTO_TYPE_SPEC[key] = tf.uint8
            elif 'depth' in key:
                self.PROTO_TYPE_SPEC[key] = tf.uint16
            elif 'primitive' in key:
                self.PROTO_TYPE_SPEC[key] = tf.string
            else:
                self.PROTO_TYPE_SPEC[key] = tf.float32
        self.num_action_chunk = num_action_chunk
        self.num_frame_stack = num_frame_stack
        self.image_keys = image_keys
        self.state_keys = state_keys
        self.use_transitions = use_transitions
        self.traj_length = traj_length
        self.traj_end_offset = traj_end_offset
        self._peg_key = peg_key
        self._primitive_key = primitive_key
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)
        self.cache = cache
        # construct a dataset for each sub-list of paths
        datasets = []

        
        for sub_data_paths in data_paths:
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

        
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
        
        # always prefetch at the end of the pipeline
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        

        self.tf_dataset = dataset



    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """
        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = dataset.interleave(
            lambda filepath: tf.data.TFRecordDataset(filepath).map(lambda data: (filepath, data)),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

        # dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # decode serialized examples
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        if self._primitive_key:
            dataset = dataset.map(self._filter_primitive, num_parallel_calls=tf.data.AUTOTUNE)

        if self._peg_key:
            dataset = dataset.map(self._filter_peg, num_parallel_calls=tf.data.AUTOTUNE)

        if self.num_frame_stack > 1:
            dataset = dataset.map(self._stack_frames, num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.num_action_chunk > 1:
            dataset = dataset.map(self._action_chunk, num_parallel_calls=tf.data.AUTOTUNE)

        if not self.use_transitions:
            dataset = dataset.map(self._subsample_trajectory, num_parallel_calls=tf.data.AUTOTUNE)

        if self.add_label:
            dataset = dataset.map(self._add_label, num_parallel_calls=tf.data.AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        # unbatch to yield individual transition
        if self.use_transitions:
            dataset = dataset.unbatch()
        

        return dataset

    def _filter_primitive(self, traj):
        filtered_traj = {}
        match = None
        for primitive in self._primitive_key:
            match_ = tf.equal(traj['primitive'], primitive)
            if match is None:
                match = match_
            else:
                match = tf.logical_or(match, match_)
        for key, item in traj.items():
            filtered_traj[key] = tf.boolean_mask(item, match)
        return filtered_traj

    def _filter_peg(self, traj):
        filtered_traj = {}
        match = None
        for peg in self._peg_key:
            match_ = tf.equal(traj['peg_id'], peg)
            if match is None:
                match = match_
            else:
                match = tf.logical_or(match, match_)
        print(tf.shape(match)[0])
        for key, item in traj.items():
            filtered_traj[key] = tf.boolean_mask(item, match)
        return filtered_traj

    def _subsample_trajectory(self, traj):
        # sample a random starting point
        num_timesteps = tf.shape(traj['actions'])[0]
        pad = 0
        if self.traj_length is None:
            start = 0
            end = num_timesteps
        else:
            if self.traj_end_offset is None:
                self.traj_end_offset = 0

            start = tf.random.uniform(
                (), maxval=num_timesteps - self.traj_end_offset, dtype=tf.int32)
            end = tf.minimum(start + self.traj_length, num_timesteps)
            pad = tf.maximum(start + self.traj_length - num_timesteps, 0)


        def _slice_and_pad(x):
            if pad > 0:
                padding = tf.repeat(x[:1], pad, axis=0)
                padding = tf.zeros_like(padding)
                x = tf.concat([x[start:end], padding], axis=0)
            else:
                x = x[start:end]
            return x

        for key in traj.keys():
            traj[key] = tf.nest.map_structure(_slice_and_pad, traj[key])


        if pad > 0:
            traj['padding'] = tf.concat(
                [
                    tf.ones([end - start, 1], dtype=tf.float32),
                    tf.zeros([pad, 1], dtype=tf.float32)
                ],
                axis=0,
            )
        else:
            traj['padding'] = tf.ones([end - start, 1], dtype=tf.float32)

        return traj

        

    def _decode_example(self, filename, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }

        
        # restructure the dictionary into the downstream format

        output = {}

        for key in self.image_keys + self.state_keys + ['actions']:
            if "depth" in key:
                output[key] = tf.expand_dims(parsed_tensors[key], axis=-1)
            else:
                output[key] = parsed_tensors[key]

        return output
        
    def _stack_frames(self, traj):
        def _stack_frames_helper(img):
            frames = [img]
            previous=img
            for _ in range(self.num_frame_stack - 1):
                previous = tf.concat([previous[:1], previous[:-1]], axis=0)
                frames.append(previous)
            x = tf.concat(frames[::-1], axis=-1)
            return x
        
        for key in self.image_keys:
            traj[key] = tf.nest.map_structure(_stack_frames_helper, traj[key])
        return traj
    
    def _action_chunk(self, traj):
        actions = [traj['actions']]
        next_actions = traj['actions']
        for _ in range(self.num_action_chunk - 1):
            next_actions = tf.concat([next_actions[1:], next_actions[-1:]], axis=0)
            actions.append(next_actions)
        traj['actions'] = tf.concat(actions, axis=-1)
        return traj
    
    def _add_label(self, traj):
        traj_len = tf.shape(traj['actions'])[0]
        traj['label'] = tf.concat([tf.zeros((traj_len-8,), dtype=tf.int32), tf.ones((8,), dtype=tf.int32)], axis=0)
        return traj
    
    def get_iterator(self) -> Iterator[FrozenDict]:
        # yield FrozenDicts. this can be bypassed by using
        # `dataset.tf_dataset.as_numpy_iterator()` instead
        iterator = map(FrozenDict, self.tf_dataset.as_numpy_iterator())
        return iterator



def get_data(iterator):
    data = next(iterator).unfreeze()
    return copy.deepcopy(data)



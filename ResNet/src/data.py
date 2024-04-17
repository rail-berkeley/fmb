from copy import deepcopy
import numpy as np
import torch
from torchvision.transforms import RandAugment, TrivialAugmentWide, AugMix, RandomCrop, ColorJitter
from scipy.spatial.transform import Rotation as R
from flax.core import FrozenDict


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    length = batch[list(batch.keys())[0]].shape[0]
    indices = np.random.randint(length, size=size)
    return index_batch(batch, indices)


def parition_batch_train_test(batch, train_ratio, random=False):
    length = batch[list(batch.keys())[0]].shape[0]
    if random:
        train_indices = np.random.rand(length) < train_ratio
    else:
        train_indices = np.linspace(0, 1, length) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch[list(batch.keys())[0]].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def subset_dataset(dataset, percent):
    length = dataset['actions'].shape[0]
    n = int(percent * length)
    idx = np.random.choice(length, size=n, replace=False)
    for k in dataset.keys():
        dataset[k] = dataset[k][idx]
    return dataset


def split_batch_pmap(batch, num_devices): # ADDED
    for key in batch.keys():
        batch[key] = batch[key].reshape((num_devices, -1, *batch[key].shape[1:]))
    return batch


def flatten_dataset(dataset):
    new_data = dict()
    image_observation_keys = list(dataset['observations'][0]['image_observation'].keys())
    state_observation_keys = list(dataset['observations'][0]['state_observation'].keys())
    state_observation_keys.append('jacobian')
    for key in image_observation_keys + state_observation_keys:
        new_data[f'obs/{key}'] = []

    new_data['actions'] = dataset['actions']
    new_data['last_actions'] = np.concatenate([np.zeros((1,7)), dataset['actions'][:-1]], axis=0)
    # Flatten obs keys into dict of lists
    for transition in dataset['observations']:
        for key, value in transition['image_observation'].items():
            new_data[f'obs/{key}'].append(value)
        for key, value in transition['state_observation'].items():
            new_data[f'obs/{key}'].append(value)
    for key in new_data.keys():
        if 'obs' in key:
            new_data[key] = np.array(new_data[key])
    if 'obs/jacobian' in new_data.keys():
        del new_data['obs/jacobian']
    return new_data


def preprocess_image_classifier_dataset(dataset, image_keys):
    dataset = deepcopy(dataset)
    for key in image_keys:
        if 'depth' in key:
            dataset[f'obs/{key}'] = np.expand_dims(dataset[f'obs/{key}'].astype(np.float32), axis=-1) / 65535.0
        else:
            dataset[f'obs/{key}'] = dataset[f'obs/{key}'].astype(np.float32) / 255.0

    dataset['class'] = np.array(dataset['class'])

    return dataset


def preprocess_robot_dataset(dataset, clip_action, image_keys, state_keys, last_action, train_gripper=False, tcp_frame=False):
    dataset = deepcopy(dataset)
    if isinstance(dataset, FrozenDict):
        dataset = dataset.unfreeze()
    for key in image_keys:
        if 'depth' in key:
            dataset[f'obs/{key}'] = dataset[f'obs/{key}'].astype(np.float32)/ 65535.0
        else:
            dataset[f'obs/{key}'] = dataset[f'obs/{key}'].astype(np.float32) / 255.0

    if tcp_frame:
        dataset = rotate_action_frame(dataset)

    dataset['actions'][..., :-1] = np.clip(dataset['actions'][..., :-1], -clip_action, clip_action).astype(np.float32)
    if last_action:
        dataset['last_actions'][..., :-1] = np.clip(dataset['last_actions'][..., :-1], -clip_action, clip_action).astype(np.float32)
    if train_gripper: 
        dataset['actions'][..., -1] = np.clip(dataset['actions'][..., -1], 1-clip_action, clip_action).astype(np.float32)
        if last_action:
            dataset['last_actions'][..., -1] = np.clip(dataset['last_actions'][..., -1], 1-clip_action, clip_action).astype(np.float32)
    else:
        dataset['actions'] = dataset['actions'][..., :-1]
        if last_action:
            dataset['last_actions'] = dataset['last_actions'][..., :-1]

    if 'rewards' in dataset:
        if type(dataset['rewards']) == list:
            del dataset['rewards']
        else:
            dataset['rewards'] = dataset['rewards'].astype(np.float32)
    
    dataset['state'] = np.concatenate([dataset[f'obs/{key}'] for key in state_keys], axis=1)
    if last_action:
        dataset['state'] = np.concatenate((dataset['state'], dataset['last_actions']), axis=1).astype(np.float32)
    return dataset


def rotate_action_frame(dataset):
    dataset = deepcopy(dataset)
    last_actions = dataset['last_actions'][..., :-1]
    states = dataset['obs/tcp_pose']
    actions = dataset['actions'][..., :-1]
    velocities = dataset['obs/tcp_vel']
    wrenchs = np.concatenate((dataset['obs/tcp_force'], dataset['obs/tcp_torque']), axis=1)
    for i in range(states.shape[0]):
        state = states[i]
        action = actions[i]
        velocity = velocities[i]
        wrench = wrenchs[i]
        last_action = last_actions[i]

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

        Adj_inv = np.linalg.inv(adjoint)
    
        #Action in base frame
        action = Adj_inv @ action
        actions[i] = action

        #:Last action in base frame
        last_action = Adj_inv @ last_action
        actions[i] = last_action

        #Velocity in base frame
        velocity = Adj_inv @ velocity
        velocities[i] = velocity

        #Force in base frame
        wrench = Adj_inv @ wrench
        wrenchs[i] = wrench
    dataset['actions'][..., :-1] = actions
    dataset['last_actions'][..., :-1] = last_actions
    dataset['obs/tcp_vel'] = velocities
    dataset['obs/tcp_force'] = wrenchs[:, :3]
    dataset['obs/tcp_torque'] = wrenchs[:, 3:]
    return dataset


def get_data_augmentation(augmentation):
    if augmentation == 'none':
        return None
    elif augmentation == 'rand':
        return torch.jit.script(RandAugment())
    elif augmentation == 'trivial':
        return torch.jit.script(TrivialAugmentWide())
    elif augmentation == 'augmix':
        return torch.jit.script(AugMix())
    elif augmentation == 'color':
        return torch.jit.script(ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    elif augmentation == 'crop':
        return torch.jit.script(RandomCrop(128, padding=(4,)))
    else:
        raise ValueError('Unsupported augmentation type!')


def augment_images(augmentation, image):
    if augmentation is None:
        return image

    # Switch to channel first
    image = np.clip(image, 0., 1.)
    image = np.transpose((image * 255.0).astype(np.ubyte), (0, 3, 1, 2))
    with torch.no_grad():
        image = torch.from_numpy(image)
        image = augmentation(image)
        image = image.numpy()

    # Switch to channel last
    image = np.transpose(image, (0, 2, 3, 1)).astype(np.float32) / 255.0
    return image


def augment_batch(augmentation, batch, keys):
    batch = deepcopy(batch)
    for key in keys:
        batch[f'obs/{key}'] = augment_images(augmentation, batch[f'obs/{key}'])
    return batch

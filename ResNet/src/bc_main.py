from functools import partial
import pprint
import re
import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
import optax
import absl.app
import absl.flags
import sys
from .jax_utils import JaxRNG, next_rng, named_tree_map
from .model import TanhGaussianResNetPolicy, TanhGaussianResNetMixedPolicy
from .data import (
    subsample_batch, preprocess_robot_dataset,
    augment_batch, get_data_augmentation, split_batch_pmap
)
from .utils import (
    define_flags_with_default, set_random_seed,
    get_user_flags, WandBLogger, average_metrics
)

from .rlds.rlds_dataset import RLDSDataset

FLAGS_DEF = define_flags_with_default(
    dataset_path='',
    dataset_name='fmb_dataset',
    peg="",
    primitive="",
    dataset_image_keys='side_image',
    state_keys='tcp_pose',
    tcp_frame=False,
    last_action=True,
    num_pegs=0,
    num_primitives=0,
    num_frame_stack=1,
    num_action_chunk=1,
    
    resnet_type='ResNet18',
    image_augmentation='none',
    clip_action=0.99,
    train_gripper=True,
    train_mse=False,
    
    seed=42,
    train_ratio=0.9,
    batch_size=128,
    total_steps=10000,
    lr=1e-4,
    lr_warmup_steps=0,
    weight_decay=0.05,
    clip_gradient=1e9,
    log_freq=50,
    save_freq=100000,
    eval_batches=20,
    eval_freq=200,
    save_model=True,
    policy=TanhGaussianResNetPolicy.get_default_config(),
    logger=WandBLogger.get_default_config(),
    device='gpu',
    cache=False,
)
FLAGS = absl.flags.FLAGS

device_list = jax.devices()
num_devices = len(device_list)

base_data_config = dict(
    shuffle_buffer_size=5000,
    prefetch_num_batches=50,
)

def main(argv):
    assert FLAGS.dataset_path != ''
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logger, variant=variant)
    if FLAGS.device=='gpu':
        save_func = partial(wandb_logger.save_pickle)
    elif FLAGS.device=='tpu':
        from jax_smi import initialise_tracking
        initialise_tracking()
        save_func = partial(wandb_logger.tpu_save_pickle, filepath=f'{FLAGS.logger.output_dir}/{FLAGS.logger.project}/{wandb_logger.experiment_id}')

    set_random_seed(FLAGS.seed)

    dataset_path = FLAGS.dataset_path.split(';')
    image_keys = FLAGS.dataset_image_keys.split(':')
    state_keys = FLAGS.state_keys.split(':')    
    dataset_img_keys = [f'obs/{key}' for key in image_keys]
    dataset_state_keys = [f'obs/{key}' for key in state_keys]

    print(f"peg_keys={[int(i) for i in FLAGS.peg.split(':')] if FLAGS.peg != '' else None}")
    print(f"primitive_keys={FLAGS.primitive.split(':') if FLAGS.primitive != '' else None}")
    train_dataset = RLDSDataset(
        dataset_names=FLAGS.dataset_name.split(';'),
        tfds_data_dir=FLAGS.dataset_path,
        image_obs_key=image_keys,
        image_processor="default",
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        obs_horizon=FLAGS.num_frame_stack,
        act_pred_horizon=FLAGS.num_action_chunk,
        num_pegs=FLAGS.num_pegs,
        num_primitives=FLAGS.num_primitives,
        primitive_key=FLAGS.primitive.split(':') if FLAGS.primitive != '' else None,
        peg_keys=[int(i) for i in FLAGS.peg.split(':')] if FLAGS.peg != '' else None,
        cache=FLAGS.cache,
        **base_data_config
    )
    train_loader_iterator = train_dataset.get_iterator()
    val_dataset = RLDSDataset(
        dataset_names=FLAGS.dataset_name.split(';'),
        tfds_data_dir=FLAGS.dataset_path,
        image_obs_key=image_keys,
        image_processor="default",
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        obs_horizon=FLAGS.num_frame_stack,
        act_pred_horizon=FLAGS.num_action_chunk,
        train=False,
        num_pegs=FLAGS.num_pegs,
        num_primitives=FLAGS.num_primitives,
        primitive_key=FLAGS.primitive.split(':') if FLAGS.primitive != '' else None,
        peg_keys=[int(i) for i in FLAGS.peg.split(':')] if FLAGS.peg != '' else None,
        cache=FLAGS.cache,
        **base_data_config
    )
    batch = next(train_loader_iterator)
    batch = preprocess_robot_dataset(batch, 
        FLAGS.clip_action, 
        image_keys, 
        state_keys, 
        FLAGS.last_action, 
        FLAGS.train_gripper, 
        FLAGS.tcp_frame
    )

    FLAGS.policy['resnet_type'] = tuple([(FLAGS.resnet_type+'Depth' if 'depth' in key else FLAGS.resnet_type)   for key in image_keys])
    if FLAGS.train_gripper:
        policy = TanhGaussianResNetMixedPolicy(
            output_dim=batch['actions'].shape[-1],
            config_updates=FLAGS.policy,
        )
    else:
        policy = TanhGaussianResNetPolicy(
            output_dim=batch['actions'].shape[-1],
            config_updates=FLAGS.policy,
        )

    params = policy.init(
        state=batch['state'][:5, ...],
        images=[batch[f'obs/{key}'][:5, ...] for key in image_keys],
        shape_vec=batch['shape_vec'][:5, ...] if 'shape_vec' in batch.keys() else None,
        primitive_vec=batch['primitive_vec'][:5, ...] if 'primitive_vec' in batch.keys() else None,
        rngs=next_rng(policy.rng_keys())
    )

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.lr,
        warmup_steps=FLAGS.lr_warmup_steps,
        decay_steps=FLAGS.total_steps,
        end_value=0.0,
    )
    
    if FLAGS.train_gripper:
        def weight_decay_mask(params):
            def decay(name, _):
                for rule in TanhGaussianResNetMixedPolicy.get_weight_decay_exclusions():
                    if re.search(rule, name) is not None:
                        return False
                return True
            return named_tree_map(decay, params, sep='/')
    else:
        def weight_decay_mask(params):
            def decay(name, _):
                for rule in TanhGaussianResNetPolicy.get_weight_decay_exclusions():
                    if re.search(rule, name) is not None:
                        return False
                return True
            return named_tree_map(decay, params, sep='/')

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=FLAGS.weight_decay,
            mask=weight_decay_mask
        )
    )
    train_state = TrainState.create(params=params, tx=optimizer, apply_fn=None)
    train_state = flax.jax_utils.replicate(train_state)

    @partial(jax.pmap, axis_name='num_devices', out_axes=(0, None), devices=device_list) # ADDED
    def train_step(rng, train_state, state, action, images, shape_vec=None, primitive_vec=None):
        rng_generator = JaxRNG(rng)
        def loss_fn(params):
            if FLAGS.train_gripper:
                log_probs, mean, gripper = policy.apply(
                    params, state, action, images, 
                    shape_vec=shape_vec,
                    primitive_vec=primitive_vec,
                    return_mean=True,
                    method=policy.log_prob,
                    rngs=rng_generator(policy.rng_keys())
                )
                robot_action = action.reshape((action.shape[0], -1, 7))[..., :-1].reshape(mean.shape)
                gripper_actions = action[..., 6::7]
                gripper_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(gripper, gripper_actions))
                loss = -jnp.mean(log_probs)
                mse = jnp.mean(jnp.sum(jnp.square(mean - robot_action), axis=-1))
                gripper_accuracy = jnp.mean(jnp.isclose((jax.nn.sigmoid(gripper)>0.5), gripper_actions, atol=0.1))
                if FLAGS.train_mse:
                    return mse + gripper_loss, (mse, gripper_accuracy, -jnp.mean(log_probs), gripper_loss)
                else:
                    return loss + gripper_loss, (mse, gripper_accuracy, -jnp.mean(log_probs), gripper_loss)
            else:
                log_probs, mean = policy.apply(
                    params, state, action, images, 
                    shape_vec=shape_vec,
                    primitive_vec=primitive_vec,
                    return_mean=True,
                    method=policy.log_prob,
                    rngs=rng_generator(policy.rng_keys())
                )
                loss = -jnp.mean(log_probs)
                mse = jnp.mean(jnp.sum(jnp.square(mean - action), axis=-1))
                if FLAGS.train_mse:
                    return mse, mse
                else:
                    return loss, mse

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        if FLAGS.train_gripper:
            (loss, (mse, gripper_accuracy, log_probs, gripper_loss)), grads = grad_fn(train_state.params)
        else:
            (loss, mse), grads = grad_fn(train_state.params)
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        train_state = train_state.apply_gradients(grads=grads)
        if FLAGS.train_gripper:
            metrics = dict(
                loss=loss,
                mse=mse,
                gripper_accuracy=gripper_accuracy,
                log_probs=log_probs,
                gripper_loss=gripper_loss,
                learning_rate=learning_rate(train_state.step),
            )
        else:
            metrics = dict(
                loss=loss,
                mse=mse,
                learning_rate=learning_rate(train_state.step),
            )
        metrics = {k: jax.lax.pmean(metrics[k], axis_name='num_devices') for k in metrics.keys()}
        return train_state, metrics

    @partial(jax.pmap, axis_name='num_devices', out_axes=(None), devices=device_list) # ADDED
    def eval_step(rng, train_state, state, action, images, shape_vec=None, primitive_vec=None):
        rng_generator = JaxRNG(rng)
        def loss_fn(params):
            if FLAGS.train_gripper:
                log_probs, mean, gripper = policy.apply(
                    params, state, action, images, 
                    shape_vec=shape_vec,
                    primitive_vec=primitive_vec,
                    return_mean=True,
                    method=policy.log_prob,
                    rngs=rng_generator(policy.rng_keys())
                )
                robot_action = action.reshape((action.shape[0], -1, 7))[..., :-1].reshape(mean.shape)
                gripper_actions = action[..., 6::7]
                gripper_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(gripper, gripper_actions))
                loss = -jnp.mean(log_probs)
                mse = jnp.mean(jnp.sum(jnp.square(mean - robot_action), axis=-1))
                gripper_accuracy = jnp.mean(jnp.isclose((jax.nn.sigmoid(gripper)>0.5), gripper_actions, atol=0.1))
                if FLAGS.train_mse:
                    return mse + gripper_loss, (mse, gripper_accuracy, -jnp.mean(log_probs), gripper_loss)
                else:
                    return loss + gripper_loss, (mse, gripper_accuracy, -jnp.mean(log_probs), gripper_loss)
            else:
                log_probs, mean = policy.apply(
                    params, state, action, images,
                    shape_vec=shape_vec,
                    primitive_vec=primitive_vec,
                    return_mean=True,
                    method=policy.log_prob,
                    rngs=rng_generator(policy.rng_keys())
                )
                loss = -jnp.mean(log_probs)
                mse = jnp.mean(jnp.sum(jnp.square(mean - action), axis=-1))
                if FLAGS.train_mse:
                    return mse, mse
                else:
                    return loss, mse
        if FLAGS.train_gripper:
            loss, (mse, gripper_accuracy, log_probs, gripper_loss) = loss_fn(train_state.params)
        else:
            loss, mse = loss_fn(train_state.params)
        if FLAGS.train_gripper:
            metrics = dict(
                eval_loss=loss,
                eval_mse=mse,
                eval_gripper_accuracy=gripper_accuracy,
                eval_log_probs=log_probs,
                eval_gripper_loss=gripper_loss,
            )
        else:
            metrics = dict(
                eval_loss=loss,
                eval_mse=mse,
            )
        metrics = {k: jax.lax.pmean(metrics[k], axis_name='num_devices') for k in metrics.keys()}
        return metrics

    augmentation = get_data_augmentation(FLAGS.image_augmentation)
    rng = next_rng()

    best_loss, best_log_probs, best_mse, best_train_loss, best_gripper_accuracy, best_gripper_loss = float('inf'), float('inf'), float('inf'), float('inf'), -float('inf'), float('inf')
    best_loss_model, best_log_probs_model, best_mse_model, best_train_loss_model  = None, None, None, None

    for step in range(FLAGS.total_steps+1):
        batch = next(train_loader_iterator)
        batch = preprocess_robot_dataset(batch, FLAGS.clip_action, image_keys, state_keys, FLAGS.last_action, FLAGS.train_gripper, FLAGS.tcp_frame)
        batch = augment_batch(augmentation, batch, image_keys)
        batch = split_batch_pmap(batch, num_devices)
        rng = next_rng()
        rng = jax.random.split(rng, num=num_devices)
        rngs = jax.pmap(lambda x: x)(rng)
        train_state, metrics = train_step(
            rngs, train_state, batch['state'], batch['actions'],
            [batch[f'obs/{key}'] for key in image_keys],
            shape_vec=batch['shape_vec'] if 'shape_vec' in batch.keys() else None,
            primitive_vec=batch['primitive_vec'] if 'primitive_vec' in batch.keys() else None,
        )
        metrics['step'] = step

        if step % FLAGS.log_freq == 0:
            if metrics['loss'] < best_train_loss:
                best_train_loss = metrics['loss']
                best_train_loss_model =  jax.device_get(flax.jax_utils.unreplicate(train_state)).params
            metrics['best_train_loss'] = best_train_loss
            wandb_logger.log(metrics)
            pprint.pprint(metrics)

        if step % FLAGS.eval_freq == 0:
            eval_metrics = []
            for _ in range(FLAGS.eval_batches):
                batch = next(val_dataset.get_iterator())
                batch = preprocess_robot_dataset(batch, FLAGS.clip_action, image_keys, state_keys, FLAGS.last_action, FLAGS.train_gripper, FLAGS.tcp_frame)   
                batch = split_batch_pmap(batch, num_devices)
                rng = next_rng()
                rng = jax.random.split(rng, num=num_devices)
                rngs = jax.pmap(lambda x: x)(rng)
                metrics = eval_step(
                    rngs, train_state, batch['state'], batch['actions'],
                    [batch[f'obs/{key}'] for key in image_keys],
                    shape_vec=batch['shape_vec'] if 'shape_vec' in batch.keys() else None,
                    primitive_vec=batch['primitive_vec'] if 'primitive_vec' in batch.keys() else None,
                )
                eval_metrics.append(metrics)
            eval_metrics = average_metrics(jax.device_get(eval_metrics))
            eval_metrics['step'] = step

            if eval_metrics['eval_loss'] < best_loss:
                best_loss = eval_metrics['eval_loss']
                best_loss_model = jax.device_get(flax.jax_utils.unreplicate(train_state)).params

            if eval_metrics['eval_mse'] < best_mse:
                best_mse = eval_metrics['eval_mse']
                best_mse_model = jax.device_get(flax.jax_utils.unreplicate(train_state)).params

            if FLAGS.train_gripper:
                if eval_metrics['eval_log_probs'] < best_log_probs:
                    best_log_probs = eval_metrics['eval_loss']
                    best_log_probs_model = jax.device_get(flax.jax_utils.unreplicate(train_state)).params

                if eval_metrics['eval_gripper_loss'] < best_gripper_loss:
                    best_gripper_loss = eval_metrics['eval_gripper_loss']

                if eval_metrics['eval_gripper_accuracy'] > best_gripper_accuracy:
                    best_gripper_accuracy = eval_metrics['eval_gripper_accuracy']

            eval_metrics['best_loss'] = best_loss
            eval_metrics['best_mse'] = best_mse
            if FLAGS.train_gripper:
                eval_metrics['best_log_probs'] = best_log_probs
                eval_metrics['best_gripper_loss'] = best_gripper_loss
                eval_metrics['best_gripper_accuracy'] = best_gripper_accuracy

            wandb_logger.log(eval_metrics)
            pprint.pprint(eval_metrics)
            if FLAGS.save_model:
                if not FLAGS.train_gripper:
                    save_data = {
                        'variant': variant,
                        'step': step,
                        'train_state': jax.device_get(flax.jax_utils.unreplicate(train_state)).params,
                        'best_mse_model': best_mse_model,
                        'best_train_loss_model': best_train_loss_model,
                    }
                else:
                    save_data = {
                        'variant': variant,
                        'step': step,
                        'train_state': jax.device_get(flax.jax_utils.unreplicate(train_state)).params,
                        'best_mse_model': best_mse_model,
                        'best_train_loss_model': best_train_loss_model,
                    }
                
                try:
                    save_func(obj=save_data, filename=f'model.pkl')
                except KeyboardInterrupt:
                    print("ATTEMPTING TO SAVE MODEL. DO NOT KILL")
                    save_func(obj=save_data, filename=f'model.pkl')
                    sys.exit()
        if FLAGS.save_model and step % FLAGS.save_freq == 0 and step > 0:
            if not FLAGS.train_gripper:
                save_data = {
                    'variant': variant,
                    'step': step,
                    'train_state': jax.device_get(flax.jax_utils.unreplicate(train_state)).params,
                    'best_mse_model': best_mse_model,
                    'best_train_loss_model': best_train_loss_model,
                }
            else:
                save_data = {
                    'variant': variant,
                    'step': step,
                    'train_state': jax.device_get(flax.jax_utils.unreplicate(train_state)).params,
                    'best_mse_model': best_mse_model,
                    'best_train_loss_model': best_train_loss_model,
                }
            try:
                save_func(obj=save_data, filename=f'model_{step}.pkl')
            except KeyboardInterrupt:
                print("ATTEMPTING TO SAVE MODEL. DO NOT KILL")
                save_func(obj=save_data, filename=f'model_{step}.pkl')
                sys.exit()

if __name__ == '__main__':
    absl.app.run(main)
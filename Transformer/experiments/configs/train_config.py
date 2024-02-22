from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from copy import deepcopy


NUM_TOKEN_PER_STATE = 2
NUM_TOKEN_PER_TASK = 4
VOCAB_SIZE = 256
def update_config(config, **kwargs):
    new_config = deepcopy(config)
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) or isinstance(config[key], ConfigDict):
                new_config[key] = update_config(config[key], **value)
            else:
                new_config[key] = value
        else:
            new_config[key] = value
    return ConfigDict(new_config)


def get_config(config_string):
    base_wandb_config = dict(
        project="fmb_orca_ijrr",
        group=placeholder(str),
        entity=placeholder(str),
    )

    base_peg_config = dict(
        batch_size=2,
        num_steps=int(2e5),
        log_interval=100,
        eval_interval=int(2e5),
        save_interval=20000,
        save_dir="/media/nvmep3p/fmb2/orca_checkpoints",
        data_path="/media/nvmep3p_2/rlds",
        resume_path=placeholder(str),
        seed=42,
        pretrained_weights=[],
        wandb=base_wandb_config,
        prefetch_num_batches=100,
    )

    base_board_config = dict(
        batch_size=2,
        num_steps=int(2e5),
        log_interval=100,
        eval_interval=int(2e5),
        save_interval=50000,
        save_dir="/media/nvmep3p/fmb2/orca_checkpoints",
        data_path="/media/nvmep3p_2/rlds",
        dataset_name="fmb_dataset",
        resume_path=placeholder(str),
        seed=42,
        pretrained_weights=[],
        wandb=base_wandb_config,
        prefetch_num_batches=100,
        cache=True
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
        shuffle_buffer_size=5000,
        prefetch_num_batches=20,
        augment=False,
        augment_next_obs_goal_differently=False,
        normalization_type=normalization_type,
    )

    base_optimizer_config = dict(
        learning_rate=1e-4,
        warmup_steps=2000,
        decay_steps=int(2e6),
    )

    base_model_config = dict(
        policy_kwargs=dict(
            num_layers=3,
            mlp_dim=512,
            vocab_size=256,
            num_heads=4,
            dropout_rate=0.1,
            normalization_type=normalization_type,
        ),
    )

    base_model_config_smaller = dict(
        policy_kwargs=dict(
            num_layers=2,
            mlp_dim=256,
            vocab_size=256,
            num_heads=4,
            dropout_rate=0.1,
            normalization_type=normalization_type,
        ),
    )

    board_config = dict(
        mask_peg_id=False,
        num_peg=6,
        num_primitive=6,
    )

    peg_config = dict(
        mask_peg_id=True,
        num_peg=11,
        num_primitive=6,
    )

    grasp_config = dict(
        primitive_key=["grasp"],
    )
    insert_config = dict(
        primitive_key=["insert"],
        peg_keys=[1,4,6,8,9], # per primitive condition, only insert condition on peg
    )
    regrasp_config = dict(
        primitive_key=["regrasp"],
    )
    place_config = dict(
        primitive_key=["place_on_fixture"],
    )
    rotate_config = dict(
        primitive_key=["rotate"],
    )

    base_encoder_kwargs = dict(
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            add_spatial_coordinates=True,
            act="swish",
        ),
        use_token_learner=True,
        conditioning_type="none",
    )

    film_encoder_kwargs = dict(
        encoder="resnetv1-34-bridge-film",
        encoder_kwargs=dict(
            add_spatial_coordinates=True,
            act="swish",
        ),
        use_token_learner=True,
        conditioning_type="film",
    )

    base_state_encoder_kwargs = dict(
        mlp="mlp-3",
        mlp_kwargs=dict(
            arch="512-512",
            output_dim=512 * NUM_TOKEN_PER_STATE,
        ),
        num_token_per_state=NUM_TOKEN_PER_STATE,
    )

    base_task_one_hot_encoder_kwargs = dict(
        mlp="mlp-3",
        mlp_kwargs=dict(
            arch="512-512",
            output_dim=512 * NUM_TOKEN_PER_TASK,
        ),
        num_primitive_tokens=NUM_TOKEN_PER_TASK,
        num_peg_tokens=NUM_TOKEN_PER_TASK
    )

    unified_obs_encoder_kwargs = dict(
        image_tokenizer_kwargs={**base_encoder_kwargs, **peg_config},
        state_tokenizer_kwargs=base_state_encoder_kwargs,
    )

    unified_obs_encoder_film_kwargs = dict(
        image_tokenizer_kwargs={**film_encoder_kwargs, **peg_config},
        state_tokenizer_kwargs=base_state_encoder_kwargs,
    )

    unified_obs_encoder_board_kwargs = dict(
        image_tokenizer_kwargs={**base_encoder_kwargs, **board_config},
        state_tokenizer_kwargs=base_state_encoder_kwargs,
    )

    unified_obs_encoder_film_board_kwargs = dict(
        image_tokenizer_kwargs={**film_encoder_kwargs, **board_config},
        state_tokenizer_kwargs=base_state_encoder_kwargs,
    )

    possible_structures = {
        "transformer_bc_film": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_film_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "fmb-unified-task-tokenizer": {**base_task_one_hot_encoder_kwargs, **peg_config},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_peg_config,
            )
        ),
        "transformer_bc_film_board": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-unmasked-obs-tokenizer": {**unified_obs_encoder_film_board_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "fmb-unified-task-tokenizer": {**base_task_one_hot_encoder_kwargs, **board_config},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'peg_keys': [1,2,3,4]},
                **base_board_config,
            )
        ),

        "transformer_bc_dummy_board": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_board_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'peg_keys': [1,2,3,4]},
                **base_board_config,
            )
        ),

        "transformer_bc_dummy_grasp": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, **grasp_config},
                **base_peg_config,
            )
        ),

        "transformer_bc_dummy_insert": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, **insert_config},
                **base_peg_config,
            )
        ),

        "transformer_bc_dummy_regrasp": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, **regrasp_config},
                **base_peg_config,
            )
        ),

        "transformer_bc_dummy_place": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, **place_config},
                **base_peg_config,
            )
        ),

        "transformer_bc_dummy_rotate": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, **rotate_config},
                **base_peg_config,
            )
        ),

        "transformer_bc_dummy_flat_4": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'cache': True},
                **base_peg_config,
                dataset_name="fmb_peg4_dataset",
            )
        ),

        "transformer_bc_dummy_flat_6": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'cache': True},
                **base_peg_config,
                dataset_name="fmb_peg6_dataset",
            )
        ),

        "transformer_bc_dummy_flat_8": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "dummy-task-tokenizer": {},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'cache': True},
                dataset_name="fmb_peg8_dataset",
            )
        ),

        "transformer_bc_cond_4": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "fmb-unified-task-tokenizer": {**base_task_one_hot_encoder_kwargs, **peg_config},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'cache': True},
                **base_peg_config,
                dataset_name="fmb_peg4_dataset",
            )
        ),

        "transformer_bc_cond_6": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "fmb-unified-task-tokenizer": {**base_task_one_hot_encoder_kwargs, **peg_config},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'cache': True},
                **base_peg_config,
                dataset_name="fmb_peg6_dataset",
            )
        ),

        "transformer_bc_cond_8": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "fmb-unified-obs-tokenizer": {**unified_obs_encoder_kwargs},
                    },
                    task_tokenizer_kwargs={
                        "fmb-unified-task-tokenizer": {**base_task_one_hot_encoder_kwargs, **peg_config},
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={**base_data_config, 'cache': True},
                **base_peg_config,
                dataset_name="fmb_peg8_dataset",
            )
        ),

    }

    return possible_structures[config_string]
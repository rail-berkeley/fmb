# Transformer Policy
## Environment
```
conda create -n fmb_transformer python=3.10
conda activate fmb_transformer
pip install -e .
pip install -r requirements.txt
```
For GPU:
```
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

## Training
```bash
bash ./experiments/scripts/train_rlds_fmb.sh
```

### Config File
The Transformer-based policy training code utilizes a config file for all the experiment options at:
`./experiments/configs/train_config.py`.

Inside the config, there are multiple experiment configurations stored in the dictionary named `possible_structures`. Many experiements share components of the config. Below are description of some useful config keys.

| Experiment Config | Descriptions |
| --- | --- |
| `policy_kwargs` | Network size |
| `observation_tokenizer_kwargs` | Config for the image and state tokenizer. This includes the image encoder type (`ResNet` or `FiLM ResNet`) and specifying the size of the one-hot conditing vector based on the number of objects and primitives to learn. |
| `task_tokenizer_kwargs` | Config for the conditioning the entire policy on object and primitive ID. `dummy-task-tokenizer` is used to skip the conditioning. `fmb-unified-task-tokenizer` is used to condition on either object ID, primitive ID, or both; the size of the conditioning vector is also specified. |
| `optimizer` | Optimizer config like learning rate and learning rate schedule.  |
| `dataset_kwargs` | Config related to the dataset and dataloading, such as the shuffle buffer size, and keys for filtering data baserd on specific object ID and primitive names. |
| `save_dir` | Path to save the learned checkpoints to |
| `data_path` | Path of the dataset |
| `dataset_name` | Name of the dataset to train on |

## Policy Rollout
To evaluate a policy on a single primitive
```bash
bash ./scripts/transformer_rollout.sh
```
To evaluate a policy on multiple primitives
```bash
bash ./scripts/transformer_seqrollout.sh
```

|Flags|Description |
| --- |---|
|checkpoint_path| Path to load the trained checkpoint from. |
| wandb_run_name | The key of the saved models to evaluate. Refer to bc_main.py to see the list of models that are saved. |
| act_mean | The mean of the action in the training dataset used to normalize the actions. |
| act_std | The standard deviation of the action in the training dataset used to normalize the actions. |
| primitive | The primitive to evaluate the policy on. This is used to determine the reset pose of the environment. |
| peg | The object ID to pass into the policy during rollout. This is only used for object ID conditioned policies. |

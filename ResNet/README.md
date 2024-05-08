# ResNet+MLP Policy
## Installation
Assume the machines have the lastest Nvdia drivers and CUDA Versions (either 12.1 or 11.x)
Run
```bash
conda create -n fmb_resnet python=3.9
conda activate fmb_resnet
pip install -r requirements.txt
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install torch

cd fmb/robot_infra
pip install -e .
```

## Training
```bash
bash ./scripts/train.sh
```

| Flags | Descriptions |  Format |
| --- | --- | --- |
| `--dataset_path` | Path to the directory containing the RLDS dataset. Multiple paths to multiple datasets can be set by separting paths with semi-colon. | PATH/TO/DIRECTORY_1;PATH/TO/DIRECTORY_2 |
| `--dataset_name` | Name and version number of the RLDS dataset. If no version number is specified, the latest is used. Multiple datasets can be loaded at the same time by separting them with semi-colon.| DATASET_NAME_1:VERSION;DATASET_NAME_2:VERSION |
| `--peg` | ID of the objects that should be filtered and trained on. Leaving the string empty will train on all objects in the dataset. | ID_1:ID_2:ID_3 |
| `--primitive` | Name of the primitives that should be filtered and trained on. Leaving the string empty will train on all primitives in the dataset. | PRIMITIVE_1:PRIMITIVE_2:PRIMITIVE_3 | 
| `--dataset_image_keys` | Name of the camera views to train the policy on. | CAM_1:CAM_2:CAM_3:CAM_4 |
| `--state_keys` | Name of the state observations to trian the policy on. | STATE_1:STATE_2:STATE_3 |
| `--tcp_frame` | Whether to transform the state observation and action from the base into the end-effector frame. This flag can be used to achieve pose invariant policies when trained with only wrist camera views. | boolean |
| `--last_action` | Whether to append the last action as part of the observation. | boolean |
| `--num_pegs` | Length of the one-hot vector that should be created to condition the policy on the object ID. Setting this to 0 means the policy will not be explicitly conditioned on the task object ID. | int |
| `--num_primitives`| Length of the one-hot vector that should be created to condition the policy on the primitive ID. Setting this to 0 means the policy will not be explicitly conditioned on the primitive ID. | int |
| `--num_frame_stack` | Number of past observations to include when predicting each action. | int |
| `--num_action_chunk` | Number of actions to predict per step. | int |
| `--resnet_type` | Size of the ResNets to use as the image encoder. | ResNet18 \ ResNet34 \ ResNet50 \ ResNet101 \ ResNet152 \ ResNet200 |
| `--image_augmentation` | Type of image augmentation to use for training | none \ rand \ trivial \ augmix \ color \ crop |
| `--clip_action` | Value to clip the training actions to ±clip_action value. | float |
| `--train_gripper` | Whether the last dimension of the action contains a binary gripper action. If true, the a BCE loss is used on the last dimension. | boolean |
| `--train_mse` | Whether to train the continuous action dimensions using MSE loss rather than log prob. | boolean |
| `--cache` | Whether to cache the entire dataset onto memory for faster dataloading. This is only possible if the entire dataset is small enough to fit onto memory. | boolean|

## Policy Rollout
To evaluate a policy on a single primitive
```bash
bash ./scripts/rollout_bc.sh
```
To evaluate a policy on multiple primitives
```bash
bash ./scripts/sequential_rollout.sh
```
|Flags|Description |
| --- |---|
|load_checkpoint| Path to load the trained checkpoint from. |
| model_key | The key of the saved models to evaluate. Refer to bc_main.py to see the list of models that are saved. |
| primitive | The primitive to evaluate the policy on. This is used to determine the reset pose of the environment. |

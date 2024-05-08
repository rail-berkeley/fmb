# FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning

![](./docs/intro.gif)

**Webpage: [https://functional-manipulation-benchmark.github.io/index.html](https://functional-manipulation-benchmark.github.io/index.html)**


FMB is a benchmark for robot learning consisting of various manipulation tasks, 22,550 expert demonstrations, and a set of imitation learning baselines. This repo contains the code and instructions to reproduce the benchmark, including robot controller, dataset builder, training and evaluation code for the various imitation learning baselines. 

**Table of Contents**
|Module|Description|
|---|---|
| [Robot Infra](./robot_infra) | For controllering the robot. |
| [Dataset Builder](./fmb_dataset_builder/) | Converting your own RLDS dataset for training. |
| [ResNet-based Policies](./ResNet/) | Training and evaluating ResNet-based policies. |
| [Transformer-based Policies](./Transformer/) | Training and evaluating Transformer-based policies. |
| [Diffusion Policies]() | Training and evaluating Diffusion policies. |

## Example Usage

We demonstrate how to use the FMB codebase to process data, train, and evaluate policies on the real robot. Let's imagine that we want to train a ResNet-based, object-ID conditioned insertion policy.

### Installation
Installl each module according to the instructions included. For this example, we will need:
- [Robot Infra](./robot_infra)
- [Dataset Builder](./fmb_dataset_builder/)
- [ResNet-based Policies](./ResNet/)

## 1. Processing Data

![](./images/peg.png)

> Example is located in [examples/async_peg_insert_drq/](../examples/async_peg_insert_drq/)

> Env and default config are located in `serl_robot_infra/franka_env/envs/peg_env/`

> The `franka_env.envs.wrappers.SpacemouseIntervention` gym wrapper provides the ability to intervene the robot with a spacemouse. This is useful for demo collection, testing robot, and making sure the training Gym environment works as intended.

The peg insertion task is best for getting started with running SERL on a real robot. As the policy should converge and achieve 100% success rate within 30 minutes on a single GPU in the simplest case, this task is great for trouble-shooting the setup quickly. The procedure below assumes you have a Franka arm with a Robotiq Hand-E gripper and 2 RealSense D405 cameras.

### Procedure
1. 3D-print (1) **Assembly Object** of choice and (1) corresponding **Assembly Board** from the **Single-Object Manipulation Objects** section of [FMB](https://functional-manipulation-benchmark.github.io/files/index.html). Fix the board to the workspace and grasp the peg with the gripper.
2. 3D-print (2) wrist camera mounts for the RealSense D405 and install onto the threads on the Robotiq Gripper. Create your own config from [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py), and update the camera serial numbers in `REALSENSE_CAMERAS`.
3. Adjust for the weight of the wrist camera by editing `Desk > Settings > End-effector > Mechnical Data > Mass`.
4. Unlock the robot and activate FCI in Desk. Then, start the franka_server by running:
    ```bash
    python serl_robo_infra/robot_servers/franka_server.py --gripper_type=<Robotiq|Franka|None> --robot_ip=<robot_IP> --gripper_ip=<[Optional] Robotiq_gripper_IP>
    ```
    This should start the impedance controller and a Flask server ready to recieve requests.
5. The reward in this task is given by checking whether the end-effector pose matches a fixed target pose. Grasp the desired peg with  `curl -X POST http://127.0.0.1:5000/close_gripper` and manually move the arm into a pose where the peg is inserted into the board. Print the current pose with `curl -X POST http://127.0.0.1:5000/getpos_euler` and update the `TARGET_POSE` in [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py) with the measured end-effector pose.

    **Note: make sure the wrist joint is centered (away from joint limits) and z-axis euler angle is positive at the target pose to avoid discontinuities.

6. Set `RANDOM_RESET` to `False` inside the config file to speedup training. Note the policy would only generalize to any board pose when this is set to `True`, but only try this after the basic task works.
7. Record 20 demo trajectories with the spacemouse.
    ```bash
    cd examples/async_peg_insert_drq
    python record_demo.py
    ```
    The trajectories are saved in `examples/async_peg_insert_drq/peg_insertion_20_trajs_{UUID}.pkl`.
8. Edit `demo_path` and `checkpoint_path` in `run_learner.sh` and `run_actor.sh`. Train the RL agent with the collected demos by running both learner and actor nodes.
    ```bash
    bash run_learner.sh
    bash run_actor.sh
    ```
9. If nothing went wrong, the policy should converge with 100% success rate within 30 minutes without `RANDOM_RESET` and 60 minutes with `RANDOM_RESET`.
10. The checkpoints are automatically saved and can be evaluated by setting the `--eval_checkpoint_step=CHECKPOINT_NUMBER_TO_EVAL` and `--eval_n_trajs=N_TIMES_TO_EVAL` flags in `run_actor.sh`. Then run:
    ```bash
    bash run_actor.sh
    ```
    If the policy is trained with `RANDOM_RESET`, it should be able to insert the peg even when you move the board at test time.


Let's take the peg insertion task as an example. We wrapped the env as such. The composability of the gym wrappers allows us to easily add or remove functionalities to the gym env. ([code](../examples/async_peg_insert_drq/async_drq_randomized.py))

```python
env = gym.make('FrankaPegInsert-Vision-v0')  # create the gym env
env = GripperCloseEnv(env)         # always keep the gripper close for peg insertion
env = SpacemouseIntervention(env)  # utilize spacemouse to intervene the robot
env = RelativeFrame(env)           # transform the TCP abs frame of ref to relative frame
env = Quat2EulerWrapper(env)       # convert rotation from quaternion to euler
env = SERLObsWrapper(env)          # convert observation to SERL format
env = ChunkingWrapper(env)         # chunking the observation
env = RecordEpisodeStatistics(env) # record episode statistics
```


<!-- ## Dataset Builder
The complete FMB dataset is released in `.npy` at the [dataset page](https://functional-manipulation-benchmark.github.io/dataset/index.html). The `.npy` files can be filtered and converted into  -->
<!-- This code consists of three components:
1. [fmb_dataset_builder](./fmb_dataset_builder/): used to convert data into RLDS format
2. [ResNet](./ResNet/): used to train and eval fResNet+MLP policies
3. [Transformer](./Transformer/): used to trian and eval Transformer-based policies -->

## BibTex
If you found this code useful, consider citing the following paper:
```
@article{luo2024fmb,
  title={FMB: a Functional Manipulation Benchmark for Generalizable Robotic Learning},
  author={Luo, Jianlan and Xu, Charles and Liu, Fangchen and Tan, Liam and Lin, Zipeng and Wu, Jeffrey and Abbeel, Pieter and Levine, Sergey},
  journal={arXiv preprint arXiv:2401.08553},
  year={2024}
}
```
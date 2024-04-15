# FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning
**Webpage: [https://functional-manipulation-benchmark.github.io/index.html](https://functional-manipulation-benchmark.github.io/index.html)**

FMB is a benchmark for robot learning consisting of various manipulation tasks, 22,550 expert demonstrations, and a set of imitation learning baselines. This repo contains the code and instructions to reproduce the benchmark, including robot controller, dataset builder, training and evaluation code for the various imitation learning baselines. 

**Table of Contents**
- [Robot Controller](#robot-controller)
- [Dataset Builder]()
- [ResNet-based Policies]()
- [Transformer-based Policies]()
- [Diffusion Policies]()

## Robot Controller
The robot infra used to collect the dataset is released as part of the [serl](https://github.com/rail-berkeley/serl/tree/main/serl_robot_infra) project and works for both Franka Emikda Panda and the newer Franka Research 3 arms. 

### Prerequisite
- ROS Noetic
- Franka Panda or Franka Research 3 arm and gripper
- `libfranka>=0.8.0` and `franka_ros>=0.8.0` installed according to [Franka FCI Documentation](https://frankaemika.github.io/docs/installation_linux.html)

### Install
```bash
cd robot_infra
conda create -n fmb_robot_infra python=3.9
conda activat fmb_robot_infra
pip install -e .
```
Install the [`serl_franka_controllers`](https://github.com/rail-berkeley/serl_franka_controllers) package.

    sudo apt-get install ros-serl_franka_controllers

### Usage
1. Launch the robot server, which run the robot controller and a Flask server which streams robot commands to the gym envionrment using HTTP requests. 
    ```bash
    conda activat fmb_robot_infra
    python franka_server.py --robot_ip=<robot_IP>
2. Create an instance of the gym environment in a second terminal.
    ```python
    import gym
    import fmb_robot_infra
    env = gym.make("Franka-FMB-v0")
    ```

## Dataset Builder
The complete FMB dataset is released in `.npy` at the [dataset page](https://functional-manipulation-benchmark.github.io/dataset/index.html). The `.npy` files can be filtered and converted into 
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
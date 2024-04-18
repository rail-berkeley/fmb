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
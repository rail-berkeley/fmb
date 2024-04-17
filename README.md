# FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning

![](./docs/intro.gif)

**Webpage: [https://functional-manipulation-benchmark.github.io/index.html](https://functional-manipulation-benchmark.github.io/index.html)**


FMB is a benchmark for robot learning consisting of various manipulation tasks, 22,550 expert demonstrations, and a set of imitation learning baselines. This repo contains the code and instructions to reproduce the benchmark, including robot controller, dataset builder, training and evaluation code for the various imitation learning baselines. 

**Table of Contents**
- [Robot Infra](./robot_infra)
- [Dataset Builder](./fmb_dataset_builder/)
- [ResNet-based Policies](./ResNet/)
- [Transformer-based Policies](./Transformer/)
- [Diffusion Policies]()



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
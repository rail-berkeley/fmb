# FMB
This repository is the code [FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning](https://functional-manipulation-benchmark.github.io/index.html).


This code consists of three components:
1. [fmb_dataset_builder](./fmb_dataset_builder/): used to convert data into RLDS format
2. [ResNet](./ResNet/): used to train and eval fResNet+MLP policies
3. [Transformer](./Transformer/): used to trian and eval Transformer-based policies

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
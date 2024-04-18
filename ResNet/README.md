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
```

## Training
```bash
. ./scripts/train.sh
```
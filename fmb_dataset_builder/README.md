# RLDS Dataset Conversion
For efficient dataloading, the entire FMB codebase requires the data to be converted into the RLDS format. 
The complete FMB dataset in RLDS format is released on [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/fmb) for convinence. 

However, for the best performance when training on a subset of the data (for example, a particular skill or task), we recommand pre-process the raw data into smaller filtered RLDS datasets. 

The raw dataset can be found on our [dataset page](https://functional-manipulation-benchmark.github.io/dataset/index.html) in `.npy` format. This module provides the code for creating a custom RLDS dataset from the `.npy` files.

## Installation
First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

## Converting Data
Each dataset folder (ie. `fmb_board_dataset`) specifiies one RLDS dataset. 

If you want to only train on a subset of the full data (ie. only one skill or only one object shape), then you can create a new RLDs dataset and edit the `*dataset_builder.py` file to filter and only inlcude the desired demonstrations, which can reduce the dataloading overhead during training.
```
ulimit -n 20000
cd <dataset_name>
tfds build --data_dir=<output_path>
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/output_path/<name_of_your_dataset>`.
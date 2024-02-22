# RLDS Dataset Conversion

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

If you want to only train on a subset of the full data (ie. only one skill or only one object shape), then you can create a new RLDs dataset with only the desired demonstrations, which can reduce the dataloading overhead during training.
```
ulimit -n 20000
cd <dataset_name>
tfds build --overwrite --data_dir=<output_path>
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/output_path/<name_of_your_dataset>`.
#! /bin/bash
#SBATCH --job-name=bc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --account=co_rail
#SBATCH --qos=savio_lowprio
#SBATCH --partition=savio3_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# Exit the script if it is not launched from slurm.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script is not launched with slurm, exiting!"
    exit 1
fi

module load gnu-parallel

export SCRIPT_PATH="$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | head -n 1)"
export SCRIPT_DIR="$(dirname $SCRIPT_PATH)"
cd $SCRIPT_DIR


EXP_NAME='brc_test'
OUTPUT_DIR="$SCRIPT_DIR/brc_experiment_output"
mkdir -p "$OUTPUT_DIR/$EXP_NAME"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/$EXP_NAME/"

# Controls how many processes are running in parallel for each array task
export RUNS_PER_TASK=1


parallel --delay 20 --linebuffer -j $RUNS_PER_TASK \
    '[ $SLURM_ARRAY_TASK_ID == $(({#} % $SLURM_ARRAY_TASK_COUNT)) ] && 'singularity run \
        -B /var/lib/dcv-gl --nv --writable-tmpfs $SCRIPT_DIR/code_img.sif \
        src.bc_main \
            --seed={1} \
            --dataset_path="$SCRIPT_DIR/data/data.npy" \
            --dataset_image_keys={2} \
            --image_augmentation={3} \
            --total_steps=20000 \
            --eval_freq=100 \
            --save_model=True \
            --lr={4} \
            --weight_decay={5} \
            --policy.spatial_aggregate={6} \
            --policy.resnet_type={7} \
            --policy.state_injection={8} \
            --policy.share_resnet_between_views=False \
            --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
            --logger.mode=online \
            --logger.prefix='PegInsertion' \
            --logger.project="$EXP_NAME" \
            --logger.random_delay=60.0 \
        ::: 24 \
        ::: 'side_image:wrist45_image:wrist225_image' \
        ::: 'none' 'rand' 'color' \
        ::: 1e-3 \
        ::: 3e-3 \
        ::: 'average' \
        ::: 'ResNet18' \
        ::: 'no_gripper' 'full' \

# 'side_image' 'side_image:wrist45_image:wrist225_image' \

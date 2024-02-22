#! /bin/bash

if [ -z "$1" ]; then # Check if the script is invoked without any argument
    slurm_run () {
        sbatch \
            --job-name=dec100 \
            --time=168:00:00 \
            --account=co_rail \
            --qos=rail_gpu3_normal \
            --partition=savio3_gpu \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=4 \
            --gres=gpu:TITAN:1 \
            --mem=32G \
            $@
    }
    #  mem 40gb, 4 cpu/task
    # Use --qos=rail_gpu3_normal and --gres=gpu:TITAN:1 for normal jobs
    for seed in 0; do
        slurm_run "${BASH_SOURCE[0]}" $seed
    done
# path to code sif
else
    singularity run -B /var/lib/dcv-gl --nv --writable-tmpfs /global/scratch/users/arovinsky001/brc/code.sif \
            imitate_episodes \
            --seed=$1 \
            --task_name sandwich_bag_insert_carrot_0721 \
            --ckpt_dir /global/scratch/users/arovinsky001/sandwich_bag_insert_carrot_0721_ckpt0725_decoder_rgb_batch20_brc \
            --policy_class ACT \
            --kl_weight 10 \
            --chunk_size 100 \
            --hidden_dim 512 \
            --batch_size 20 \
            --dim_feedforward 3200 \
            --num_epochs 4000 \
            --lr 1e-4 \
            --temporal_agg \
            --grad_accum_steps 1 \
            --zero_qpos \
            --use_decoder_model
fi


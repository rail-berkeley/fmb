#!/bin/bash
EXP_NAME='EXP_NAME'
export PROJECT_HOME="$(pwd)"
OUTPUT_DIR="PATH_TO_CHECKPOINT"
export XLA_PYTHON_CLIENT_PREALLOCATE='false'
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="$PYTHONPATH:$PROJECT_HOME/src"
 
python -m src.bc_main \
        --dataset_path="/home/undergrad/fmb_dev/data/" \
        --dataset_name="fmb_peg1_dataset:2.0.0" \
        --seed=24 \
        --dataset_image_keys='side_1:side_2:wrist_1:wrist_2' \
        --state_keys='tcp_pose:tcp_vel:tcp_force:tcp_torque' \
        --policy.state_injection='full' \
        --last_action=False \
        --image_augmentation='none' \
        --total_steps=100000 \
        --eval_freq=200 \
        --train_ratio=0.98 \
        --batch_size=64 \
        --save_model=True \
        --lr=1e-4 \
        --weight_decay=1e-3 \
        --policy.spatial_aggregate='average' \
        --resnet_type='ResNet34' \
        --policy.share_resnet_between_views=False \
        --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
        --logger.mode=disabled \
        --logger.prefix='PegInsertion' \
        --logger.project="$EXP_NAME" \
        --train_gripper=True \
        --device='gpu' \
        --num_pegs=1 \
        --num_primitives=4 \
        --primitive="grasp:place_on_fixture:regrasp:insert"
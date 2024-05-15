XLA_PYTHON_CLIENT_PREALLOCATE='false' \
python transformer_rollout.py \
    --checkpoint_path='PATH_TO_CHECKPOINT' \
    --wandb_run_name='WANDB_RUN_NAME' \
    --act_mean 0 0 0 0 0 0 0 \
    --act_std 0 0 0 0 0 0 0 \
    --primitive='PRIMITIVE' \
    --peg=PEG_ID
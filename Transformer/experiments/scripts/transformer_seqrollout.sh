XLA_PYTHON_CLIENT_PREALLOCATE='false' \
python transformer_seqrollout.py \
    --checkpoint_path='PATH_TO_CHECKPOINT' \
    --wandb_run_name='WANDB_RUN_NAME' \
    --primitive='PRIMITIVE' \
    --peg=PEG_ID




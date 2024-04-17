XLA_PYTHON_CLIENT_PREALLOCATE='false' \
python -m src.rollout_main \
    --load_checkpoint='PATH_TO_CHECKPOINT.pkl' \
    --model_key='train_state' \
    --primitive='insert'
    
XLA_PYTHON_CLIENT_PREALLOCATE='false' \
python -m src.rollout_main \
    --load_checkpoint='/media/nvmep3p/fmb2/experiment_output/fmb2_insert_final/6f46daa0d525492d8f05a29af16fecd0/model.pkl' \
    --model_key='train_state' \
    --primitive='insert'
    
export XLA_PYTHON_CLIENT_PREALLOCATE=false

NAME="transformer_bc_cond_4"

python experiments/train_rlds_fmb.py \
    --config experiments/configs/train_config.py:transformer_bc_cond_insert \
    --name $NAME
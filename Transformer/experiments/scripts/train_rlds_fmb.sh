export XLA_PYTHON_CLIENT_PREALLOCATE=false

NAME="transformer_bc_cond_4"
CMD="python experiments/train_rlds_fmb.py \
    --config experiments/configs/train_config.py:transformer_bc_cond_4 \
    --name $NAME"
$CMD

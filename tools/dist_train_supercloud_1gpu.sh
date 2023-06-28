#!/bin/bash
#SBATCH --job-name=segtrain_1gpu    # Job name
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks=1                  # same as num GPU
#SBATCH --cpus-per-task=5           # default value


CONFIG=$1
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

module load anaconda/2023a-pytorch

export PYTHONPATH="$(pwd)":$PYTHONPATH

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(pwd)/tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:2}

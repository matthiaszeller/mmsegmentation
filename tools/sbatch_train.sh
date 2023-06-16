#!/usr/bin/env bash
# single node !
#SBATCH --job-name=mmseg
#SBATCH --nodes=1
#SBATCH --ntasks=2                  # same as num gpu
#SBATCH --ntasks-per-node=2         # same as num gpu per node
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=5

set -x

CONFIG=$1
PY_ARGS=${@:2}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}

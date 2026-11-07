#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/trsppo/lib/
# robot="${ROBOT:-x1_go2bend}"

# In another terminal: conda activate trsppo && tensorboard --logdir runs
python -u scripts/train.py --robot "x1" --algorithm ppo --headless --num-envs 4096 --iterations 6000 --tensorboard "$@"
python -u scripts/train.py --robot "x1" --algorithm trsppo --headless --num-envs 4096 --iterations 6000 --tensorboard "$@"


python -u scripts/train.py --robot "x1_go2bend" --algorithm ppo --headless --num-envs 4096 --iterations 6000 --tensorboard "$@"
python -u scripts/train.py --robot "x1_go2bend" --algorithm trsppo --headless --num-envs 4096 --iterations 6000 --tensorboard "$@"

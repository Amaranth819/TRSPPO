#!/bin/bash
# set -e

source "/home/$(whoami)/miniconda3/etc/profile.d/conda.sh"
conda activate tsda

# Use Conda's C++ runtime for Pillow/torchvision, and expose MuJoCo's libraries.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/home/$(whoami)/.mujoco/mujoco210/bin"

# Pass --agent, --cfg, and any other train.py options through unchanged.
exec python -u train.py "$@" --no_save_video

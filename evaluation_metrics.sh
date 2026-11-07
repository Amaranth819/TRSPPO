#!/usr/bin/env bash
# Evaluate all three robots, both algorithms, four gaits, and six signed speeds.
# Activate the same Conda environment used for training before running this file.
set -euo pipefail

EVALUATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$EVALUATION_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Examples:
# bash evaluation_metrics.sh --dry-run
# bash evaluation_metrics.sh --robot go2 --algorithm trsppo --run RUN_NAME
# bash evaluation_metrics.sh --robot x1 --algorithm ppo --gait trot --velocities 0.5
exec "${PYTHON:-python}" -u "$EVALUATION_ROOT/scripts/evaluate.py" "$@"

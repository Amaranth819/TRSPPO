#!/usr/bin/env bash
set -euo pipefail

EVALUATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EVALUATION_ROOT"

bash evaluation_metrics.sh --robot go2 --algorithm ppo --run "1" "$@"
bash evaluation_metrics.sh --robot go2 --algorithm trsppo --run "115608.822913" "$@"

bash evaluation_metrics.sh --robot x1 --algorithm ppo --run "2026-10-30_09-38-28.956998" "$@"
bash evaluation_metrics.sh --robot x1 --algorithm trsppo --run "2026-10-30_12-04-37.971291" "$@"

bash evaluation_metrics.sh --robot x1_go2bend --algorithm ppo --run "2026-10-30_14-32-48.134909" "$@"
bash evaluation_metrics.sh --robot x1_go2bend --algorithm trsppo --run "2026-10-30_17-00-45.017274" "$@"

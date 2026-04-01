#!/usr/bin/env bash
# ./scripts/run_benchmark_sampling_vs_iteration.sh \
#   --config config/MailerBoxTask.json \
#   --methods Sampling,Iteration \
#   --scalings 1.0,1.2 \
#   --box-positions "0.6:0.1:0.4;0.65:0.1:0.4" \
#   --box-yaws 0.0,15,-15,30,-30 \
#   --closed-states true,false \
#   --seeds 0 \
#   --output data/benchmark_sampling_vs_iteration.csv


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BENCHMARK_SCRIPT="scripts/benchmark_sampling_vs_iteration.py"

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n box python "${BENCHMARK_SCRIPT}" "$@"
fi

exec python "${BENCHMARK_SCRIPT}" "$@"

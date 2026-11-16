#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python -B -u benchmark_speed.py "$@" 2>&1 | tee -a logs/benchmark_console.log

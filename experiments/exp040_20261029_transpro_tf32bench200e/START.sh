#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python -B -u run.py train 2>&1 | tee -a logs/train_console.log

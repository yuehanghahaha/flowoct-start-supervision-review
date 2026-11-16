#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p logs
python -B -u suite.py prepare "$@" 2>&1 | tee -a logs/setup_a100.log

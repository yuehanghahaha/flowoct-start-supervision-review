#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /workspace/.conda/envs/lyh_flowoct/bin/python -B runs/slot2/control.py "$@"

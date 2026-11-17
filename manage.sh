#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:-status}"
MANIFEST="${2:-}"
RUN="${3:-$(dirname "$0")/runs/main}"
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
USER_HOME_RESOLVED="$(cd "$HOME" && pwd -P)"
RUN="$(realpath -m "$RUN")"
case "$ROOT/" in "$USER_HOME_RESOLVED/"*) ;; *) echo 'Enter your allocated persistent-home container first.' >&2; exit 1;; esac
case "$RUN/" in "$USER_HOME_RESOLVED/"*) ;; *) echo 'Results must remain in your persistent home.' >&2; exit 1;; esac
mkdir -p "$RUN"
PY="$HOME/.conda/envs/lyh_flowoct/bin/python"
TMUX="$HOME/.conda/envs/lyh_tmux/bin/tmux"
test -x "$PY" && test -x "$TMUX"
for item in tmp tmux torch triton cuda; do mkdir -p "$RUN/cache/$item"; done
export TMUX_TMPDIR="$RUN/cache/tmux" TMPDIR="$RUN/cache/tmp" TMP="$RUN/cache/tmp" TEMP="$RUN/cache/tmp"
export TORCH_HOME="$RUN/cache/torch" TRITON_CACHE_DIR="$RUN/cache/triton" CUDA_CACHE_PATH="$RUN/cache/cuda"
export XDG_CACHE_HOME="$RUN/cache" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
chmod 700 "$TMUX_TMPDIR"
SESSION="exp181_subslab4_$(basename "$RUN")"
case "$ACTION" in
  start|resume)
    test -n "$MANIFEST"
    if "$TMUX" has-session -t "=$SESSION" 2>/dev/null; then echo 'Session already active.'; exit 1; fi
    if [[ "$ACTION" == start && ( -e "$RUN/checkpoints/latest.pt" || -e "$RUN/teacher/latest.pt" ) ]]; then
      echo 'Saved state exists. Use resume.' >&2; exit 1
    fi
    rm -f -- "$RUN/pause.request"
    mkdir -p "$RUN/logs"
    LOG="$RUN/logs/$(date +%Y%m%d_%H%M%S).log"
    printf -v COMMAND '%q ' "$PY" -B -u "$ROOT/run.py" fit --manifest "$MANIFEST" --run "$RUN"
    printf -v QUOTED_LOG '%q' "$LOG"
    "$TMUX" new-session -d -s "$SESSION" -c "$ROOT" bash -c "set -o pipefail; $COMMAND 2>&1 | tee -a $QUOTED_LOG"
    echo "Session: $SESSION; log: $LOG"
    ;;
  pause) touch "$RUN/pause.request"; echo 'Pause requested after complete update/patient.' ;;
  status) "$PY" -B "$ROOT/run.py" status --run "$RUN" ;;
  attach) "$TMUX" attach-session -t "=$SESSION" ;;
  *) echo 'Usage: manage.sh start|resume|pause|status|attach MANIFEST RUN' >&2; exit 2 ;;
esac

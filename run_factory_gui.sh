#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/sbedi/Nextcloud/automatic_to_autonomous/CNC Factory"
PYTHON_BIN="/opt/anaconda3/bin/python"

cd "$PROJECT_DIR"

# Clear env vars that can break Tk rendering in some terminals.
unset PYTHONHOME PYTHONPATH TCL_LIBRARY TK_LIBRARY

# Ignore whatever env is currently activated in the shell.
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV

# Keep terminal output clean on macOS system Tk.
export TK_SILENCE_DEPRECATION=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python not found at $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/factory_gui.py" "$@"

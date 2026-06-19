#!/usr/bin/env bash
set -euo pipefail

# Convenience alias for users who type "factory_gui.sh".
exec "$(dirname "$0")/run_factory_gui.sh" "$@"

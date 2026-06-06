#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../../run_kernelmem_radio_one.sh" "$@"

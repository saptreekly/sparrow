#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <qasm-file> [<qasm-file> ...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for qasm in "$@"; do
  "$SCRIPT_DIR/run_vqe_bench.sh" "$qasm"
  echo "Completed benchmark for $qasm"
  echo
  echo
 done

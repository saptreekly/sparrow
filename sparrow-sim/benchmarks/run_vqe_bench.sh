#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <qasm-file>" >&2
  exit 1
fi

QASM="$1"
if [ ! -f "$QASM" ]; then
  echo "error: QASM file '$QASM' not found" >&2
  exit 1
fi

mkdir -p logs

for mode in dense sparse predictive; do
  echo "-- Running $mode mode on $QASM --"
  /usr/bin/time -l \
    cargo run --manifest-path sparrow-sim/Cargo.toml --release -- \
      --mode "$mode" --threshold 1e-9 "$QASM" \
    2>&1 | tee "logs/$(basename "${QASM%.qasm}")_${mode}.log"
  echo
  echo
  # ensure subsequent iterations reuse the compiled binary
  touch sparrow-sim/target/release/sparrow-sim
 done

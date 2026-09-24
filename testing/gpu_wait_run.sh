#!/bin/bash
# Hold a GPU allocation until a go file appears, then run the given script: gpu_wait_run.sh <GO_FILE> <script> [args]
GO=$1; shift
echo "=== allocation up $(date) on $(hostname); waiting for $GO ==="
for i in $(seq 1 240); do
  if [ -f "$GO" ]; then echo "=== go $(date) ==="; bash "$@"; exit $?; fi
  sleep 10
done
echo "=== gave up $(date) ==="; exit 1

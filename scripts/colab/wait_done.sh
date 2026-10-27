#!/bin/bash
# Poll every 2 minutes until all runs are DONE (exit 0) or the training process dies (exit 2).
source "$(dirname "$0")/env.sh"
for i in $(seq 1 180); do
  out=$(bash "$R/scripts/colab/run.sh" poll.py)
  line=$(echo "$out" | grep "alive:")
  echo "$(date +%H:%M:%S) $line"
  echo "$line" | grep -q "DONE: True" && { echo "$out" | tail -8; exit 0; }
  echo "$line" | grep -q "alive: False" && { echo "$out" | tail -14; exit 2; }
  sleep 120
done
exit 3

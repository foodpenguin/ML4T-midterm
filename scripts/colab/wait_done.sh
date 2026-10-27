#!/bin/bash
D=/mnt/d/penguin_data/code/quantitative_trading_course/ML4T_Midterm/artifacts/deep/colab
for i in $(seq 1 120); do
  out=$(bash $D/run.sh poll.py)
  line=$(echo "$out" | grep "alive:")
  echo "$(date +%H:%M:%S) $line $(echo "$out" | grep -E '^\[.*fold' | tail -1)"
  echo "$line" | grep -q "DONE: True" && { echo "$out" | tail -6; exit 0; }
  echo "$line" | grep -q "alive: False" && { echo "$out" | tail -12; exit 2; }
  sleep 120
done
exit 3

#!/bin/bash
# usage: run.sh <helper.py>   runs a helper from scripts/colab on the VM
source "$(dirname "$0")/env.sh"
"$C" exec -s "$S" -f "$R/scripts/colab/$1" 2>&1 | quiet

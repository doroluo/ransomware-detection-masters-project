#!/bin/bash
# Stop the GNN queue with results saved: bash stop_queue.sh [--now]
# Touches RUNS/STOP: queue_gnn.sh finishes the run in progress, logs "stopped before <name>" for the rest and still
# writes the report and gate summary. With --now the running train.py python process(es) are killed too (matched by
# command line); the killed run has no metrics.json, so a relaunch reruns it.
# GT_RUNS overrides the runs directory, GT_TRAIN_PATTERN the regex that finds train.py (testing).
R=/c/Users/chaoa/Downloads/graph_transformer_runs
R=${GT_RUNS:-$R}; PAT=${GT_TRAIN_PATTERN:-train[.]py}
case $1 in ""|--now) ;; *) echo "usage: bash stop_queue.sh [--now]"; exit 2;; esac
mkdir -p $R && touch $R/STOP && echo "created $R/STOP: the queue finishes the current run and skips the rest"
if [ "$1" = --now ]; then
  killed=$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$PAT' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force; \$_.ProcessId }" | tr -d '\r' | tr '\n' ' ')
  if [ -n "$killed" ]; then echo "killed train.py process(es): $killed"; else echo "no train.py process was running"; fi
fi

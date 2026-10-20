#!/bin/bash
# The Model 1 runs (token-sequence transformer over the block export ours_cap0): bash scripts/model1_runs.sh [--dry]
# Run from anywhere (it cd's to the code root). In this order:
#   1. python -m token_sequence.train --protocol kfold --arch x86 --fold <f> --seed <s>  for seeds 0 1 2, folds 0-4 (15 runs, seed by
#      seed): kfold<f>_x86_tokenseq_s<s> (RunConfig.name of model 'tokenseq'; defaults seq-len 512, max-windows 64, export ours_cap0)
#   2. report.py (RESULTS.md and the table on stdout, log RUNS/logs/report_model1.log)
#   3. gates.py --stage model1 > RUNS/gates_model1.md
# A run is skipped when RUNS/<name>/metrics.json exists. RUNS/STOP makes the script log "stopped before <name>" and skip. A run that
# exits non-zero is retried once ("retry <name>", the first log kept as <name>.log.try1), not when RUNS/STOP exists. Per-run logs:
# RUNS/logs/<name>.log. The script exits 0 only when all 15 runs have a metrics.json and report.py and gates.py both succeed.
# --dry prints "run <name> <train args>" for each run and the two closing steps, and writes nothing; tests/test_model1_trainer.py
# checks the names against RunConfig.name. GT_PY / GT_RUNS / GT_EXPORT override the interpreter / runs directory / export directory.
cd "$(dirname "$0")/.."
PY=${GT_PY:-/c/Users/chaoa/Downloads/gt_venv/Scripts/python.exe}
R=${GT_RUNS:-/c/Users/chaoa/Downloads/graph_transformer_runs}
STOP=$R/STOP
DRY=0; [ "$1" = --dry ] && DRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1   # unbuffered: the per-run log shows each epoch as it ends
[ $DRY = 0 ] && mkdir -p $R/logs

run() { name=$1; shift
  if [ $DRY = 1 ]; then echo "run $name $*"; return; fi
  [ -f $R/$name/metrics.json ] && { echo "skip $name"; return; }
  [ -f $STOP ] && { echo "stopped before $name"; return; }
  echo "start $name $(date +%T)"
  $PY -m token_sequence.train "$@" > $R/logs/$name.log 2>&1; rc=$?
  if [ $rc != 0 ] && [ ! -f $STOP ]; then echo "retry $name"; mv -f $R/logs/$name.log $R/logs/$name.log.try1
    $PY -m token_sequence.train "$@" > $R/logs/$name.log 2>&1; rc=$?; fi
  echo "finished $name rc=$rc $(date +%T)"; }

for s in 0 1 2; do for f in 0 1 2 3 4; do
  run kfold${f}_x86_tokenseq_s${s} --protocol kfold --arch x86 --fold $f --seed $s
done; done

if [ $DRY = 1 ]; then echo "cpu report.py"; echo "cpu gates.py --stage model1 > RUNS/gates_model1.md"; exit 0; fi
missing=0
for s in 0 1 2; do for f in 0 1 2 3 4; do [ -f $R/kfold${f}_x86_tokenseq_s${s}/metrics.json ] || missing=$((missing + 1)); done; done
echo "start report $(date +%T)"
$PY report.py > $R/logs/report_model1.log 2>&1; rrc=$?
echo "finished report rc=$rrc $(date +%T)"
echo "start gates model1 $(date +%T)"
$PY gates.py --stage model1 > $R/gates_model1.md 2> $R/logs/gates_model1.log; grc=$?
echo "finished gates model1 rc=$grc, $missing of 15 runs without metrics.json $(date +%T)"
[ $missing = 0 ] && [ $rrc = 0 ] && [ $grc = 0 ]

#!/bin/bash
# The api-token ablation (vocab `noapi`: mn, sig and beh tokens per block, no api token): bash scripts/noapi_runs.sh [--dry]
# Run from the code root (master5 checks the worktree out at the commit named in graph_transformer_runs/noapi_ready, cd's here
# and calls it). In this order:
#   1. the CPU baseline  baselines.py --protocol kfold --fold 0 1 2 3 4 --arch x86 --vocab noapi  (TF-IDF over mn + sig + beh,
#      directories kfold<f>_x86_s0_voc-noapi), skipped when all five metrics.json exist
#   2. the GPU runs  train.py --protocol kfold --arch x86 --model gnn --mask-blocks 0.15 --vocab noapi  for seeds 0 1 2, folds 0-4
#      (15 runs, seed by seed): kfold<f>_x86_gnn-mblk15-voc-noapi_s<seed>
#   3. gates.py --stage noapi > RUNS/gates_noapi.md
# A run is skipped when RUNS/<name>/metrics.json exists. RUNS/STOP makes the script log "stopped before <name>" and skip. A run
# that exits non-zero is retried once ("retry <name>", the first log kept as <name>.log.try1), not when RUNS/STOP exists.
# Per-run logs: RUNS/logs/<name>.log. The script exits 0 only when all 15 runs have a metrics.json and the gates file is written.
# --dry prints "run <name> <train.py args>" for each run and the two CPU steps, and writes nothing; tests/test_noapi.py checks the
# names against RunConfig.name. GT_PY / GT_RUNS override the interpreter / runs directory.
cd "$(dirname "$0")/.."
PY=${GT_PY:-/c/Users/chaoa/Downloads/gt_venv/Scripts/python.exe}
R=${GT_RUNS:-/c/Users/chaoa/Downloads/graph_transformer_runs}
STOP=$R/STOP
DRY=0; [ "$1" = --dry ] && DRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ $DRY = 0 ] && mkdir -p $R/logs

run() { name=$1; shift
  if [ $DRY = 1 ]; then echo "run $name $*"; return; fi
  [ -f $R/$name/metrics.json ] && { echo "skip $name"; return; }
  [ -f $STOP ] && { echo "stopped before $name"; return; }
  echo "start $name $(date +%T)"
  $PY train.py "$@" > $R/logs/$name.log 2>&1; rc=$?
  if [ $rc != 0 ] && [ ! -f $STOP ]; then echo "retry $name"; mv -f $R/logs/$name.log $R/logs/$name.log.try1
    $PY train.py "$@" > $R/logs/$name.log 2>&1; rc=$?; fi
  echo "finished $name rc=$rc $(date +%T)"; }

BASE="--protocol kfold --fold 0 1 2 3 4 --arch x86 --vocab noapi"
if [ $DRY = 1 ]; then echo "cpu baselines.py $BASE"
else
  todo=0; for f in 0 1 2 3 4; do [ -f $R/baselines/kfold${f}_x86_s0_voc-noapi/metrics.json ] || todo=1; done
  if [ $todo = 0 ]; then echo "skip baselines noapi"
  elif [ -f $STOP ]; then echo "stopped before baselines noapi"
  else echo "start baselines noapi $(date +%T)"; $PY baselines.py $BASE > $R/logs/baselines_vocab_noapi.log 2>&1
    echo "finished baselines noapi rc=$? $(date +%T)"; fi
fi

for s in 0 1 2; do for f in 0 1 2 3 4; do
  run kfold${f}_x86_gnn-mblk15-voc-noapi_s${s} --protocol kfold --fold $f --arch x86 --seed $s --model gnn --mask-blocks 0.15 --vocab noapi
done; done

if [ $DRY = 1 ]; then echo "cpu gates.py --stage noapi > RUNS/gates_noapi.md"; exit 0; fi
missing=0
for s in 0 1 2; do for f in 0 1 2 3 4; do [ -f $R/kfold${f}_x86_gnn-mblk15-voc-noapi_s${s}/metrics.json ] || missing=$((missing + 1)); done; done
echo "start gates noapi $(date +%T)"
$PY gates.py --stage noapi > $R/gates_noapi.md 2> $R/logs/gates_noapi.log; grc=$?
echo "finished gates noapi rc=$grc, $missing of 15 runs without metrics.json $(date +%T)"
[ $missing = 0 ] && [ $grc = 0 ]

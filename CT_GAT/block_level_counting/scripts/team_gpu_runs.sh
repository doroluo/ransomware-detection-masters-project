#!/bin/bash
# GPU follow-up of the team-pipelines comparison: bash scripts/team_gpu_runs.sh [--dry]   (not started by the step that wrote it)
# Waits for "master4 done" in RUNS/logs/master4.log (the GPU queue's last line), then runs, in this order:
#   1. the matched arms of the 500-block comparison, kfold x86, seeds 0 1 2, folds 0-4, 15% block masking, subgraph cap with a
#      budget of 500 blocks, --tag b500: the typed GNN (kfold<f>_x86_gnn-mblk15-sg-cap500_s<seed>_b500) and its --control
#      no_edges twin (kfold<f>_x86_gnn-noedges-mblk15-sg-cap500_s<seed>_b500); seed by seed, typed arm first
#   2. queue_gnn.sh --stage paths_codebert (its own skip / budget / retry rules; the CodeBERT arms of path 1 and 2)
#   3. report.py, then gates.py --stage team > RUNS/gates_team.md and gates.py --stage paths_codebert > RUNS/gates_paths_codebert.md
# A run is skipped when RUNS/<name>/metrics.json exists. RUNS/STOP (the GPU queue's file; stop_queue.sh creates it) makes the script
# log "stopped before <name>" and skip, also while it waits for master4. A run that exits non-zero is retried once ("retry
# <name>", first log kept as <name>.log.try1), not when RUNS/STOP exists. Per-run logs: RUNS/logs/<name>.log.
# --dry prints every step in order and writes nothing. GT_PY / GT_RUNS override the interpreter / runs directory.
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

# matched <variant> <train.py args>: seeds 0 1 2, folds 0-4 of the cap-500 arm (the variant is RunConfig.variant)
matched() { variant=$1; shift
  for s in 0 1 2; do for f in 0 1 2 3 4; do
    run kfold${f}_x86_${variant}_s${s}_b500 --protocol kfold --fold $f --arch x86 --seed $s --model gnn --mask-blocks 0.15 \
      --subgraph cap --budget 500 --tag b500 "$@"
  done; done; }

if [ $DRY = 1 ]; then echo "wait for 'master4 done' in $R/logs/master4.log"
else echo "team_gpu_runs waiting for master4 $(date +%T)"
  until grep -q 'master4 done' $R/logs/master4.log 2>/dev/null; do
    [ -f $STOP ] && { echo "stopped while waiting for master4"; exit 0; }
    sleep 120
  done
  echo "master4 done seen $(date +%T)"; fi

matched gnn-mblk15-sg-cap500                                        # 1. our matched arm, typed
matched gnn-noedges-mblk15-sg-cap500 --control no_edges             #    and its no-edge twin
if [ $DRY = 1 ]; then echo "queue bash scripts/queue_gnn.sh --stage paths_codebert"      # 2. CodeBERT paths 1 and 2, three seeds
elif [ -f $STOP ]; then echo "stopped before stage paths_codebert"
else echo "start stage paths_codebert $(date +%T)"; bash scripts/queue_gnn.sh --stage paths_codebert > $R/logs/queue_paths_codebert.log 2>&1
  echo "finished stage paths_codebert rc=$? $(date +%T)"; fi
if [ $DRY = 1 ]; then echo "cpu report.py"; echo "cpu gates.py --stage team > RUNS/gates_team.md"
  echo "cpu gates.py --stage paths_codebert > RUNS/gates_paths_codebert.md"
else                                                                # 3. report and gates
  echo "start report $(date +%T)"; $PY report.py > $R/logs/report.log 2>&1; echo "finished report rc=$?"
  $PY gates.py --stage team > $R/gates_team.md 2> $R/logs/gates_team.log; echo "finished gates team rc=$?"
  $PY gates.py --stage paths_codebert > $R/gates_paths_codebert.md 2> $R/logs/gates_paths_codebert.log
  echo "finished gates paths_codebert rc=$?"; fi
echo "team_gpu_runs done $(date +%T)"

#!/bin/bash
# Resumable GNN run queue: bash queue_gnn.sh --stage day1 [--dry]
# A run is skipped when RUNS/<name>/metrics.json exists. GPU_BUDGET_MIN (default 290) caps the cumulative GPU wall
# time: once it is spent, remaining GPU runs are logged as "skipped (budget)" and only the CPU/report steps still
# run. --dry prints what would run (DRY_RUN_MIN=<min> charges that many minutes per run to try the guard).
# GT_PY / GT_RUNS override the interpreter / runs directory (testing).
cd "$(dirname "$0")"
PY=${GT_PY:-/c/Users/chaoa/Downloads/gt_venv/Scripts/python.exe}
R=${GT_RUNS:-/c/Users/chaoa/Downloads/graph_transformer_runs}
STAGE=; DRY=0
while [ $# -gt 0 ]; do
  case $1 in --stage) STAGE=$2; shift;; --dry) DRY=1;; *) STAGE=;; esac; shift
done
[ "$STAGE" = day1 ] || { echo "usage: bash queue_gnn.sh --stage day1 [--dry]"; exit 2; }
GPU_BUDGET_MIN=${GPU_BUDGET_MIN:-290}
DRY_RUN_MIN=${DRY_RUN_MIN:-0}
GPU_SECS=0

if [ $DRY = 0 ]; then
  mkdir -p $R/logs
  if ! powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'train[.]py' }) { exit 1 }"; then
    echo "a train.py process is alive; refusing to start"; exit 1
  fi
  $PY -c "import torch, sklearn, scipy, train, data, model_gnn, stats, gates; assert torch.cuda.is_available(); print('preamble ok', torch.cuda.get_device_name(0))" || { echo "preamble failed"; exit 1; }
fi

run() { name=$1; shift
  if [ $DRY = 0 ] && [ -f $R/$name/metrics.json ]; then echo "skip $name"; return; fi
  if [ $GPU_SECS -ge $((GPU_BUDGET_MIN * 60)) ]; then echo "skipped (budget) $name"; return; fi
  if [ $DRY = 1 ]; then echo "run $name $*"; GPU_SECS=$((GPU_SECS + DRY_RUN_MIN * 60)); return; fi
  echo "start $name $(date +%T)"; t0=$SECONDS
  $PY train.py "$@" > $R/logs/$name.log 2>&1; rc=$?
  GPU_SECS=$((GPU_SECS + SECONDS - t0))
  echo "finished $name rc=$rc $(date +%T) gpu_min=$((GPU_SECS / 60))"; }

# arm <variant> <train.py args>: kfold x86 folds 0-4, seed 0
arm() { variant=$1; shift
  for f in 0 1 2 3 4; do run kfold${f}_x86_${variant}_s0 --protocol kfold --fold $f --arch x86 --seed 0 "$@"; done; }

# 1. CPU baselines (with validation predictions and thresholds); no GPU
BASE_ARGS="--protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0"
if [ $DRY = 1 ]; then echo "cpu baselines $BASE_ARGS"
elif [ "$(grep -l threshold_val $R/baselines/kfold[0-4]_x86_s0/metrics.json 2>/dev/null | wc -l)" = 5 ]; then echo "skip baselines"
else echo "start baselines $(date +%T)"; $PY baselines.py $BASE_ARGS > $R/logs/baselines_kfold_x86_s0.log 2>&1; echo "finished baselines rc=$? $(date +%T)"; fi

arm gnn-noedges --model gnn --control no_edges                       # 2. R2 tfidf
arm gnn-noedges-enc-mean --model gnn --control no_edges --encoder mean  # 3. R2 mean
arm gnn --model gnn                                                  # 4. R7
arm gnn-untyped --model gnn --untyped                                # 5. R5
arm gnn-shuffle --model gnn --control shuffle                        # 6. R4
run kfold3_x86_nograph_s2 --protocol kfold --fold 3 --arch x86 --seed 2 --no-graph   # 7. transformer completions
run kfold4_x86_graph_s2 --protocol kfold --fold 4 --arch x86 --seed 2
run kfold4_x86_nograph_s2 --protocol kfold --fold 4 --arch x86 --seed 2 --no-graph

# 8. report steps
if [ $DRY = 1 ]; then
  echo "cpu report.py"; echo "cpu gates.py --stage day1 > RUNS/gates_day1.md"; echo "cpu val_threshold.py > RUNS/val_threshold.md"
else
  echo "start report $(date +%T)"; $PY report.py > $R/logs/report.log 2>&1; echo "finished report rc=$?"
  $PY gates.py --stage day1 > $R/gates_day1.md 2> $R/logs/gates_day1.log; echo "finished gates rc=$?"
  $PY val_threshold.py > $R/val_threshold.md 2> $R/logs/val_threshold.log; echo "finished val_threshold rc=$?"
  echo "queue done $(date +%T) gpu_min=$((GPU_SECS / 60))"
fi

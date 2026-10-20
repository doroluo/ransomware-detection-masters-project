#!/bin/bash
# CPU runs of the teammates' GAT pipelines under our folds (experiments/team_pipelines): bash scripts/team_runs.sh [--dry]
# Three lanes run side by side, all on the CPU (--device cpu and CUDA_VISIBLE_DEVICES=-1, so the GPU queue is not touched):
#   gnn500  train_a --arm gnn500, seeds 0 1 2, folds 0-4 (seed by seed)
#   cb500   train_a --arm cb500 on the CodeBERT block vectors ours_cap0_codebert, seeds 0 1 2, folds 0-4
#   sig500  waits until RUNS/team/b_sig500/build_summary.json accounts for all 1810 binaries (processed + excluded == 1810:
#           the ids `build_b --exclude` left out on purpose count), then train_b --arm sig500, seeds 0 1 2, folds 0-4 (train_b
#           leaves the excluded ids out of every split). An incomplete summary (the build stopped early) ends the lane with a
#           message. `--check-build` prints whether the summary is complete, incomplete or absent, and exits.
# Each trainer uses TEAM_THREADS torch threads (default 3): the two A lanes take 6 of the 16 logical cores while the pipeline-B
# build (one process) and the GPU queue's loaders keep the rest; the sig500 lane starts only when the build is done.
# A run is skipped when RUNS/<name>/metrics.json exists. RUNS/STOP_TEAM (a file of its own: it never stops the GPU queue, whose
# file is RUNS/STOP) makes every lane log "stopped before <name>" and skip. A run that exits non-zero is retried once ("retry
# <name>", the first log kept as <name>.log.try1), not when RUNS/STOP_TEAM exists. Per-run logs: RUNS/logs/<name>.log. The names
# are RunConfig.name of experiments.team_pipelines.common.team_config (tests/test_team_pipelines.py checks it).
# --dry prints "run <name> <module args>" for every run in lane order (gnn500, cb500, sig500) and writes nothing.
# --v2 runs the one lane sig500v2 instead of the three: train_b --arm sig500v2 on RUNS/team/b_sig500_v2, the rebuild with Yanping's
#   streaming preprocess (build_b --source main-c7899f0), seeds 0 1 2 x folds 0-4 (15 runs `kfold{f}_x86_gnn_s{seed}_team-sig500v2`).
#   It waits for that build's summary to report all 1810 binaries processed and none excluded (an excluded id or a build that
#   ended early stops the lane with a message). The other flags work as without it.
# GT_PY / GT_RUNS override the interpreter / runs directory.
cd "$(dirname "$0")/.."
PY=${GT_PY:-/c/Users/chaoa/Downloads/gt_venv/Scripts/python.exe}
R=${GT_RUNS:-/c/Users/chaoa/Downloads/graph_transformer_runs}
STOP=$R/STOP_TEAM
DRY=0; V2=0; CHECK=0
for a in "$@"; do case $a in --dry) DRY=1;; --v2) V2=1;; --check-build) CHECK=1;; esac; done
ARM=sig500; BUILD=$R/team/b_sig500
[ $V2 = 1 ] && { ARM=sig500v2; BUILD=$R/team/b_sig500_v2; }
export TEAM_THREADS=${TEAM_THREADS:-3} CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1   # unbuffered: the per-run log shows each epoch as it ends
[ $DRY = 0 ] && mkdir -p $R/logs

run() { name=$1; shift
  if [ $DRY = 1 ]; then echo "run $name $*"; return; fi
  [ -f $R/$name/metrics.json ] && { echo "skip $name"; return; }
  [ -f $STOP ] && { echo "stopped before $name"; return; }
  echo "start $name $(date +%T)"
  $PY -m "$@" > $R/logs/$name.log 2>&1; rc=$?
  if [ $rc != 0 ] && [ ! -f $STOP ]; then echo "retry $name"; mv -f $R/logs/$name.log $R/logs/$name.log.try1
    $PY -m "$@" > $R/logs/$name.log 2>&1; rc=$?; fi
  echo "finished $name rc=$rc $(date +%T)"; }

# lane <arm> <trainer module> <extra args>: seeds 0 1 2, folds 0-4
lane() { arm=$1; mod=$2; shift; shift
  for s in 0 1 2; do for f in 0 1 2 3 4; do
    run kfold${f}_x86_gnn_s${s}_team-${arm} experiments.team_pipelines.$mod --fold $f --seed $s --arm $arm --device cpu "$@"
  done; done; }

# build_done: 0 when the build summary accounts for all 1810 binaries (total 1810, processed + excluded == 1810: build_b --exclude
# lists the ids it left out on purpose; with --v2 nothing may be excluded); 2 when it says less (the build ended early);
# 1 while it is absent or unreadable
build_done() { f=$BUILD/build_summary.json; [ -f $f ] || return 1
  $PY -c "import json,sys; s=json.load(open(sys.argv[1])); n = len(s.get('excluded', {})); sys.exit(0 if s['total'] == 1810 and s['processed'] + n == 1810 and (n == 0 or sys.argv[2] == '0') else 2)" $f $V2
  return $?; }

if [ $CHECK = 1 ]; then build_done; rc=$?
  case $rc in 0) echo "build complete";; 2) echo "build incomplete";; *) echo "build summary absent";; esac; exit 0; fi

sig_lane() {
  if [ $DRY = 1 ]; then
    if [ $V2 = 1 ]; then echo "wait $BUILD/build_summary.json reports processed == 1810/1810, none excluded"
    else echo "wait $BUILD/build_summary.json reports processed + excluded == 1810/1810"; fi
  else echo "$ARM lane waiting for the pipeline-B build $(date +%T)"
    while true; do
      [ -f $STOP ] && { echo "stopped while waiting for the build"; return; }
      build_done; rc=$?
      [ $rc = 0 ] && break
      [ $rc = 2 ] && { echo "build_summary.json is incomplete: $(tr -d '\n ' < $BUILD/build_summary.json | cut -c1-120); not running $ARM"; return; }
      sleep 120
    done
    echo "pipeline-B build complete $(date +%T)"; fi
  lane $ARM train_b; }

echo "team_runs start $(date +%T) TEAM_THREADS=$TEAM_THREADS dry=$DRY"
if [ $V2 = 1 ]; then sig_lane
elif [ $DRY = 1 ]; then lane gnn500 train_a; lane cb500 train_a --block-vec ours_cap0_codebert; sig_lane
else
  lane gnn500 train_a &
  lane cb500 train_a --block-vec ours_cap0_codebert &
  sig_lane &
  wait
fi
echo "team_runs done $(date +%T)"

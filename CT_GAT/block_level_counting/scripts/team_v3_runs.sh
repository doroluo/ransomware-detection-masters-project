#!/bin/bash
# GPU lane of the pipeline-B arms on the fixed preprocessing (v3): bash scripts/team_v3_runs.sh [--dry|--check-build]
# Waits until RUNS/team/b_sig500_v3/build_summary.json reports 1810 of 1810 binaries processed and none excluded (build_b --source
# main-fix), then runs on the GPU (--device cuda) train_b for the three arms, seeds 0 1 2, folds 0-4 (45 runs, seed by seed, the three
# arms of a seed one after the other, folds 0-4 inside an arm):
#   sig500v3       --features all  (64-d pooled embedding + 77 block features = 141-d, today's input)
#   sig500v3-feat  --features feat (77 block features + CFG: Model 2 of the team diagram)
#   sig500v3-emb   --features emb  (64-d pooled embedding + CFG: Model 3)
# names kfold<f>_x86_gnn_s<s>_team-<arm>; then gates.py --stage team > RUNS/gates_team.md.
# A build summary that says less than 1810 of 1810 or lists an excluded id (the build ended early) ends the lane with a message and
# runs nothing. A run is skipped when RUNS/<name>/metrics.json exists. RUNS/STOP (the GPU queue's file) makes the lane log
# "stopped before <name>" and skip, also while it waits for the build. A run that exits non-zero is retried once ("retry <name>",
# the first log kept as <name>.log.try1), not when RUNS/STOP exists. Per-run logs: RUNS/logs/<name>.log. The script exits 0 only when
# all 45 runs have a metrics.json and gates.py succeeded.
# --dry prints "run <name> <module args>" for the 45 runs and the closing step and writes nothing; tests/test_team_v3.py checks the
# names against RunConfig.name. --check-build prints complete, incomplete or absent and exits. GT_PY / GT_RUNS override the interpreter
# / runs directory; TEAM_THREADS (default 4) caps the CPU threads of each trainer.
cd "$(dirname "$0")/.."
PY=${GT_PY:-/c/Users/chaoa/Downloads/gt_venv/Scripts/python.exe}
R=${GT_RUNS:-/c/Users/chaoa/Downloads/graph_transformer_runs}
STOP=$R/STOP
BUILD=$R/team/b_sig500_v3
DRY=0; CHECK=0
for a in "$@"; do case $a in --dry) DRY=1;; --check-build) CHECK=1;; esac; done
export TEAM_THREADS=${TEAM_THREADS:-4} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
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

# build_done: 0 when the summary says total 1810, processed 1810 and nothing excluded; 2 when it says less or lists an exclusion
# (the build ended early); 1 while it is absent
build_done() { f=$BUILD/build_summary.json; [ -f $f ] || return 1
  $PY -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['total'] == 1810 and s['processed'] == 1810 and not s.get('excluded') else 2)" $f
  return $?; }

if [ $CHECK = 1 ]; then build_done; rc=$?
  case $rc in 0) echo "build complete";; 2) echo "build incomplete";; *) echo "build summary absent";; esac; exit 0; fi

# arm_runs <arm> <features>: seed s, folds 0-4
arm_runs() { arm=$1; feat=$2; s=$3
  for f in 0 1 2 3 4; do
    run kfold${f}_x86_gnn_s${s}_team-${arm} experiments.team_pipelines.train_b --fold $f --seed $s --arm $arm --features $feat --source main-fix --device cuda
  done; }

echo "team_v3_runs start $(date +%T) TEAM_THREADS=$TEAM_THREADS dry=$DRY"
if [ $DRY = 1 ]; then echo "wait $BUILD/build_summary.json reports processed == 1810/1810, none excluded"
else echo "sig500v3 lane waiting for the pipeline-B build $(date +%T)"
  while true; do
    [ -f $STOP ] && { echo "stopped while waiting for the build"; exit 0; }
    build_done; rc=$?
    [ $rc = 0 ] && break
    [ $rc = 2 ] && { echo "build_summary.json is incomplete: $(tr -d '\n ' < $BUILD/build_summary.json | cut -c1-120); not running"; exit 1; }
    sleep 120
  done
  echo "pipeline-B build complete $(date +%T)"; fi

for s in 0 1 2; do
  arm_runs sig500v3 all $s
  arm_runs sig500v3-feat feat $s
  arm_runs sig500v3-emb emb $s
done

if [ $DRY = 1 ]; then echo "cpu gates.py --stage team > RUNS/gates_team.md"; echo "team_v3_runs done $(date +%T)"; exit 0; fi
missing=0
for s in 0 1 2; do for f in 0 1 2 3 4; do for arm in sig500v3 sig500v3-feat sig500v3-emb; do
  [ -f $R/kfold${f}_x86_gnn_s${s}_team-${arm}/metrics.json ] || missing=$((missing + 1))
done; done; done
echo "start gates team $(date +%T)"
$PY gates.py --stage team > $R/gates_team.md 2> $R/logs/gates_team.log; grc=$?
echo "finished gates team rc=$grc, $missing of 45 runs without metrics.json $(date +%T)"
echo "team_v3_runs done $(date +%T)"
[ $missing = 0 ] && [ $grc = 0 ]

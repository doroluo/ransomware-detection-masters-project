#!/bin/bash
# Resumable GNN run queue: bash queue_gnn.sh --stage day1|day1b|v3|subgraph|paths|vocab|imports|imports_header|paths_codebert [--dry]
# A run is skipped when RUNS/<name>/metrics.json exists. GPU_BUDGET_MIN (default 290 for day1, 600 for the other stages) caps the
# cumulative GPU wall time: once it is spent, remaining GPU runs are logged as "skipped (budget)" and only the
# CPU/report steps still run. If RUNS/STOP exists before a run (stop_queue.sh creates it), the run is logged as
# "stopped before <name>" and skipped; a stale RUNS/STOP is removed at start. --dry prints what would run
# (DRY_RUN_MIN=<min> charges that many minutes per run to try the guard).
# GT_PY / GT_RUNS override the interpreter / runs directory, GT_TRAIN_PATTERN the regex that finds a live train.py
# (testing). Every python call runs with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the GPU is shared). A run
# that exits non-zero is retried once, logged as "retry <name>" (the first log is kept as <name>.log.try1); no retry
# when RUNS/STOP exists, so stop_queue.sh --now does not restart the run it killed. Stage v3 (kfold_v3 on the combined
# cache, folds 0-4, seed 0, 15% block masking and cell-balanced checkpoint selection, --select-on cell) exports GT_CACHE
# for every python call and runs, in order: the CPU baselines (all, matched, all arch-balanced), the arch-balanced
# all-arch arms (--arch-balance, names contain -ab; the main result) R2 no-edges, R7 typed, R4 shuffle, the same three
# unweighted, then the matched arms. Stage subgraph (kfold x86, v2 cache, seed 0, 15% block masking) measures the chain
# reduction (RUNS/subgraph_chain.json, CPU), then runs R7 typed with the cap and lcc subgraph policies at 20k and 5k blocks
# and the no-edge arm with the raw, presence and rawlog block weightings. Stage paths (kfold x86, v2 cache, seed 0, 15% block
# masking) writes the out-of-fold TF-IDF logits (CPU), runs R7 with --stack-tfidf on the five folds, then builds the reference
# encoder's block vectors (the block export RUNS/export/ours_cap0, reference checkpoints RUNS/encoders/ref and the vectors of
# fold 0, each skipped when it exists) and runs path 1 (no-edge GNN on block vectors) and path 2 (typed GNN on block vectors)
# on fold 0 only, as a plumbing check. The real encoder arms are added when the teammates' checkpoints arrive.
# Stage paths_codebert (kfold x86, v2 cache, 15% block masking, seeds 0 1 2, folds 0-4: 30 runs) builds the CodeBERT checkpoints
# (RUNS/encoders/codebert, the block text is the vocabulary strings of the export, transformer_paths/codebert_encoder.py) and
# their block vectors (<cache>/side/blockvec/ours_cap0_codebert, GPU, HF_HOME defaults to C:/Users/chaoa/Downloads/hf_home),
# each skipped when it exists, then runs path 1 (no-edge GNN on the vectors) and path 2 (typed GNN on them), seed by seed.
# The validation split follows --seed, as in the token arms R2 and R7 that the gates pair them with.
# Stage vocab (kfold x86, v2 cache, seed 0; needs <cache>/side/yanping/) runs the TF-IDF baselines over the reader's
# vocabulary and the hybrid one (CPU, directories kfold<f>_x86_s0_voc-<name>), then the no-edge arm with 15% block masking
# under --vocab yanping and --vocab hybrid (ours already exists as gnn-noedges-mblk15). Stage imports (needs
# <cache>/side/imports/) runs the CPU import baselines (baselines.py --imports, directories kfold<f>_x86_s0_imp) and then
# asks `gates.py --stage imports --verdict`: R7 with 15% block masking and --imports unreferenced (gnn-mblk15-imp) runs only
# when the pooled AUC of imports_unref_lr is at least the majority floor + 0.05; the verdict is printed either way.
# Stage imports_header (needs <cache>/side/imports/) runs R7 typed with 15% block masking and --imports header (gnn-mblk15-imph)
# on kfold x86, seed 0, folds 0-4, with no gate; its summary is `gates.py --stage imports` (R7, header-imports LR and TF-IDF against
# R7 + header imports), written to RUNS/gates_imports_header.md. The baselines come from stage imports.
cd "$(dirname "$0")/.."
PY=/c/Users/chaoa/Downloads/gt_venv/Scripts/python.exe
R=/c/Users/chaoa/Downloads/graph_transformer_runs
PY=${GT_PY:-$PY}; R=${GT_RUNS:-$R}; PAT=${GT_TRAIN_PATTERN:-train[.]py}
STAGE=; DRY=0
while [ $# -gt 0 ]; do
  case $1 in --stage) STAGE=$2; shift;; --dry) DRY=1;; *) STAGE=;; esac; shift
done
[ "$STAGE" = day1 ] || [ "$STAGE" = day1b ] || [ "$STAGE" = v3 ] || [ "$STAGE" = subgraph ] || [ "$STAGE" = paths ] || [ "$STAGE" = paths_codebert ] || [ "$STAGE" = vocab ] || [ "$STAGE" = imports ] || [ "$STAGE" = imports_header ] || { echo "usage: bash queue_gnn.sh --stage day1|day1b|v3|subgraph|paths|vocab|imports|imports_header|paths_codebert [--dry]"; exit 2; }
GATES_STAGE=$STAGE; [ $STAGE = imports_header ] && GATES_STAGE=imports
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ $STAGE = v3 ] && export GT_CACHE=C:/Users/chaoa/Downloads/graph_transformer_cache_combined
if [ $STAGE = day1 ]; then GPU_BUDGET_MIN=${GPU_BUDGET_MIN:-290}; else GPU_BUDGET_MIN=${GPU_BUDGET_MIN:-600}; fi
DRY_RUN_MIN=${DRY_RUN_MIN:-0}
GPU_SECS=0

if [ $DRY = 0 ]; then
  mkdir -p $R/logs
  if ! powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$PAT' }) { exit 1 }"; then
    echo "a train.py process is alive; refusing to start"; exit 1
  fi
  $PY -c "import torch, sklearn, scipy, training.train, evaluation.stats, evaluation.gates, pipeline.batching, pipeline.model; assert torch.cuda.is_available(); print('preamble ok', torch.cuda.get_device_name(0))" || { echo "preamble failed"; exit 1; }
  rm -f $R/STOP
fi

run() { name=$1; shift
  if [ $DRY = 0 ] && [ -f $R/$name/metrics.json ]; then echo "skip $name"; return; fi
  if [ -f $R/STOP ]; then echo "stopped before $name"; return; fi
  if [ $GPU_SECS -ge $((GPU_BUDGET_MIN * 60)) ]; then echo "skipped (budget) $name"; return; fi
  if [ $DRY = 1 ]; then echo "run $name $*"; GPU_SECS=$((GPU_SECS + DRY_RUN_MIN * 60)); return; fi
  echo "start $name $(date +%T)"; t0=$SECONDS
  $PY train.py "$@" > $R/logs/$name.log 2>&1; rc=$?
  if [ $rc != 0 ] && [ ! -f $R/STOP ]; then echo "retry $name"; mv -f $R/logs/$name.log $R/logs/$name.log.try1
    $PY train.py "$@" > $R/logs/$name.log 2>&1; rc=$?; fi
  GPU_SECS=$((GPU_SECS + SECONDS - t0))
  echo "finished $name rc=$rc $(date +%T) gpu_min=$((GPU_SECS / 60))"; }

# arm <variant> <train.py args>: kfold x86 folds 0-4, seed 0
arm() { variant=$1; shift
  for f in 0 1 2 3 4; do run kfold${f}_x86_${variant}_s0 --protocol kfold --fold $f --arch x86 --seed 0 "$@"; done; }

# arm_v3 <arch> <variant> <train.py args>: kfold_v3 folds 0-4, seed 0
arm_v3() { arch=$1; variant=$2; shift; shift
  for f in 0 1 2 3 4; do run v3kfold${f}_${arch}_${variant}_s0 --protocol kfold_v3 --fold $f --arch $arch --seed 0 "$@"; done; }

# baselines_v3 <arch> [--arch-balance]: CPU baselines for kfold_v3 seed 0 (directories end in _ab with --arch-balance),
# skipped when all five folds already have validation thresholds
baselines_v3() { arch=$1; shift; sfx=; [ "$*" = --arch-balance ] && sfx=_ab
  BASE_ARGS="--protocol kfold_v3 --fold 0 1 2 3 4 --arch $arch --seed 0 $*"; BASE_ARGS=${BASE_ARGS% }
  if [ $DRY = 1 ]; then echo "cpu baselines $BASE_ARGS"
  elif [ "$(grep -l threshold_val $R/baselines/v3kfold[0-4]_${arch}_s0$sfx/metrics.json 2>/dev/null | wc -l)" = 5 ]; then echo "skip baselines v3 $arch$sfx"
  else echo "start baselines v3 $arch$sfx $(date +%T)"; $PY baselines.py $BASE_ARGS > $R/logs/baselines_v3kfold_${arch}_s0$sfx.log 2>&1; echo "finished baselines v3 $arch$sfx rc=$? $(date +%T)"; fi; }

# baselines <train.py-style suffix> <extra baselines.py args>: CPU baselines for kfold x86 seed 0, skipped when all five
# folds already have validation thresholds
baselines() { sfx=$1; shift
  BASE_ARGS="--protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 $*"
  if [ $DRY = 1 ]; then echo "cpu baselines $BASE_ARGS"
  elif [ "$(grep -l threshold_val $R/baselines/kfold[0-4]_x86_s0$sfx/metrics.json 2>/dev/null | wc -l)" = 5 ]; then echo "skip baselines$sfx"
  else echo "start baselines$sfx $(date +%T)"; $PY baselines.py $BASE_ARGS > $R/logs/baselines_kfold_x86_s0$sfx.log 2>&1; echo "finished baselines$sfx rc=$? $(date +%T)"; fi; }

# chain_measure: the chain-merge measurement over the kfold x86 pool (CPU), skipped when RUNS/subgraph_chain.json exists
chain_measure() {
  if [ $DRY = 1 ]; then echo "cpu subgraph_chain.py"
  elif [ -f $R/subgraph_chain.json ]; then echo "skip subgraph_chain"
  else echo "start subgraph_chain $(date +%T)"; $PY subgraph_chain.py > $R/logs/subgraph_chain.log 2>&1; echo "finished subgraph_chain rc=$? $(date +%T)"; fi; }

# baselines_stack: CPU out-of-fold TF-IDF logits for kfold x86 seed 0 (stack_logits.csv in the five baseline directories),
# skipped when all five exist
baselines_stack() {
  STACK_ARGS="--protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --stack-logits"
  if [ $DRY = 1 ]; then echo "cpu baselines $STACK_ARGS"
  elif [ "$(ls $R/baselines/kfold[0-4]_x86_s0/stack_logits.csv 2>/dev/null | wc -l)" = 5 ]; then echo "skip stack logits"
  else echo "start stack logits $(date +%T)"; $PY baselines.py $STACK_ARGS > $R/logs/stack_logits_kfold_x86_s0.log 2>&1; echo "finished stack logits rc=$? $(date +%T)"; fi; }

# cpu_step <skip file> <log name> <command...>: a CPU step of the paths stage, skipped when its output file exists
cpu_step() { f=$1; name=$2; shift; shift
  if [ $DRY = 1 ]; then echo "cpu ${*:3}"
  elif [ -f "$f" ]; then echo "skip $name"
  else echo "start $name $(date +%T)"; "$@" > $R/logs/$name.log 2>&1; echo "finished $name rc=$? $(date +%T)"; fi; }

# blockvec_ref: the block export, the reference encoder checkpoints and fold 0's block vectors, each skipped when it exists
blockvec_ref() { C=${GT_CACHE:-C:/Users/chaoa/Downloads/graph_transformer_cache_v2}
  cpu_step $R/export/ours_cap0/store/store.json block_export $PY -m transformer_paths.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0
  cpu_step $R/encoders/ref/fold4/best.pt reference_encoder $PY -m transformer_paths.reference_encoder --export ours_cap0 --name ref
  cpu_step $C/side/blockvec/ours_cap0_ref/fold0/meta.json precompute_blockvec $PY -m transformer_paths.precompute_blockvec --encoder-name ref --export ours_cap0 --folds 0; }

# codebert_vectors: the CodeBERT checkpoints and block vectors of the five folds (needs the block export), each skipped when
# it exists
codebert_vectors() { C=${GT_CACHE:-C:/Users/chaoa/Downloads/graph_transformer_cache_v2}
  export HF_HOME=${HF_HOME:-C:/Users/chaoa/Downloads/hf_home} HF_HUB_OFFLINE=1
  cpu_step $R/encoders/codebert/fold4/best.pt codebert_checkpoints $PY -m transformer_paths.codebert_encoder --write-checkpoints --export ours_cap0 --name codebert
  cpu_step $C/side/blockvec/ours_cap0_codebert/fold4/meta.json precompute_blockvec_codebert $PY -m transformer_paths.precompute_blockvec --encoder-name codebert --export ours_cap0 --folds 0 1 2 3 4; }

# arm_seed <seed> <variant> <train.py args>: kfold x86 folds 0-4 at one seed
arm_seed() { seed=$1; variant=$2; shift; shift
  for f in 0 1 2 3 4; do run kfold${f}_x86_${variant}_s${seed} --protocol kfold --fold $f --arch x86 --seed $seed "$@"; done; }

# imports_gate <train.py args of R7 + imports>: the pooled AUC of imports_unref_lr decides (gates.py --verdict exits 0 on
# a pass, 3 otherwise, also when the baselines are missing); the verdict line is printed and the arm runs only on a pass
imports_gate() { variant=$1; shift
  if [ $DRY = 1 ]; then echo "cpu gates.py --stage imports --verdict (the arm below runs only on PASS)"; arm $variant "$@"; return; fi
  verdict=$($PY gates.py --stage imports --verdict 2>> $R/logs/gate_imports.log); rc=$?
  echo "$verdict"
  if [ $rc = 0 ]; then arm $variant "$@"; else echo "gate not passed: skipping kfold<f>_x86_${variant}_s0"; fi; }

if [ $STAGE = vocab ]; then
V="--model gnn --mask-blocks 0.15 --control no_edges"
baselines _voc-yanping --vocab yanping                                # 1. CPU TF-IDF over the reader's tokens
baselines _voc-hybrid --vocab hybrid
arm gnn-noedges-mblk15-voc-yanping $V --vocab yanping                 # 2. R2 no-edges, reader vocabulary
arm gnn-noedges-mblk15-voc-hybrid $V --vocab hybrid                   # 3. R2 no-edges, hybrid vocabulary
elif [ $STAGE = imports ]; then
baselines _imp --imports                                              # 1. CPU import baselines
imports_gate gnn-mblk15-imp --model gnn --mask-blocks 0.15 --imports unreferenced  # 2. R7 + unreferenced imports, if the gate passes
elif [ $STAGE = imports_header ]; then
arm gnn-mblk15-imph --model gnn --mask-blocks 0.15 --imports header   # 1. R7 + header imports, no gate
elif [ $STAGE = paths ]; then
baselines_stack                                                       # 1. CPU stack logits
arm gnn-stack-mblk15 --model gnn --mask-blocks 0.15 --stack-tfidf     # 2. stack: R7 + out-of-fold TF-IDF logit
blockvec_ref                                                          # 3. reference-encoder block vectors, fold 0
V="--model gnn --mask-blocks 0.15 --val-seed 0 --block-input vec --block-vec ours_cap0_ref"
run kfold0_x86_gnn-noedges-mblk15-bv-ours_cap0_ref_s0 --protocol kfold --fold 0 --arch x86 --seed 0 $V --control no_edges  # 4. path 1
run kfold0_x86_gnn-mblk15-bv-ours_cap0_ref_s0 --protocol kfold --fold 0 --arch x86 --seed 0 $V                            # 5. path 2
elif [ $STAGE = paths_codebert ]; then
codebert_vectors                                                      # 1. CodeBERT checkpoints and block vectors, five folds
V="--model gnn --mask-blocks 0.15 --block-input vec --block-vec ours_cap0_codebert"
for s in 0 1 2; do
arm_seed $s gnn-noedges-mblk15-bv-ours_cap0_codebert $V --control no_edges  # 2. path 1, seed s
arm_seed $s gnn-mblk15-bv-ours_cap0_codebert $V                       # 3. path 2, seed s
done
elif [ $STAGE = v3 ]; then
M="--model gnn --mask-blocks 0.15 --select-on cell"
baselines_v3 all                                                      # 1. CPU baselines
baselines_v3 matched
baselines_v3 all --arch-balance
arm_v3 all gnn-noedges-mblk15-ab-selcell $M --control no_edges --arch-balance  # 2. arch-balanced, arch all: R2
arm_v3 all gnn-mblk15-ab-selcell $M --arch-balance                    # 3. R7
arm_v3 all gnn-shuffle-mblk15-ab-selcell $M --control shuffle --arch-balance  # 4. R4
arm_v3 all gnn-noedges-mblk15-selcell $M --control no_edges           # 5. unweighted, arch all: R2
arm_v3 all gnn-mblk15-selcell $M                                      # 6. R7
arm_v3 all gnn-shuffle-mblk15-selcell $M --control shuffle            # 7. R4
arm_v3 matched gnn-noedges-mblk15-selcell $M --control no_edges       # 8. arch matched: R2
arm_v3 matched gnn-mblk15-selcell $M                                  # 9. R7
arm_v3 matched gnn-shuffle-mblk15-selcell $M --control shuffle        # 10. R4
elif [ $STAGE = subgraph ]; then
chain_measure                                                         # 1. chain reduction, CPU
M="--model gnn --mask-blocks 0.15"
arm gnn-mblk15-sg-cap20k $M --subgraph cap --budget 20000            # 2. R7, cap order at 20k and 5k blocks
arm gnn-mblk15-sg-cap5k $M --subgraph cap --budget 5000
arm gnn-mblk15-sg-lcc20k $M --subgraph lcc --budget 20000            # 3. R7, call-graph components
arm gnn-mblk15-sg-lcc5k $M --subgraph lcc --budget 5000
arm gnn-noedges-mblk15-enc-raw $M --control no_edges --encoder raw    # 4. R2 weightings
arm gnn-noedges-mblk15-enc-presence $M --control no_edges --encoder presence
arm gnn-noedges-mblk15-enc-rawlog $M --control no_edges --encoder rawlog
elif [ $STAGE = day1b ]; then
baselines _tf50 --train-fraction 0.5                                  # 1. learning-curve TF-IDF
baselines _tf25 --train-fraction 0.25
arm gnn-mtok15 --model gnn --mask-tokens 0.15                         # 2.
arm gnn-dfn5-dlib40 --model gnn --drop-functions 0.05 --drop-library 0.4  # 3.
arm gnn-undir --model gnn --direction undirected                      # 4.
arm gnn-fwd --model gnn --direction forward                           # 5.
arm gnn-dfn15 --model gnn --drop-functions 0.15                       # 6.
arm gnn-tf50 --model gnn --train-fraction 0.5                         # 7. learning curve
arm gnn-tf25 --model gnn --train-fraction 0.25                        # 8.
arm gnn-mblk15 --model gnn --mask-blocks 0.15                         # 9.
arm gnn-dedge20 --model gnn --drop-edges 0.2                          # 10.
arm gnn-flip20 --model gnn --flip-edges 0.2                           # 11.
arm gnn-mtok50 --model gnn --mask-tokens 0.5                          # 12.
arm gnn-bwd --model gnn --direction backward                          # 13.
arm gnn-L8 --model gnn --gnn-fn-layers 8                              # 14.
else
# 1. CPU baselines (with validation predictions and thresholds); no GPU
baselines ""
arm gnn-noedges --model gnn --control no_edges                       # 2. R2 tfidf
arm gnn-noedges-enc-mean --model gnn --control no_edges --encoder mean  # 3. R2 mean
arm gnn --model gnn                                                  # 4. R7
arm gnn-untyped --model gnn --untyped                                # 5. R5
arm gnn-shuffle --model gnn --control shuffle                        # 6. R4
run kfold3_x86_nograph_s2 --protocol kfold --fold 3 --arch x86 --seed 2 --no-graph   # 7. transformer completions
run kfold4_x86_graph_s2 --protocol kfold --fold 4 --arch x86 --seed 2
run kfold4_x86_nograph_s2 --protocol kfold --fold 4 --arch x86 --seed 2 --no-graph
fi

# 8. report steps
if [ $DRY = 1 ]; then
  echo "cpu report.py"; echo "cpu gates.py --stage $GATES_STAGE > RUNS/gates_$STAGE.md"; echo "cpu val_threshold.py > RUNS/val_threshold.md"
else
  echo "start report $(date +%T)"; $PY report.py > $R/logs/report.log 2>&1; echo "finished report rc=$?"
  $PY gates.py --stage $GATES_STAGE > $R/gates_$STAGE.md 2> $R/logs/gates_$STAGE.log; echo "finished gates rc=$?"
  $PY val_threshold.py > $R/val_threshold.md 2> $R/logs/val_threshold.log; echo "finished val_threshold rc=$?"
  echo "queue done $(date +%T) gpu_min=$((GPU_SECS / 60))"
fi

#!/usr/bin/env bash
# run_v3_cpu.sh - the four CPU models on the dataset-v3 folds (MalwareBazaar
# batches 1-3), joint pool and arch-matched selection, skipping anything that
# already has a metrics.json. Start it from Task Scheduler so it outlives the
# Claude session (see tools/resume_v2_runs.sh for the recipe).
#
#   bash tools/run_v3_cpu.sh /path/to/log/dir
set -u
LOG="${1:-$HOME/v3_logs}"; mkdir -p "$LOG"
cd "$(dirname "$0")/.."
export RANSOM_FH_DIR="$PWD/results/family_holdout_v3"
R="$RANSOM_FH_DIR"
export RANSOM_IMPORTS_FLAT="$PWD/manifests/imports/imports_flat_v3.json"   # the batch-3 merge; imports_flat.json stays the v2 table
PY314=python
PY312="C:/Users/chaoa/Downloads/Tokenization-Testing-for-Malware-Data/.venv/Scripts/python.exe"
have() { [ -f "$R/$1/$2/metrics.json" ]; }
for ds in all arch_matched; do
  have "$ds" tfidf/LinearSVC || { nohup $PY314 family_holdout/run_tfidf.py --dataset "$ds" --out "$R" > "$LOG/tfidf_$ds.log" 2>&1 & }
  have "$ds" imports_baseline/names_tfidf_LogReg || { nohup $PY314 family_holdout/run_imports_baseline.py --dataset "$ds" --out "$R" > "$LOG/imports_$ds.log" 2>&1 & }
  have "$ds" graph2vec/tuned_best_sparse_histogram_oofthr_imports || { nohup $PY314 family_holdout/run_graph2vec.py --dataset "$ds" --out "$R" --imports > "$LOG/g2v_$ds.log" 2>&1 & }
  have "$ds" tokenization/SVM-RBF_SW_w2v_imports || { nohup "$PY312" -u family_holdout/run_tokenization.py --dataset "$ds" --out "$R" --imports > "$LOG/tok_$ds.log" 2>&1 & }
done
echo "launched; logs under $LOG"
wait
echo "== all v3 CPU runs finished $(date)"

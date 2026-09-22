#!/usr/bin/env bash
# resume_v2_runs.sh - relaunch the dataset-v2 evaluation runs after a host or
# session restart. Every run either resumes from what is on disk or restarts
# only the configurations that have no result directory yet.
#
#   bash tools/resume_v2_runs.sh /path/to/log/dir
#
# To make the runs independent of the Claude session (they die with it
# otherwise), start it from Task Scheduler:
#   schtasks /create /tn ransom_v2_resume /sc once /st 23:59 /f /tr '"C:\Program Files\Gitinash.exe" -lc "cd /c/Users/chaoa/Downloads/rdmp-llm && bash tools/resume_v2_runs.sh /c/Users/chaoa/Downloads/v2_logs > /c/Users/chaoa/Downloads/v2_logs/resume.out 2>&1"'
#   schtasks /run /tn ransom_v2_resume
#
# Finished results live under results/family_holdout_v2/<dataset>/<pipeline>/
# and are never recomputed. GPU jobs run one at a time (the sequence
# transformer needs ~15.6 GB alone; sharing the card ends in CUDA OOM).
set -u
LOG="${1:-$HOME/v2_logs}"; mkdir -p "$LOG"
cd "$(dirname "$0")/.."
REPO="$PWD"
export RANSOM_FH_DIR="$REPO/results/family_holdout_v2"
R="$RANSOM_FH_DIR"
PY314=python
PY312="C:/Users/chaoa/Downloads/Tokenization-Testing-for-Malware-Data/.venv/Scripts/python.exe"
PYCUDA="C:/Users/chaoa/Downloads/cnn_vit_venv/Scripts/python.exe"
D=/c/Users/chaoa/Downloads

have() { [ -f "$R/$1/$2/metrics.json" ]; }   # dataset, pipeline/model dir

# --- CPU runs (each in the background) --------------------------------------
for ds in all arch_matched; do
  for m in LogReg LinearSVC; do have "$ds" "tfidf/$m" || { nohup $PY314 family_holdout/run_tfidf.py --dataset "$ds" --out "$R" > "$LOG/tfidf_$ds.log" 2>&1 & break; }; done
  have "$ds" imports_baseline/mnem+names_LogReg || { nohup $PY314 family_holdout/run_imports_baseline.py --dataset "$ds" --out "$R" > "$LOG/imports_$ds.log" 2>&1 & }
  have "$ds" graph2vec/tuned_best_sparse_histogram_oofthr_imports || { nohup $PY314 family_holdout/run_graph2vec.py --dataset "$ds" --out "$R" --imports > "$LOG/g2v_$ds.log" 2>&1 & }
  combos=""
  have "$ds" tokenization/RF_WPC_w2v_imports   || combos="$combos RF/WPC"
  have "$ds" tokenization/MLP_WP_w2v_imports   || combos="$combos MLP/WP"
  have "$ds" tokenization/SVM-RBF_SW_w2v_imports || combos="$combos SVM-RBF/SW"
  [ -n "$combos" ] && { nohup "$PY312" -u family_holdout/run_tokenization.py --dataset "$ds" --out "$R" --imports --combos "$(echo $combos | tr ' ' ',')" > "$LOG/tok_$ds.log" 2>&1 & }
done

# --- GPU chain (sequential) --------------------------------------------------
export RANSOM_SEQ_MODELS="C:/Users/chaoa/Downloads/seq_models_v2"
export RANSOM_SEQ_CACHE="C:/Users/chaoa/Downloads/seq_models_v2/token_cache"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
SEQCFG="$LOG/seq_config_v2.yaml"
[ -f "$SEQCFG" ] || sed 's#C:/Users/chaoa/Downloads/seq_models#C:/Users/chaoa/Downloads/seq_models_v2#g' seq_model/config.yaml > "$SEQCFG"
seqrun() {
  "$PYCUDA" -u seq_model/run_family_holdout.py --dataset all --config "$SEQCFG" --out "$R" --imports manifests/imports/imports_flat.json --log "$RANSOM_SEQ_MODELS/logs/fh_all.log" "$@" >> "$LOG/seq_all.log" 2>&1
  echo "== seq ($*) ended rc=$? $(date)" >> "$LOG/seq_all.log"
}
(
  for a in 1 2 3; do seqrun --scheme kfold; tail -1 "$LOG/seq_all.log" | grep -q "rc=0" && break; done
  "$PYCUDA" -u family_holdout/run_cnn_vit.py --dataset all --out-root "$R" --data-root "$D/cnn_vit_data/family_holdout_v2" --models-root "$D/cnn_vit_models/family_holdout_v2" >> "$LOG/cnnvit_all.log" 2>&1
  for a in 1 2 3; do seqrun; tail -1 "$LOG/seq_all.log" | grep -q "rc=0" && break; done
  echo "== gpu chain done $(date)" >> "$LOG/seq_all.log"
) &
echo "launched; logs under $LOG"
# stay alive until every run has finished, so a Task Scheduler task that
# started this script keeps owning the processes
wait
echo "== all v2 runs finished $(date)"

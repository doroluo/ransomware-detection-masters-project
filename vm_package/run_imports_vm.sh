#!/usr/bin/env bash
# run_imports_vm.sh - step 4b of vm_package/README.md, end to end, on the VM.
#
#   unzip vm_package.zip -d ~/vm_package && cd ~/vm_package
#   bash vm_package/run_imports_vm.sh
#
# Parses the import and delay-load directories of every PE in the four
# Mendeley folders with pefile (read-only; nothing is executed), writes one
# JSON + one CSV per folder under ~/imports, checks the counts, and bundles
# the eight files into ~/imports_out.tar.gz for copying back to the host.
# Needs only python3 and pefile (installed by step 2). Seconds per folder.
set -euo pipefail

DL="${DL:-$HOME/Downloads}"           # where the four corpus folders live
OUT="${OUT:-$HOME/imports}"
mkdir -p "$OUT"

python3 -c "import pefile" 2>/dev/null || {
  echo "pefile missing: python3 -m pip install --no-index --find-links asm_tool/wheels/cpXY -r asm_tool/requirements-vm.txt" >&2
  exit 1
}

run() {  # name  folder  expected_files
  local name="$1" dir="$2" expect="$3"
  if [ ! -d "$dir" ]; then echo "missing folder: $dir" >&2; exit 1; fi
  echo "== $name  ($dir)"
  python3 imports/extract_imports.py --in "$dir" \
      --out "$OUT/$name.json" --manifest "$OUT/$name.csv"
  local n
  n=$(($(wc -l < "$OUT/$name.csv") - 1))
  echo "   manifest rows: $n (expected about $expect)"
  python3 - "$OUT/$name.json" "$OUT/$name.csv" <<'PY'
import csv, json, sys, collections
j = json.load(open(sys.argv[1], encoding="utf-8"))
rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
st = collections.Counter(r["status"] for r in rows)
err = sum(1 for v in j.values() if v.get("error"))
print(f"   unique PEs in json: {len(j)}   status: {dict(st)}   parse errors: {err}")
PY
}

run mendeley_mal_train  "$DL/Ransomware_Training/rans"    1023
run mendeley_mal_test   "$DL/Ransomware_Test/rans_test"    385
run mendeley_good_train "$DL/Goodware_Training/goodware"  1134
run mendeley_good_test  "$DL/Goodware_Test/goodware_test"  133

# an .iat.json sibling appears only if a corpus crossed the 50 MB spill size
tar -C "$OUT" -czf "$HOME/imports_out.tar.gz" $(cd "$OUT" && ls mendeley_*.json mendeley_*.csv mendeley_*.iat.json 2>/dev/null)
echo
echo "wrote $HOME/imports_out.tar.gz:"
tar -tzvf "$HOME/imports_out.tar.gz"
echo
echo "copy imports_out.tar.gz to the host and extract it into"
echo "  C:/Users/chaoa/Downloads/rdmp-llm/manifests/imports/"

# IDA CFG extraction and desktop outputs

**Superseded configuration:** the mnemonic-only full-run supervisor described
below was stopped before it began full extraction. The active quality-first
pipeline is documented in `IDA_CFG_QUALITY_REVIEW.md`; its outputs are under
`reports/ida_cfg_quality_20261006/full/`. This document remains as the record of
the original pilots and schema. Do not resume the old controller for new data.

Entry point: `preprocessing/vm_ida_cfg.py`. This is a separate implementation;
the pre-existing `export_ida_cfg.py`, its tests and README edits are preserved.
The VM has IDA Pro 9.4 with a working idalib binding and Python 3.8.5. No package
installation is needed. Only static analysis is used, and no sample is executed.

## Output layout and joins

Windows destination: `reports/ida_cfg_20261006/` in this repository.
`pilot/` contains the first verified small run; `x64_pilot/` contains two
additional verified x64 cases; `full/` receives the full run.
Under either destination:

```
cfg/<sha256>.json.gz                    # one graph per unique file content
groups/
  mendeley/goodware/train/<sha256>.json.gz
  mendeley/goodware/test/<sha256>.json.gz
  mendeley/ransomware/train/<family>/<sha256>.json.gz
  mendeley/ransomware/test/<family>/<sha256>.json.gz
  balanced/goodware/<original-subfolders>/<sha256>.json.gz
  vs/ransomware/<family>/<sha256>.json.gz
  virusshare/<sha256>.json.gz
manifest.csv                           # every inventoried file, including failures
mendeley_samples.csv                   # all Mendeley files, original train/test
inventory.json                         # complete file metadata and matched cohort rows
cohort_membership.json                 # every original CSV row, unmodified fields
split_audit.json                       # duplicate-hash train/test and label conflicts
dataset_audit.json                     # CSV/hash-reference match and missing counts
summary.json                           # file and unique-graph counts, failure reasons
checksums.json                         # SHA-256 of every transferred payload file
```

Each leaf group folder also has a `manifest.csv`. Graphs shared across folders
are hard links on filesystems that support them; otherwise they are copies.
Names of grouping directories use safe characters; `relative_path` retains
the exact original path. Canonical graph files contain no labels or family names.

`mendeley_samples.csv` preserves `original_split`, `set`, `group`, `family`,
`sha256`, `relative_path`, `source`, `size`, detected `arch`, input `kind`, all
cohort metadata flattened as `cohort_*`, `in_cohort`, `exclude_reason`, extraction
status, truncation and graph path. `cohort_membership.json` preserves every
original CSV row including its CSV filename; `inventory.json` retains all
matching rows if there are multiple memberships. Excluded rows are retained.

For later architecture-specific experiments, select rows by `arch`, graph
status and the relevant cohort tags, then create separate experiment split
CSVs keyed by SHA-256. Keep `original_split` unchanged. Check `split_audit.json`
before training: identical hashes must not occur on both sides of a new split.
Fit vocabulary/normalization on training hashes only. Metadata columns must not
become graph features. VirusShare `label` means `is_ransomware` from its source
CSV: `0` does **not** mean benign. Missing labels remain blank.

## Graph schema and limits

`schema = ida-cfg-safe/1`. Functions and blocks have ordinal IDs; block IDs are
local to their function. Each block has ordered `mnemonics` and aligned
`operand_types` lists (IDA numeric categories, no operand values). Function
edges are directed IDA successors with external blocks excluded. `calls`
records block/instruction positions and recovered internal target function
IDs. Empty targets mean external, indirect, unresolved, or outside a capped
graph; they are not proof of no callees. Library/thunk flags remain available.

There are no bytes, addresses, operand values, symbol names, extracted strings,
or full assembly listings. Only gzip JSON graphs and explicitly selected
derived metadata are packaged. Logs, IDA databases and analysis copies remain
in the VM and are never added to the package.

IDA auto-analysis precedes extraction. Static recovery can miss packed code,
indirect transfers, or functions. No unpacking or dynamic analysis is performed.
Managed PE, unsupported architectures, malformed/non-PE inputs and dataset
support files are inventoried with explicit reasons, without pretending they
have native CFGs. Native PE candidates are attempted even if a cohort excluded
them; filter on the preserved cohort flags for ML.

Each worker has 300 seconds and a 2,500 MiB address-space limit. At most two
workers run. Caps of 50,000 blocks and 1,000,000 instructions per sample set
`truncated=true`. An empty graph is a failure. Inputs are copied privately to
the task's `scratch/`, checked against the original SHA-256 and made non-executable;
temporary copies and databases are removed when the worker finishes. A killed
supervisor may leave scratch remnants inside the VM; they must never be copied out.
A 3 GiB disk reserve stops new work; packaging also checks available space.

## Execution and recovery

VM task folder: `/home/seed/work/out/ida_cfg_20261006`.
All commands below run **inside the VM**, through the documented SSH key with
`BatchMode=yes`. Do not run another parent while one is active.

```sh
python3 ~/work/out/ida_cfg_20261006/vm_ida_cfg.py summary
# Resume: validated successes and recorded failures are skipped by default.
nohup python3 -u ~/work/out/ida_cfg_20261006/vm_ida_cfg.py run \
  > ~/work/out/ida_cfg_20261006/resume.log 2>&1 < /dev/null &
# After it finishes, validate and package:
python3 ~/work/out/ida_cfg_20261006/vm_ida_cfg.py package
```

Use `--retry-failures` only for a deliberate retry after diagnosing the reason.
Do not regenerate inventory while extraction is running. `status/<sha256>.json`
records outcome, timing, counts and truncation; matching worker logs stay in
`logs/`. `batch.log`, `pipeline_state.json`, `run_state.json`, `summary.json`
and (on failure) `pipeline_failure.json` report progress.

`run_vm_cfg.py` waits up to three hours for the initial inventory, runs the
full extraction, then packages it. It is launched under `nohup`. The Windows
`receive_vm_cfg.ps1` process polls for the checksum receipt, downloads only the
explicit safe package, verifies its checksum, validates each graph and checksum,
then materializes the group folders. It runs hidden, waits at most seven days,
and writes `full/transfer_status.json`. The VM and Windows host must remain on.
If it stops, rerun the receiver; it never opens or transfers sample paths.

```powershell
& .\CT_GAT\preprocessing\receive_vm_cfg.ps1
```

## Verification evidence

- Eight regression tests pass on Windows Python 3.14 and VM Python 3.8.
- Tests reject raw assembly fields, invalid graph edges/call targets, malformed
  instruction alignment, archive traversal/symlinks and checksum mismatches.
- The real IDA pilot attempted 13 unique native PE files across all groups:
  12 valid CFGs, 1 `no_recovered_instructions`, no truncation.
- Pilot inventory includes 23 files: 14 native-PE occurrences (one duplicate)
  and nine managed PE files. These are selection-stage counts, not full counts.
- The 12 pilot graphs were packaged, transferred by SCP, checksum-verified and
  schema-validated on Windows; group/family/split folders were materialized.
- Two additional native x64 samples passed real IDA extraction and desktop
  transfer validation: 14 verified pilot CFG files total, one empty-graph failure.
- The actual `receive_vm_cfg.ps1` receiver passed both pilot transfers. A hidden
  receiver process waits for the full package; its status file is in `full/`.
- Full run counts are live in `summary.json`; a started background job does not
  establish full completion. Final transfer is confirmed only by
  `full/transfer_status.json` with `state=complete`.

IDA CFG API semantics: https://python.docs.hex-rays.com/ida_gdl/index.html

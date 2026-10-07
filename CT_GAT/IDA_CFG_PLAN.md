# Static IDA CFG extraction plan

The initial minimal-feature plan below is superseded by
`IDA_CFG_QUALITY_REVIEW.md` following the user's request to prioritize ML
dataset usefulness over extraction time. Original pilot outputs are preserved.

## Scope and handling

Analyze the five VM collections: Mendeley goodware and ransomware (preserving
train/test), balanced goodware, VS ransomware, and VirusShare. Inventory every
file in sample directories; record unsupported/non-PE inputs and failures.
Never execute samples, unpack them, change datasets, install packages, or alter
VM networking, shared folders, or snapshots. Preserve the existing unfinished
`export_ida_cfg.py` and README changes.

All VM artifacts, private analysis copies, databases and logs live under
`~/work/out/ida_cfg_20261006/`. Use Python 3.8 and the already installed IDA Pro
9.4 idalib binding. Each sample runs in a separate process, with memory/time
limits and its own SHA-256-named temporary directory. Run the parent under
nohup; resume validated successful outputs and retain explicit failure reasons.

## Output contract

One `cfg/<original-sha256>.json.gz` per unique successfully analyzed input.
Each graph contains functions with ordinal IDs; basic blocks with local IDs,
IDA block types, ordered mnemonic tokens and operand *type categories*; and
directed intraprocedural successor edges. Calls are separately represented by
source block/instruction and an ordinal target function when resolved.
No instruction bytes, operand values, addresses, symbol names, strings, or full
assembly listing leave the VM. Use original IDA mnemonics, without silently
mixing Capstone spelling. A strict allowlist validator rejects unexpected fields.

Use IDA auto-analysis and `FlowChart(..., FC_NOEXT)`; keep the FlowChart object
alive while visiting its blocks. Edges are recovered static relationships,
not a claim of runtime reachability. Packed, managed, indirect-transfer and
unrecognized code can be incomplete. Do not label an empty graph a success.
Caps are explicit and truncated graphs carry a flag. Native x86/x64 PE only;
managed/other inputs remain in the inventory with a reason.

Keep labels, source group, family, CSV flags and train/test membership in an
independent manifest. Hash once and extract unique contents once across groups;
retain every file-to-hash association. Audit conflicting labels and hashes across
train/test groups before ML. VirusShare directory membership is not a verified
ransomware label: preserve source labels separately and leave unverified labels
unset. Do not manufacture a train/test split.

## Execution gates

1. Verify SSH, disk space, installed IDA APIs, cohort metadata and counts.
2. Implement inventory, isolated extraction, schema validation, resume and
   allowlisted packaging. Test invariants and failure handling with fixtures.
3. Build inventory in the VM and run a small real-IDA pilot across groups.
4. Validate graph endpoints, instruction alignment, hashes and output contract;
   inspect pilot counts and timing, then start full extraction with two workers.
5. Package only validated graph JSON and derived manifests/reports, with a
   SHA-256 checksum index. Transfer that explicit package using SCP, validate
   it on the desktop, and report input files, unique hashes, successful CFGs,
   duplicates, skipped inputs, truncation, failures and pending work by group.

## References

- VM access/handling: `C:\Users\chaoa\Downloads\asm and mm\Shared\CODEX_VM_ACCESS.md`
- IDA successor semantics: https://python.docs.hex-rays.com/ida_gdl/index.html
- Installed API source: `/home/seed/ida-pro-9.4/idalib/python/idapro/__init__.py`

Full disassembly transfer is outside this mnemonic/type-only export contract.
Fit vocabularies and feature transforms on training hashes only; never feed
manifest labels, family, file names or cohort identifiers into graph features.

# Step 1: IDA export

This step has no code here. The export runs on the VM with IDA Pro 9.4 (x86 binaries only; a managed binary
exports CIL methods and step 2 rejects it). The exporter is not part of this repo.

## What an export holds

One gzip file per binary, `<sha256>.jsonl.gz`, one JSON record per line, schema `ida-cfg-research/2.1`:

- `header`: `sha256`, `schema`, `analysis_options` (IDA `processor` must be `metapc`) and `pe.imports`.
- `function` (one per function): `rva`, `library` and `thunk` flags, `blocks` and `edges`.
  - each block has `id`, `start_rva`, `end_rva`, `type`, `external` and its `instructions`;
  - each instruction has `mnemonic`, `operands` (with a `category`), `is_call`, `code_refs` (call targets) and
    `import_ids` (indices into `pe.imports`);
  - `edges` are `[source block id, target block id]` pairs inside the function.
- `footer`: `complete` is true when the export finished.

## Where the files live

| corpus | folder | index |
|---|---|---|
| `mendeley` | `C:/Users/chaoa/Downloads/ransomware-detection-masters-project/reports/google_drive_upload/2026-10-07/Mendeley_GNN_v1/cfg/` | `samples.csv`, `splits/`, `provenance/identity_aliases.csv` in the same folder |
| `ransomware_balanced` | `C:/Users/chaoa/Downloads/ida_collections_balanced/Ransomware_Balanced/` | `samples.csv` (schema `collection-cfg-asm/1`) |
| `balanced_goodware` | `C:/Users/chaoa/Downloads/ida_collections_balanced/Balanced_Goodware/` | `samples.csv` |

`pipeline/s02_build_cfg_cache/build_cache.py` (`CORPUS_ROOTS`) reads them; each `samples.csv` row has a `cfg_path`
relative to the corpus folder. Step 2 turns the exports into the cache that the rest of the pipeline reads.

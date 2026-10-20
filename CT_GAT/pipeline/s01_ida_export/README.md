# s01: IDA Pro export

The IDAPython extractor that produced every `*.jsonl.gz` control-flow graph and `*.asm.gz` listing in the Mendeley,
Ransomware_Balanced and Balanced_Goodware collections. It runs inside IDA Pro 9.4 on the offline VM, where the binaries
stay; only the derived text files leave the VM.

| File | What it does |
|---|---|
| `extract_cfg_with_ida.py` | The extractor. One IDA analysis per binary writes `<sha256>.jsonl.gz` (schema `ida-cfg-research/2.1`) and, with `--asm-out`, the `<sha256>.asm.gz` listing from the same analysis. |
| `run_mendeley_asm.py` | The `.asm.gz` writer the extractor calls (`export_current`), plus the Mendeley assembly batch controller. |
| `run_research_cfg.py` | Resumable batch controller and the acknowledged transfer of finished exports from the VM. |
| `vm_ida_cfg.py` | The first-pass VM pipeline (`summary`, `run`, `package`) whose helpers the two controllers import: sample inventory, validation, disk guard, safe packaging of derived data. |
| `test_*.py` | Unit tests (IDA is mocked; `python -m pytest -q` in this folder, 24 tests). |
| `RUNBOOK_CFG.md`, `RUNBOOK_ASM.md` | The step-by-step VM runbooks used for the 2026-10-06/07 exports. |

Provenance: every export header carries `extractor_sha256`. The exports in the collections have
`301561471ce4ff8a...`, the sha256 of `extract_cfg_with_ida.py` in this folder, byte for byte. Do not edit the file
without bumping `SCHEMA`; an edit changes the hash.

## Running it

Inside the VM, with the Python that has IDA's idalib (`import idapro`); the extractor opens the binary itself, so it is a
plain script, not an `idat -S` plugin:

```bash
python3 extract_cfg_with_ida.py --binary <path> --output <sha256>.jsonl.gz --sha256 <sha256> --arch <x86|x64> --kind <kind> [--asm-out <dir>]
```

For whole collections use the batch controllers (`run_research_cfg.py`, which calls the extractor once per binary in a
temporary copy, with `--asm-out` for Mendeley); the runbooks give the exact commands, folders and checks.

## What a `.jsonl.gz` export holds

One JSON record per line:

- `header`: `sha256`, `schema`, `extractor_sha256`, `arch`, `ida_version`, `analysis_options` (IDA `processor` is
  `metapc`), `pe` (including `pe.imports`).
- `function`, one per function: `rva`, `name`, `library` and `thunk` flags, `blocks` and `edges`.
  - each block has `id`, `start_rva`, `end_rva`, `type`, `external` and its `instructions`;
  - each instruction has `mnemonic`, `operands` (each with a `category`), `is_call`, `code_refs` (call targets) and
    `import_ids` (indices into `pe.imports`);
  - `edges` are `[source block id, target block id]` pairs inside the function.
- `orphan` and `string` records, then a `footer`: `counts`, `graph_status`, and `complete: true` when the export finished.

## Notes for the later stages

- Everything the models need (instructions, operand categories, imports, block edges, call targets) is in the
  `.jsonl.gz`, which holds code only and can be streamed one function at a time.
- The `.asm.gz` listing also prints every data byte as a `db` line. Two Mendeley listings are 341 MB compressed but
  23 GB uncompressed (99.9% data lines), so readers must stream it line by line rather than read it whole.

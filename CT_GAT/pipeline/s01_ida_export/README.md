# s01: IDA Pro export

The IDAPython extractor that produced every `*.jsonl.gz` control-flow graph and `*.asm.gz` listing in the Mendeley,
Ransomware_Balanced and Balanced_Goodware collections. It runs inside IDA Pro 9.4 on the offline VM, where the binaries
stay; only the derived text files leave the VM.

| File | What it does |
|---|---|
| `extract_cfg_with_ida.py` | The CFG export. One IDA analysis per binary writes `<sha256>.jsonl.gz` (schema `ida-cfg-research/2.1`); with `--asm-out` it also writes the listing from the same analysis. |
| `run_mendeley_asm.py` | The listing export the extractor calls: `<asm-out>/asm/<sha256>.asm.gz` (IDA `OFILE_ASM`, gzip) and `<asm-out>/status/<sha256>.json`. A listing failure is recorded in `<asm-out>/combined_errors/` and never invalidates the CFG. |

`extract_cfg_with_ida.py` is byte-identical to the version that produced the collections: its sha256 is the
`extractor_sha256` in every export header (`301561471ce4ff8a...`). Editing it changes that hash, so bump `SCHEMA` if you do.

## Running it

Inside the VM, with the Python that has IDA's idalib (`import idapro`). The extractor opens the binary itself, so it is a
plain script, not an `idat -S` plugin:

```bash
python3 extract_cfg_with_ida.py --binary <path> --output <sha256>.jsonl.gz --sha256 <sha256> \
    --arch <x86|x64> --kind <kind> [--asm-out <dir>]
```

One call per binary; loop over a sample list for a collection.

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
  23 GB uncompressed (99.9% data lines), so readers must stream a listing line by line rather than read it whole.

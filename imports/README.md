# `imports/` - the PE import-table side channel

The sequence model reads a mnemonic stream. That stream says *how* a sample
computes but almost nothing about *what it asks the OS for*, because every
API call in `asm_output/unified_*` is an opaque indirect branch:

```
0x401003:  call    dword ptr [0x4291f0]
0x140001020:  call  qword ptr [rip + 0xd639]
```

This package turns those operands back into API names. `extract_imports.py`
parses the import and delay-load import directories of a PE and writes, keyed
by SHA-256, both the flat import list (a bag-of-APIs feature on its own) and a
map from **IAT slot virtual address** to `dll!func`, which is what lets a
tokenizer rewrite the two lines above as `call kernel32.dll!createfilew`.

Nothing here executes a sample. Every file is opened read-only and parsed with
`pefile`; `fast_load=True` plus the two import directories is the whole of it,
so a 1,500-file corpus takes about 13 seconds.

## Output

```jsonc
{"<sha256>": {
   "imports":  ["kernel32.dll!createfilew", "advapi32.dll!#123", ...],
   "iat":      {"0x4291f0": "kernel32.dll!createfilew", ...},
   "dll_count": 12,
   "func_count": 310,
   "error": ""}}
```

* Names are lowercased and keep the DLL prefix; an ordinal import is
  `dll!#123`.
* `imports` is ordered and de-duplicated - import directory first, then
  delay-load. `func_count` is its length, `dll_count` the number of distinct
  DLLs.
* `iat` keys are `ImageBase + slot RVA` formatted the way the disassembler
  prints an operand: `0x` + lowercase hex, unpadded. There are slightly more
  IAT slots than `func_count` entries, because a symbol imported both
  normally and delay-loaded (or listed twice by the linker) dedups in
  `imports` but has two distinct slots.
* `error` is `""` on success and `TypeName: message` when `pefile` refused the
  image; the record is then empty rather than absent.

The JSON is written one compact line per SHA-256, so a 36 MB file is still
`grep`-able by hash. Above `--iat-split-mb` (default 50) the `iat` maps are
spilled into a sibling `<stem>.iat.json` and dropped from the main file;
`--verify-asm` reloads the sibling automatically, and `.gitignore` keeps
`manifests/imports/*.iat.json` out of the repo.

The `--manifest` CSV has one row per **input file** - including non-PE files
(`status=not_pe`) and duplicates - with `sha256, rel_path, status, dll_count,
func_count`. `status` is one of `ok`, `no_imports`, `parse_error`, `not_pe`,
`read_error`. Use it, not the JSON, to answer "what did the walk see"; the
JSON is keyed by content hash and therefore collapses duplicates.

## Running it

```bash
python imports/extract_imports.py \
    --in  C:/Users/chaoa/Downloads/Goodware_Balanced \
    --out manifests/imports/goodware_balanced.json \
    --manifest manifests/imports/goodware_balanced.csv \
    --exclude _upx_packed --exclude flagged
```

Hidden directories are always pruned; `--exclude` is repeatable and matches a
directory *name* case-insensitively (`_upx_packed`, `flagged`, `.tools`).

## What has been run

| output | corpus | files | unique PEs | size |
|---|---|---|---|---|
| `manifests/imports/goodware_balanced.{json,csv}` | `Goodware_Balanced` (minus `_upx_packed`, `flagged`) | 1,504 | 1,500 | 36.4 MB / 0.15 MB |
| `manifests/imports/mendeley_goodware_host.{json,csv}` | host `Goodware_Training/goodware` | 1,115 | 1,112 | 14.1 MB / 0.10 MB |
| `manifests/imports/mendeley_mal_train.{json,csv}` | VM `Ransomware_Training/rans` (step 4b, 15 Sep 2026) | 1,023 | 975 ok, 48 no_imports | 6.0 MB / 0.15 MB |
| `manifests/imports/mendeley_mal_test.{json,csv}` | VM `Ransomware_Test/rans_test` | 385 | 337 ok, 48 no_imports | 3.2 MB / 0.06 MB |
| `manifests/imports/mendeley_good_train.{json,csv}` | VM `Goodware_Training/goodware` (complete, unpacked copy) | 1,134 | 1,130 ok, 4 no_imports | 15.2 MB / 0.11 MB |
| `manifests/imports/mendeley_good_test.{json,csv}` | VM `Goodware_Test/goodware_test` | 133 | 133 ok | 1.7 MB / 0.01 MB |
| `manifests/imports/imports_flat.json` | `merge_imports.py` over the five above: `{sha256: [names]}` for the sequence model | | | (not committed) |

No parse errors in either run. 107 of the balanced PEs and 4 of the Mendeley
PEs have no import directory at all (`status=no_imports`): .NET images whose
only real import is the `mscoree.dll` stub, API-set forwarders, and packed
stubs that build their own IAT at run time.

### The host Mendeley copy is a fallback

`Goodware_Training/goodware` on the host is the **incomplete, partly packed**
copy: 1,115 files against the VM's 1,134, and 67 of them are still UPX-packed,
so their SHA-256 is the packed hash and matches no cohort row. Measured
against `asm and mm/Shared/cohort_mendeley.csv`:

* 1,028 of the 1,114 in-cohort `good_train` SHA-256s are covered (92.3%).
* The 86 that are not are exactly the 67 still-packed files (whose hashes
  appear nowhere in the cohort - confirmed by their `UPX0`/`UPX1` sections)
  plus the 19 files the host copy is missing.
* 0 of the 129 in-cohort `good_test` SHA-256s are covered: `Goodware_Test`
  is VM-only.

So `mendeley_goodware_host.json` exists to unblock the sequence model now.
**The VM run of step 4b in `vm_package/README.md` supersedes it**: that run
covers the unpacked 1,134-file `Goodware_Training` and `Goodware_Test`, and
its `mendeley_good_train.json` should replace this file once it is copied
back.

## Resolving a transcript

```bash
python imports/extract_imports.py \
    --verify-asm C:/Users/chaoa/Downloads/asm_output/unified_goodware_balanced/goodware_balanced \
    --json  manifests/imports/goodware_balanced.json \
    --limit 20 --seed 0 \
    --corpus C:/Users/chaoa/Downloads/Goodware_Balanced \
    --index  manifests/imports/goodware_balanced.csv
```

`--corpus`/`--index` are optional and only used to *explain* the misses (they
re-open each PE for its load-config directory and section table).

Two operand forms are handled:

* **Absolute** (x86): `call dword ptr [0x4291f0]` - the operand is the slot VA.
* **RIP-relative** (x64): `call qword ptr [rip + 0xd639]` - RIP is the address
  of the *next* instruction, which the transcript prints on the following
  line, so the slot is `next_line_address + displacement` (and `-` for a
  negative displacement). `bnd jmp qword ptr [rip + ...]` thunks are included.
  At the very last line of a transcript - or one truncated by the
  100,000-instruction cap - there is no next line and the target is reported
  unresolved rather than guessed.

On 20 transcripts sampled with `--seed 0`:

```
6,912 / 8,066 memory-indirect call/jmp targets resolved (85.69%)
  abs   983/1,793 (54.82%)
  rip  5,929/6,273 (94.52%)
misses:
  1,096  CFG guard pointer (__guard_check/dispatch_icall)
     30  function pointer in .data
     23  function pointer in .bss
      3  function pointer in .rdata
      2  no next instruction (transcript end / 100k cap)
```

**95% of the misses are not imports at all.** With `/guard:cf`, MSVC emits
`call qword ptr [__guard_dispatch_icall_fptr]` in front of *every* indirect
call, and that pointer lives in the load-config directory, usually in `.rdata`
a few bytes past the end of the IAT. It is a single slot hit hundreds or
thousands of times in one binary - `FileSyncShell.dll` accounts for 805 of the
1,096 on its own, which is why its per-file rate is 40.9% while the corpus
rate is 86%. Discounting the guard pointer, **6,912 / 6,970 = 99.17%** of
import-shaped indirect targets resolve. The remaining 56 are genuine runtime
function pointers in `.data`/`.bss` (`GetProcAddress` results, callback
tables), which by construction have no import entry.

If the tokenizer wants a token for the guard pointer it should come from the
load-config directory, not from here; `classify_image()` already reads it.

## Tests

```bash
python tests/make_fixtures.py --corpus C:/Users/chaoa/Downloads/Goodware_Balanced
python -m pytest tests/test_imports.py -q
```

The PE assertions run against `tests/fixtures/` (benign, never executed); the
RIP arithmetic is pinned on hand-written `.asm` snippets so it does not need a
disassembler.

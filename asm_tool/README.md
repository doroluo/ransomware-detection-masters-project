# asm_tool — PE → `.asm` → opcode text, packaged for the offline VM

Three scripts and a wheel set. Nothing here executes a sample: every tool opens
files read-only and parses headers or disassembles bytes.

> **There is no `asm_parser/` package and there must not be one.** The repo root
> already has `asm_parser.py` — the CNN-ViT tokenizer. A directory of that name
> would shadow it on import and silently break every adversarial script. Hence
> `asm_tool/`.

## What runs where

| step | script | runs on |
|---|---|---|
| PE tree → `.asm` tree | `../asm_parse.py` | VM (ransomware) and host (goodware) |
| `.asm` tree → opcode `.txt` | `asm_to_opcodes.py` | either |
| architecture census | `../check_arch.py` | VM, before anything else |
| disassembler equivalence | `consistency_check.py` | host only, goodware only |

`check_arch.py` now excludes the same paths `asm_parse.py` excludes, so the two
describe the same corpus. It used to count everything under `--dir`: pointed at
`Goodware_Balanced` it reported 1,525 PEs and 25 "UPX packed" for a corpus that
is 1,500 PEs with zero UPX, because `_upx_packed/` holds the packed originals of
files already unpacked in place and `.tools/` holds `upx.exe`. It now reports
1,500 / 1,125 x64 (75.0%) / 375 x86 (25.0%) / 70 .NET / 0 UPX, matching
`asm_manifest.csv` and `DISTRIBUTION.md` exactly. `--no-skip` restores the old
behaviour.

## 0. Where the output lives

Everything produced by these scripts goes under one shareable, gitignored root:

```
asm_output/
  goodware_balanced/     <- Goodware_Balanced (1,500 PEs)
  mendeley_goodware/     <- Mendeley Goodware_Training/goodware (1,115 PEs)
  ransomware/            <- produced on the VM, copied back
```

On this host that root is `C:/Users/chaoa/Downloads/asm_output/`. It replaces the
old `../Balanced_Goodware_ASM` tree, which is now
`asm_output/goodware_balanced/`. Every path in this file is relative to that
layout.

`asm_output/goodware_balanced/` holds the `asm_parse.py` extraction of that
corpus:

| entry | written by | what it is |
|---|---|---|
| `everyday/`, `hard_negative/`, `system/` | `asm_parse.py` | 1,343 `<filename>.asm`, mirrored bucket tree, capped at 100,000 instructions |
| `asm_manifest.csv` | `asm_parse.py` | 1,500 rows, one per input PE (section 1) |

A second, independent extraction of the same corpus lives in
`C:/Users/chaoa/Downloads/asm and mm/Shared/Extract_Goodware_Balanced/`
(`extract-opcode/extract_unified.py`: `asm/<sha256>.asm` full skip-data
disassembly, `mn/<sha256>.txt` mnemonic-only sequences, `manifest.csv` with
sha256 / tag / arch / entropy / `decoded_ratio` / `n_insns`, `profile.csv`,
`DISTRIBUTION.md`). Its sibling `Shared/Extract/` holds the same extraction of
the full Mendeley corpus (all four sets). Those folders are the independent
measurement that section 6 below compares `asm_parse.py` against; they are
read-only from this package.

## 1. `asm_parse.py` — the CLI

Flag names differ from the worker brief:

| brief | actual |
|---|---|
| `--in` / `--out` | `--in-dir` / `--out-dir` |
| `--workers N` | `--jobs N` |
| `--manifest f.csv` | always written to `<out-dir>/asm_manifest.csv` |
| `--resume` | the default; `--overwrite` turns it off |

```bash
python asm_parse.py --in-dir  ../Goodware_Balanced \
                    --out-dir ../asm_output/goodware_balanced
python asm_parse.py --in-dir X --out-dir Y --limit 50          # smoke test
python asm_parse.py --in-dir X --out-dir Y --refresh-manifest  # manifest only
```

Output mirrors the input tree, so class folders never collide. The walk skips
`_upx_packed/`, `.tools/`, `flagged/` and `quarantined/` by directory name and
`.csv` / `.md` / `.json` by suffix, and nothing else — every remaining file gets
a manifest row, `ok`, `truncated`, `dotnet_ilonly`, `upx_packed`,
`no_exec_section`, `empty_disassembly`, `arch_unsupported:*`, `not_pe`,
`pe_error:*`, `read_error:*` or `crash:*`. Manifest row count equals input file
count.

### Manifest columns

```
rel_path, asm_path, status, arch, instructions,
sha256, machine, bitness, is_dotnet, packed_flag, packer_guess,
sections_disassembled, n_instructions, error, source
```

The first five are the original set and keep their exact meaning, because
`asm_to_opcodes.py`, the tests and the DISTRIBUTION tables were written against
them. The rest are the worker brief's per-file row spec, added alongside;
`n_instructions` is the brief's name for `instructions` and both are written.
`source` is `disassembled`, `resumed` or `refreshed`.

`packed_flag` / `packer_guess` are **advisory only**. They never change which
files get disassembled. Three independent signals, joined with `+` in
`packer_guess`:

* a section name a known packer writes (`UPX0`/`UPX1`, `.aspack`/`.adata`,
  `.themida`, `.vmp0`/`.vmp1`, `.MPRESS1`, `PEC2`/`PECompact`, `.petite`,
  `.nsp0`, `.enigma1`, FSG, MEW, Upack, yoda, RLPack, WWPack, MoleBox, NeoLite,
  kkrunchy, Crinkler),
* max executable-section Shannon entropy > 7.2 — the same threshold
  `extract_unified.py` and `profile_packing.py` use, so the tables are
  comparable,
* an entry point that lands outside every executable section (skipped when the
  entry point is 0, which is normal for resource-only DLLs).

### Resume and refresh

`--resume` is the default. A resumed run **reproduces the row the original run
wrote** — status (including the `ok` / `truncated` distinction), arch and
instruction count — from the prior manifest, and recomputes every
header-derived column from the binary. A manifest written by an older version,
with `already_done` placeholders and blank arch/counts, is *repaired* by a plain
resume run rather than carried forward; where no prior row exists the status is
re-derived from the `.asm` itself (a file at exactly the cap was truncated,
anything shorter completed). Verified on 60 real corpus files, including 8
truncated ones: 0 differing cells against a from-scratch run, in both the
"prior manifest present" and "prior manifest degraded" cases.

`--refresh-manifest` rewrites the manifest only. It re-reads every binary,
re-disassembles in memory, recomputes every column and compares the result
against the `.asm` on disk — but writes no `.asm` file at all. A file that no
longer matches gets `asm_mismatch` in `error`. This is how both manifests below
were regenerated: the `.asm` trees are provably untouched because nothing wrote
to them, and `error` came back empty on all 2,615 manifest rows — which is a
byte-identity proof for all 2,375 `.asm` files, not for a sample of them.

`--overwrite` and `--refresh-manifest` are mutually exclusive.

### Runs

`Goodware_Balanced` → `asm_output/goodware_balanced` (1,500 PEs):

| status | count |
|---|---|
| `ok` | 1,172 |
| `truncated` | 171 |
| `no_exec_section` | 87 |
| `dotnet_ilonly` | 69 |
| `empty_disassembly` | 1 |
| **total** | **1,500** |

1,343 `.asm`, 1.23 GB. `packed_flag=1` on 11 (6 native + 5 IL-only, all from the
entropy signal — no packer section name and no stray entry point in this
corpus); `is_dotnet=1` on 70. Hard errors (`not_pe`, `pe_error:*`,
`read_error:*`, `crash:*`): **0**.

The 6 native flagged files are exactly the 6 `packed_other` rows in
`DISTRIBUTION.md`; the other 5 are IL-only assemblies, which
`extract_unified.py` tags `dotnet` before it ever looks at entropy. The single
`empty_disassembly` is the mixed-mode assembly — the 70th `.NET` file, not
IL-only, so it is disassembled, and its native section decodes to nothing.

Mendeley `Goodware_Training/goodware` → `asm_output/mendeley_goodware`
(1,115 PEs, flat):

| status | count |
|---|---|
| `ok` | 989 |
| `upx_packed` | 67 |
| `truncated` | 43 |
| `dotnet_ilonly` | 11 |
| `empty_disassembly` | 5 |
| **total** | **1,115** |

1,032 `.asm`, 0.55 GB. `packed_flag=1` on 72 (67 `upx+high_entropy`, 4
`high_entropy`, 1 `aspack+high_entropy`); `is_dotnet=1` on 13. Hard errors:
**0**. Three pairs of rows share a SHA-256; `Goodware_Balanced` has no internal
duplicates.

### Are flagged files contributing garbage opcodes?

The skip-data `decoded_ratio` cannot answer this. x86 is a dense encoding:
1 MB of random bytes decodes as valid instructions at a ratio of 0.99 in
32-bit mode and 0.94 in 64-bit mode, and compressed data does the same, so a
packer payload scores as high as real code. The 6 native `packed_flag=1`
files in `Goodware_Balanced` decode at 92.3–100%, which is therefore
uninformative; they are excluded from the opcode-feature cohort on the
entropy flag alone, the same rule applied to the 84 entropy-flagged
ransomware samples (which also decode at a median 0.99).

Mendeley goodware is the case to watch. All 67 UPX files are skipped by status
and write no `.asm`, so no decompressor stub enters the corpus. The other five
flagged files are not skipped, and four of them produce a stub-sized stream —
`GenValObj.exe` 649, `old_smb.exe` (`aspack+high_entropy`) 563, `fg761p.exe`
484, `freegate-38819-1.exe` 155 instructions; the fifth is IL-only and produces
nothing. Those four are plausibly packer stubs counted as real opcodes. They
were already in the corpus before this change and their disassembly is
deliberately unchanged; what is new is that `packed_flag` makes them
identifiable, so a downstream filter can drop them without a second pass over
the binaries. There is no unified/skip-data manifest for this dataset, so
`decoded_ratio` is not available to settle it either way.

Uncapped, `Goodware_Balanced` would disassemble to ~31 GB — one 185 MB binary
alone produces millions of lines — hence the 100,000-instruction default cap,
which is still 4.6x what `asm_parser.py` consumes.

## 2. `asm_to_opcodes.py` — the bridge to `LLM_Features`

Run this before mixing `.asm` output with anything `extract.py` produced.

```bash
python asm_tool/asm_to_opcodes.py \
    --asm-dir  ../asm_output/goodware_balanced \
    --out-dir  ../LLM_Features_Balanced/good_all \
    --index    ../Goodware_Balanced/corpus_index.csv \
    --manifest ../LLM_Features_Balanced/opcode_manifest.csv
```

`asm_parse.py` writes `0x00401000:  mov\teax, ebx`; `extract.py` writes
`mov eax, ebx`. The tokenizer folds `0x...` into `<HEX>`, so an unstripped line
becomes `<HEX>: mov eax ebx`. Feed it ransomware without the prefix and goodware
with it and the prefix separates the classes perfectly while carrying no
information at all. The conversion is idempotent, so re-running it on an already
converted corpus is safe.

Naming follows `extract.py`'s `{family}_{filename}.txt`, where family is the
sample's folder — the bucket, for `asm_output/goodware_balanced`. That prefix is
what keeps `everyday/node.exe` and `hard_negative/node.exe` apart; a name
collision aborts the run rather than overwriting. A flat tree such as
`asm_output/mendeley_goodware` gets the family `root`.

## 3. `consistency_check.py` — proof the two disassemblers agree

```bash
python asm_tool/consistency_check.py --corpus ../Goodware_Balanced -n 10
```

Samples both architectures (a 32/64-bit mode mix-up still "decodes", so a
single-arch sample would miss it) and compares the mnemonic sequence and the
normalized instruction sequence against `extract.py`'s. Result on 10 files
(5 x86, 5 x64): **match on both, all 10**, exit code 0. The only difference is
the printed branch target — `extract.py` disassembles from
`section.VirtualAddress`, `asm_parse.py` from `ImageBase + section.VirtualAddress`
— which normalizes away.

Exit code is non-zero if any file disagrees.

## 4. Tests

```bash
python tests/make_fixtures.py --corpus ../Goodware_Balanced
python -m pytest tests/ -q
```

`tests/test_asm_parse.py` is 31 of them (the rest of `tests/` covers the
tokenization pipeline). Fixtures are benign binaries copied out of the VirusTotal-clean
`Goodware_Balanced` corpus — a 32-bit PE, a 64-bit PE, a DLL, an IL-only .NET
assembly, a UPX-packed DLL, a PE with no executable section — plus four
synthesised cases: `highentropy.bin` (a copy of `pe64.bin` with its code section
overwritten by `os.urandom`, a packer-shaped PE that was never packed),
`truncated.bin`, `garbage_mz.bin` and `notpe.txt`. They are gitignored and
rebuilt on demand, so no binaries enter git.

Covered: architecture detection (including a positive check that x86 output
contains **no** 64-bit registers, the only way to catch a silent mode mix-up),
the `.NET` / UPX / no-exec flags, `packed_flag` on both the UPX fixture and the
synthetic high-entropy one — and that the high-entropy fixture is still
disassembled, since the flag must never divert a file — the absence of the flag
on ordinary binaries, every manifest column being present and populated, resume
preserving `ok`/`truncated`/arch/counts and not rewriting `.asm`, resume
repairing a degraded `already_done` manifest, `--refresh-manifest` leaving the
`.asm` tree untouched and detecting a tampered file, malformed input producing a
status row rather than an exception, the exact line format
`asm_parser.parse_asm_line` expects, the instruction cap, bridge idempotence,
and a CLI run whose manifest row count equals the fixture count.

## 5. Offline VM package

The VM is Ubuntu (SEED) on x86-64 with `python3`; wheels built for this Windows
host are useless there, so the vendored set is **manylinux2014_x86_64**, one
directory per Python minor version:

```bash
# on this host, once
python -m pip download --only-binary=:all: --platform manylinux2014_x86_64 \
    --python-version 3.8  -d asm_tool/wheels/cp38  -r asm_tool/requirements-vm.txt
python -m pip download --only-binary=:all: --platform manylinux2014_x86_64 \
    --python-version 3.10 -d asm_tool/wheels/cp310 -r asm_tool/requirements-vm.txt
python -m pip download --only-binary=:all: --platform manylinux2014_x86_64 \
    --python-version 3.12 -d asm_tool/wheels/cp312 -r asm_tool/requirements-vm.txt
```

All three succeeded. `requirements-vm.txt` is only `pefile` and `capstone` —
verified to be exactly the third-party imports of `asm_parse.py`,
`check_arch.py` and `asm_to_opcodes.py` (which is stdlib-only). `pandas` is in
`requirements.txt` for `extract.py` and `llm_features_pipeline/`, not for
anything that runs on the VM, and `pytest` is host-side only; neither is
vendored.

Both resolved wheels are ABI-independent —
`pefile-2024.8.26-py3-none-any.whl` and
`capstone-5.0.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` — so
the three directories are byte-identical and any CPython 3.x on the VM can
install from any of them. They are kept separate anyway so that a future
dependency with a real `cp3XY` ABI tag lands in the right place.

On the VM, in order:

```bash
python3 --version                       # -> pick cp38 / cp310 / cp312
python3 -m pip install --no-index --find-links asm_tool/wheels/cp310 \
                       -r asm_tool/requirements-vm.txt

python3 check_arch.py --dir <ransomware dir> \
                      --compare ../Goodware_Balanced/corpus_index.csv
python3 asm_parse.py --in-dir <ransomware dir> --out-dir asm_output/ransomware
python3 asm_tool/asm_to_opcodes.py --asm-dir asm_output/ransomware \
                                   --out-dir  LLM_Features_VM/mal_all
```

Copy `asm_tool/`, `asm_parse.py`, `check_arch.py` and `asm_tool/wheels/` to the
VM. `wheels/` is gitignored (`.gitignore` line `asm_tool/wheels/`, after the
`!asm_tool/**` re-include), so it travels by file copy, not by git.

Run `check_arch.py` **first**. If the ransomware set is predominantly one
architecture and the goodware the other, `machine` becomes a shortcut feature
and the classifier can separate the classes without learning anything about
behaviour. `Goodware_Balanced` is 75% x64 / 25% x86; a gap above 25 points is
flagged as a warning by the script.

Only `asm_output/` leaves the VM. It is text, but it is a faithful transcript of
malware code — treat the archive the way you would treat the samples.

## 6. Known limitation: the sweep stops at the first undecodable byte

**This is deliberate and must not be "fixed".**

`asm_parse.py` uses a plain linear capstone sweep with `skipdata` **off**, over
`IMAGE_SCN_MEM_EXECUTE` sections only. When the sweep meets a byte it cannot
decode — a jump table, a literal pool, CFG padding, any data the compiler
interleaved with code — `md.disasm()` simply stops and the rest of that section
is lost, silently. `extract.py` behaves identically, and `LLM_Features`, the
ransomware side of every experiment in this repo, was built with `extract.py`.
Turning `skipdata` on here would lengthen the goodware streams and leave the
ransomware streams untouched, and the resulting systematic length and content
difference between classes would be a shortcut feature — a bigger measurement
error than the one it fixes. Any change must be applied to **both** sides and
both corpora must then be rebuilt.

### How much is lost, measured

`asm_output/goodware_balanced/` contains both extractions of the same 1,500
binaries, so the cost is directly measurable: `asm_manifest.csv`
(`instructions`, no skipdata, capped at 100,000) joined on `rel_path` against
`Shared/Extract_Goodware_Balanced/manifest.csv` (`n_insns`, skipdata on, uncapped). 1,342 files are
comparable (`status` `ok`/`truncated` and `disassembled=1` on the other side).

Ratio `instructions / min(n_insns, 100000)`:

| percentile | p5 | p10 | p25 | **p50** | p75 | p90 | p95 |
|---|---|---|---|---|---|---|---|
| ratio | 0.006 | 0.022 | 0.197 | **1.000** | 1.012 | 1.061 | 1.114 |

| band | files | share |
|---|---|---|
| < 10% | 238 | 17.7% |
| 10–25% | 132 | 9.8% |
| 25–50% | 115 | 8.6% |
| 50–75% | 52 | 3.9% |
| 75–90% | 24 | 1.8% |
| 90–99.9% | 47 | 3.5% |
| ~100% | 183 | 13.6% |
| > 100% | 551 | 41.1% |

**Of the 1,171 files that did not hit the 100,000 cap, 485 (41.4%) produced
under 50% of the uncapped skip-data count**, and 238 produced under 10%.
Across all comparable files the totals are 37.8 M instructions against 65.9 M
(57.3%).

Worst 10, all `status=ok` (not capped):

| ratio | asm_parse | skip-data | arch | file |
|---|---|---|---|---|
| 0.0001 | 5 | 313,282 | x64 | `hard_negative/age-keygen.exe` |
| 0.0001 | 8 | 3,043,849 | x64 | `hard_negative/duplicacy_win_x64_3.2.5.exe` |
| 0.0001 | 9 | 340,546 | x64 | `hard_negative/age-plugin-pq.exe` |
| 0.0001 | 9 | 323,477 | x64 | `hard_negative/age-plugin-tag.exe` |
| 0.0001 | 9 | 9,436,246 | x64 | `hard_negative/rclone.exe` |
| 0.0001 | 10 | 546,610 | x64 | `hard_negative/age.exe` |
| 0.0001 | 11 | 360,982 | x64 | `hard_negative/age-inspect.exe` |
| 0.0001 | 11 | 1,171,864 | x64 | `hard_negative/d3dcompiler_47.dll` |
| 0.0001 | 11 | 5,791,823 | x64 | `hard_negative/kopia.exe` |
| 0.0001 | 4 | 34,423 | x86 | `system/wusa_03ca1617.exe` |

These are Go and Rust binaries. Their `.text` opens with runtime scaffolding
that the sweep cannot get past, so a 9 MB executable yields nine instructions.

Does it starve the model? `asm_parser.py` consumes ~21,800 instructions and the
opcode tokenizer truncates at 5,000. **238 files (17.7%) deliver fewer than
5,000 instructions where the skip-data sweep would have delivered 5,000 or
more**, and 33 files deliver fewer than 100 instructions each. The loss is not
uniform: 55.1% of x86 files fall under the 50% line versus 36.8% of x64, and it
runs 38.7–43.2% across the three buckets. **Any comparison of the two goodware
sources, or of goodware against ransomware, is a comparison of streams truncated
this way on both sides** — which is the only reason it is safe.

### The mirror-image effect: raw-section padding decoded as code

551 files (41.1%) come out with *more* instructions than the skip-data sweep.
This is a second difference from `extract_unified.py` and it also matches
`extract.py`: `asm_parse.py` feeds `section.get_data()` — the full
`SizeOfRawData` — to capstone, while `extract_unified.py` trims each section to
`min(SizeOfRawData, VirtualSize)`. The file-alignment padding past the end of
the virtual section is therefore disassembled here, usually as long runs of
`add byte ptr [eax], al` from zero bytes. It is mild for most files (p75 = 1.012)
and extreme for tiny code sections: `everyday/Deutsch.dll` has 32 bytes of real
code and produces 2,045 instructions, and `everyday/python3_d98cd5d8.dll` has 6
bytes and produces 255.

`extract_unified.py` also selects sections on `CNT_CODE | MEM_EXECUTE` where
`asm_parse.py` requires `MEM_EXECUTE`, so a section with only the `CNT_CODE` bit
is compared on one side and not the other. Both differences are recorded here
rather than removed, for the same reason as the first: `LLM_Features` was built
the `extract.py` way.

## 7. `mn_to_features.py` — the revised feature variant

Converts an `extract_unified.py` output folder (`mn/` + `manifest.csv`) into
the flat `<family>_<filename>.txt` layout the tokenization pipeline reads,
applying the cohort filter (tag `plain` or `upx_unpacked`, Thanos excluded).
Lines are **mnemonics only** — operands leak the architecture, and the
ransomware side is 96% x86 against 44–75% x64 goodware.

```bash
python asm_tool/mn_to_features.py --extract "<...>/asm and mm/Shared/Extract"     --out "<...>/LLM_Features_Revised/Features_Extraction" --exclude-family thanos
python asm_tool/mn_to_features.py --extract "<...>/asm and mm/Shared/Extract_Goodware_Balanced"     --out "<...>/LLM_Features_Revised_Balanced" --set-dir good_all
```

Produced on 13 September 2026: Mendeley 1,114 / 129 / 904 / 362
(good_train / good_test / mal_train / mal_test; 24 and 14 ransomware
families) and Goodware_Balanced 1,337. Each output folder carries a
`revised_manifest.csv` listing every input row, kept or not, with the reason.

| | traditional (`LLM_Features`) | revised (`LLM_Features_Revised*`) |
|---|---|---|
| extractor | `extract.py` / `asm_parse.py` | `extract_unified.py` |
| sweep | linear, stops at first undecodable byte | skip-data, resumes |
| cap | none / 100,000 | none |
| line | full instruction `mov eax, ebx` | mnemonic `mov` |
| exclusions | .NET, UPX (by extractor) | .NET, UPX, entropy-packed, odd arch, no code, dup, Thanos |

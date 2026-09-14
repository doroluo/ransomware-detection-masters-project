# Goodware_Balanced distribution

Profiled 2026-09-13 with `extract-opcode/profile_packing.py` (tag rules: .NET via COM descriptor, UPX via section name, packed_other = max executable-section entropy > 7.2, no_exec = no executable section with data). 1500 PEs in the three buckets; the 24 packed originals under `_upx_packed/` are listed separately and are not samples.

| tag | everyday | hard_negative | system | total |
|---|---|---|---|---|
| plain | 722 | 330 | 285 | 1337 |
| dotnet | 49 | 7 | 14 | 70 |
| no_exec | 53 | 34 | 0 | 87 |
| packed_other | 1 | 4 | 1 | 6 |
| **total** | 825 | 375 | 300 | **1500** |

| arch | everyday | hard_negative | system | total |
|---|---|---|---|---|
| x86 | 195 | 71 | 109 | 375 (25.0%) |
| x64 | 630 | 304 | 191 | 1125 (75.0%) |

Notes

- UPX was already unpacked in place when the corpus was built; the 24 packed originals live in `_upx_packed/` and are excluded from every manifest here.
- The 87 `no_exec` files are API-set forwarder DLLs (`api-ms-win-*`) shipped inside installers; they contain no code by design.
- All 6 `packed_other` files decode at 92% or better under a skip-data sweep (see `unified_manifest.csv`, column `decoded_ratio`), so the entropy flag is a false positive on this corpus.
- 75.0% x64 overall (73.5% among the 1,343 disassembled files) versus 4% x64 in the Mendeley ransomware training set. Architecture is a shortcut feature; see extract-opcode/WRITEUP.md section 4.

Folder contents

- `everyday/`, `hard_negative/`, `system/`: `asm_parse.py` output, one `<filename>.asm` per disassembled PE, same line format as `../mendeley_goodware/`, capped at 100,000 instructions.
- `asm_manifest.csv`: `asm_parse.py` manifest, one row per input PE (1,500).
- `mn/<sha256>.txt`: mnemonic-only sequences from `extract_unified.py` (skip-data sweep, uncapped, one executable section per line). Join to filenames through `unified_manifest.csv`.
- `unified_manifest.csv`: `extract_unified.py` manifest (sha256, tag, arch, entropy, decoded ratio, instruction count).
- `profile.csv`: raw profile rows behind the tables above.

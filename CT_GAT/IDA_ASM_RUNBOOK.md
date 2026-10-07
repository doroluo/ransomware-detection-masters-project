# Mendeley assembly companion

The owner authorized full text assembly transfer on October 6, 2026. This adds
IDA-generated `.asm` listings alongside the existing CFG JSONL exports. It does
not change the CFG extractor or its provenance hashes. No samples are executed
or copied to the desktop. Output and temporary IDA databases stay under
`~/work/out/`; sample inputs are private, hash-verified copies with execute bits
cleared. The samples in the dataset folders remain unchanged.

`preprocessing/run_mendeley_asm.py` opens each sample with idalib, waits for static
autoanalysis, and invokes IDA's `OFILE_ASM` export over the entire database address
range. Listings include IDA's code and data directives, symbols, operands and
comments. There are no line or address caps. Text is gzip-compressed only during
transfer; the desktop receiver writes ordinary `<sample_sha256>.asm` files.

## Current run

- Scope: Mendeley only, 2,670 unique original file hashes / 2,675 file occurrences.
- VM output: `/home/seed/work/out/ida_asm_mendeley_20261006/`.
- Desktop output: `reports/ida_asm_mendeley_20261006/`.
- `asm/` contains one canonical text listing per successful sample hash.
- `groups/mendeley/goodware/{train,test}/` and
  `groups/mendeley/ransomware/{train,test}/<family>/` preserve folder provenance.
- `manifest.csv`, per-group manifests, `inventory.json`, and `summary.json`
  preserve architecture, input kind, original split, label, family, and status.
  Group listings use hard links where possible, with file-copy fallback.
- Full original cohort membership remains in the CFG inventory and
  `mendeley_samples.csv`; join by SHA-256. Use the CFG reconciliation manifests
  and identity aliases for leakage-aware experiments, not original splits blindly.

Two small goodware/ransomware previews run first. The bulk pass waits for the
Mendeley CFG controller to complete its final verified transfer, then holds its
pipeline lock to avoid overlapping full IDA runs. It stops after Mendeley and
does not process Balanced, VS, or VirusShare.

Every original Mendeley hash is attempted, including managed and other-machine
inputs. A successful listing does not imply useful native x86 instructions:
inspect the `kind`, `arch`, actual IDA `processor`, CFG recovery quality, and
contents. Loader failures, timeouts, and unsupported formats remain explicit
failure rows rather than fabricated `.asm` files.

The VM worker has a 6 GiB process-tree RSS limit, 768 MiB available-memory reserve,
3 GiB free-disk reserve, and 1-hour/4-hour retry budgets. Previews use shorter
2-minute/4-minute budgets; failed previews are retried with full budgets in the
bulk pass. Export packages are acknowledged only after archive, compressed-file,
and expanded-text checksum validation. The receiver rejects unexpected members,
unsafe paths and NUL-containing output. It never evaluates the listing. Only
acknowledged derived VM files are removed; statuses and original samples remain.

## Starting and resuming

Copy the script and its local imports (`vm_ida_cfg.py`, `run_research_cfg.py`,
`ida_cfg_research.py`) to the VM tools directory. Use the existing IDA-enabled
Python environment, not a sample executable. Do not install packages.

```sh
nohup python3 -u /home/seed/work/out/ida_cfg_quality_20261006/tools/run_mendeley_asm.py run --preview-first > /home/seed/work/out/ida_asm_mendeley_20261006/controller.log 2>&1 < /dev/null &
```

Create the output directory before redirecting the log there. On the desktop,
run the receiver under a hidden background Python process with persistent logs:

```sh
python CT_GAT/preprocessing/run_mendeley_asm.py receive --destination reports/ida_asm_mendeley_20261006
```

Both sides resume using per-hash statuses and transfer acknowledgments. Start
only one desktop receiver for the output directory. `pipeline_state.json` on
the VM and `transfer_status.json` on the desktop report progress or errors.

## ML use

Open `.asm` as text in Notepad or VS Code. Keep the archival listing; derive
normalized token sequences separately. Exclude exporter comments, source paths,
hashes, and incidental addresses from baseline model inputs. Compare mnemonic,
operand-aware, API/import, and CFG models using the same identity-disjoint split.
Do not mix managed IL or other processors into a native x86 experiment without
an explicit preprocessing policy. Packed samples remain analyses of their
original packed bytes; an assembly export does not unpack them or recover all
possible runtime control flow.

API reference: https://python.docs.hex-rays.com/ida_loader/index.html#ida_loader.gen_file

## Validation evidence

The host regression suite passed 28 tests. Real IDA Pro 9.4 previews produced
and transferred these listings with verified compressed and plain-text checksums:

| Group | Sample SHA-256 | Lines | Text bytes |
| --- | --- | ---: | ---: |
| Goodware train | `54939b60003d8394cbfc761da90fd73042e80815e28dbcfe6608ff076a70e319` | 2,530 | 98,957 |
| Ransomware train / nefilim | `3bac058dbea51f52ce154fed0325fd835f35c1cd521462ce048b41c9b099e1e5` | 4,354 | 172,959 |

These are smoke-test results, not a claim that the entire assembly dataset is complete.

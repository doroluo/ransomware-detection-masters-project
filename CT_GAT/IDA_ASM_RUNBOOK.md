# Mendeley assembly companion

The owner authorized full text assembly transfer on October 6, 2026. This adds
IDA-generated `.asm` listings alongside the existing CFG JSONL exports. The
combined mode adds assembly export before the CFG worker closes its IDA database;
new CFG files record the updated extractor hash. Existing files keep their original
hashes, and the run policy records the transition. No samples are executed
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

Two small goodware/ransomware previews ran first. At the 1,000-sample checkpoint,
the live run switched to combined extraction. For remaining native candidates,
one IDA analysis produces both the CFG and assembly listing. The assembly
controller transfers finished listings while CFG extraction continues. Once the
CFG controller completes its final verified transfer, the assembly controller
holds its pipeline lock and backfills only missing listings. These include earlier
samples and managed/other inputs that did not open IDA during CFG extraction.
It stops after Mendeley and does not process Balanced, VS, or VirusShare.

`combined_asm.json` in the CFG output directory enables this behavior and names
the assembly output directory (`{"out":"/home/seed/work/out/ida_asm_mendeley_20261006"}`).
The combined path requires a Mendeley inventory row. Shared listings have
`analysis_mode=shared_cfg_database` and the CFG extractor hash in their per-hash
assembly status. Publication occurs only after the listing is complete and hashed.
An assembly-generation exception leaves the valid CFG intact, records a separate
error, and leaves the listing eligible for backfill. Resource-limit termination
normally follows the CFG worker retry policy. When an unfinished assembly listing
hits the disk reserve after its CFG is complete, the controller validates and
preserves the CFG, removes only incomplete assembly temporary files, and records
`status=deferred`, `reason=assembly_disk_reserve`. Automatic backfill skips that
listing until its export strategy is changed; it is never counted as successful.
If cleanup does not restore the disk reserve, the run still stops safely.

On October 7, sample
`bbbf38de4f40754f235441a8e6a4c8bdb9365dab7f5cfcdac77dbb4d6236360b`
left a 13,075,230,720-byte incomplete assembly listing and stopped the original
controller at 2,100 completed samples. That temporary file was removed, its ASM
was marked deferred, and its CFG was successfully re-extracted (420 functions,
3,717 blocks, 17,030 function instructions). Completed desktop outputs and source
datasets were preserved; extraction resumed with about 15 GiB free in the VM.

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
`extract_cfg_with_ida.py`) to the VM tools directory. Use the existing IDA-enabled
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

The host regression suite passed 32 tests, including preservation of completed
CFGs and prevention of automatic repeated oversized assembly exports.
Real IDA Pro 9.4 previews produced
and transferred these listings with verified compressed and plain-text checksums:

| Group | Sample SHA-256 | Lines | Text bytes |
| --- | --- | ---: | ---: |
| Goodware train | `54939b60003d8394cbfc761da90fd73042e80815e28dbcfe6608ff076a70e319` | 2,530 | 98,957 |
| Ransomware train / nefilim | `3bac058dbea51f52ce154fed0325fd835f35c1cd521462ce048b41c9b099e1e5` | 4,354 | 172,959 |

These are smoke-test results, not a claim that the entire assembly dataset is complete.

Combined-mode smoke tests of those same samples verified matching sample and CFG
extractor hashes, nonempty CFG instructions, and real `OFILE_ASM` listings from the
same open database. They produced 50 functions / 198 blocks for goodware and
33 functions / 332 blocks for ransomware. Assembly export added 0.028 and 0.049
seconds respectively; these small-sample timings are not a whole-dataset estimate.

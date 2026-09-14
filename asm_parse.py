#!/usr/bin/env python3
"""
asm_parse.py - disassemble PE executables into .asm text for the CNN-ViT pipeline.

Output line format (consumed by asm_parser.py's parse_asm_line):

    0x00401000:  mov\teax, ebx

Directory structure is mirrored from the input tree, so files with the same
name in different class folders do not collide.

Decoding semantics are deliberately identical to extract.py: a linear capstone
sweep over IMAGE_SCN_MEM_EXECUTE sections only, no `skipdata`, so the sweep of
a section ENDS at the first undecodable byte. LLM_Features (the ransomware
side of the experiments) was built that way, so changing it here would make the
goodware streams systematically different from the ransomware streams and hand
the classifier a shortcut. See asm_tool/README.md section 6 for the measured
cost of that choice.

Guards match extract.py: .NET assemblies and UPX-packed binaries are skipped
rather than silently producing meaningless instructions, and only x86/x64 is
supported. Packing heuristics are recorded in the manifest as flags only; they
never change which files get disassembled.

Usage:
    python asm_parse.py --in-dir  ../Goodware_Balanced \
                        --out-dir ../asm_output/goodware_balanced
    python asm_parse.py --in-dir X --out-dir Y --limit 50      # smoke test
    python asm_parse.py --in-dir X --out-dir Y --refresh-manifest
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import math
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64

IMAGE_SCN_MEM_EXECUTE = 0x20000000
SKIP_DIRS = {"_upx_packed", ".tools", "flagged", "quarantined"}
SKIP_SUFFIXES = {".csv", ".md", ".json"}

# asm_parser.py flattens each instruction to ~3 tokens and keeps only the first
# 256*256 = 65,536 of them, i.e. roughly the first 21,800 instructions. A cap
# well above that costs the model nothing and keeps the corpus shippable:
# uncapped, this corpus disassembles to ~31 GB, with one 185 MB binary alone
# producing millions of lines.
DEFAULT_MAX_INSN = 100_000

MANIFEST_NAME = "asm_manifest.csv"

# Manifest columns. The first five are the original set and keep their exact
# meaning, because asm_to_opcodes.py, the tests and the DISTRIBUTION tables
# were written against them. Everything after `instructions` is additive and
# comes from the worker brief's per-file row spec
# {sha256, machine, bitness, is_dotnet, packed_flag, packer_guess,
#  sections_disassembled, n_instructions, status, error}.
# `n_instructions` is the brief's name for `instructions`; both are written so
# either name resolves.
FIELDS = [
    "rel_path", "asm_path", "status", "arch", "instructions",
    "sha256", "machine", "bitness", "is_dotnet", "packed_flag",
    "packer_guess", "sections_disassembled", "n_instructions",
    "error", "source",
]

# Section names that identify a packer. Matched case-insensitively against the
# NUL-stripped section name. Flag only - a hit never changes what is
# disassembled.
PACKER_SECTIONS = (
    ("upx", ("upx0", "upx1", "upx2", "upx!", ".upx")),
    ("aspack", (".aspack", ".adata")),
    ("themida", (".themida", "winlicen", ".winlice")),
    ("vmprotect", (".vmp0", ".vmp1", ".vmp2")),
    ("mpress", (".mpress1", ".mpress2")),
    ("pecompact", ("pec2", "pec1", "pecompact2")),
    ("petite", (".petite",)),
    ("nspack", (".nsp0", ".nsp1", ".nsp2", "nsp0", "nsp1", "nsp2")),
    ("enigma", (".enigma1", ".enigma2")),
    ("fsg", (".fsg",)),
    ("mew", ("mew", ".mew")),
    ("upack", (".upack", ".bydamn")),
    ("yoda", (".yp", ".y0da")),
    ("rlpack", (".rlpack",)),
    ("wwpack", (".wwpack", ".wwp32")),
    ("molebox", (".molebox",)),
    ("neolite", ("neolit", "neolite")),
    ("kkrunchy", ("kkrunchy",)),
    ("crinkler", (".crinkle",)),
)

# Same threshold the reference extractor (extract-opcode/extract_unified.py)
# and profile_packing.py use, so the two packing tables are comparable.
PACKED_ENTROPY = 7.2


def shannon(buf: bytes) -> float:
    """Shannon entropy of a byte buffer, in bits per byte.

    collections.Counter is ~5x faster than pefile's 256 x bytes.count() on a
    multi-megabyte section and gives the identical histogram.
    """
    n = len(buf)
    if not n:
        return 0.0
    ent = 0.0
    for c in collections.Counter(buf).values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def section_name(sec) -> str:
    return sec.Name.rstrip(b"\x00").decode("latin-1", "replace").strip().lower()


def packing_profile(pe) -> tuple[int, str, float]:
    """-> (packed_flag, packer_guess, max executable-section entropy).

    Three independent signals, each recorded in packer_guess:
      * a section name a known packer writes,
      * max executable-section Shannon entropy > 7.2,
      * an entry point that falls outside every executable section.
    """
    evidence: list[str] = []

    names = {section_name(s) for s in pe.sections}
    for packer, markers in PACKER_SECTIONS:
        if names & set(markers):
            evidence.append(packer)

    exec_secs = [s for s in pe.sections
                 if s.Characteristics & IMAGE_SCN_MEM_EXECUTE]
    max_ent = 0.0
    for s in exec_secs:
        try:
            data = s.get_data()
        except Exception:
            continue
        if data:
            max_ent = max(max_ent, shannon(data))
    if max_ent > PACKED_ENTROPY:
        evidence.append("high_entropy")

    # An entry point of 0 is normal for resource-only DLLs and is not a
    # packing signal; only test it when there is code to point at.
    try:
        ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    except AttributeError:
        ep = 0
    if ep and exec_secs:
        inside = any(
            s.VirtualAddress <= ep
            < s.VirtualAddress + max(s.Misc_VirtualSize, s.SizeOfRawData)
            for s in exec_secs)
        if not inside:
            evidence.append("ep_outside_exec")

    return int(bool(evidence)), "+".join(evidence), round(max_ent, 3)


def blank_record() -> dict:
    return {"rel_path": "", "asm_path": "", "status": "", "arch": "",
            "instructions": "", "sha256": "", "machine": "", "bitness": "",
            "is_dotnet": "", "packed_flag": "", "packer_guess": "",
            "sections_disassembled": "", "n_instructions": "",
            "error": "", "source": ""}


def analyze(path: Path, max_insn: int, want_lines: bool = True) -> tuple[dict, list[str]]:
    """Full per-file record plus the .asm lines.

    The status ladder and the decoding are byte-for-byte the behaviour of the
    previous version; everything else on the record is new metadata that does
    not affect it.
    """
    rec = blank_record()
    path = Path(path)
    rec["sha256"] = sha256_of(path)

    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"MZ":
                rec["status"] = "not_pe"
                rec["n_instructions"] = rec["instructions"] = 0
                return rec, []
    except OSError as exc:
        rec["status"] = f"read_error:{type(exc).__name__}"
        rec["error"] = str(exc)
        rec["n_instructions"] = rec["instructions"] = 0
        return rec, []

    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]])
    except Exception as exc:
        rec["status"] = f"pe_error:{type(exc).__name__}"
        rec["error"] = str(exc)[:200]
        rec["n_instructions"] = rec["instructions"] = 0
        return rec, []

    try:
        machine = pe.FILE_HEADER.Machine
        rec["machine"] = f"0x{machine:x}"
        rec["bitness"] = {0x14C: 32, 0x8664: 64}.get(machine, "")

        try:
            cd = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]]
            rec["is_dotnet"] = int(bool(cd.VirtualAddress and cd.Size))
        except (IndexError, AttributeError, KeyError):
            rec["is_dotnet"] = 0

        try:
            flag, guess, _ent = packing_profile(pe)
            rec["packed_flag"], rec["packer_guess"] = flag, guess
        except Exception as exc:                       # metadata must never
            rec["packed_flag"], rec["packer_guess"] = "", ""   # break a run
            rec["error"] = f"packing_profile:{type(exc).__name__}"

        # .NET: an IL-only assembly's executable section holds CIL bytecode,
        # not x86, and its native entry point is a ~6-byte stub - capstone
        # would decode CIL as x86 and emit plausible-looking nonsense.
        #
        # Mixed-mode assemblies (C++/CLI) are different: they carry real
        # native code alongside the CIL, so they are disassembled normally.
        # The distinction is COMIMAGE_FLAGS_ILONLY in the CLR header, not the
        # mere presence of a COM descriptor.
        try:
            cd = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]]
            if cd.VirtualAddress and cd.Size:
                clr = pe.get_data(cd.VirtualAddress, 72)
                il_only = struct.unpack_from("<I", clr, 16)[0] & 0x1
                if il_only:
                    rec["status"] = "dotnet_ilonly"
                    rec["n_instructions"] = rec["instructions"] = 0
                    rec["sections_disassembled"] = 0
                    return rec, []
        except (IndexError, AttributeError, KeyError, struct.error) as exc:
            rec["status"] = "dotnet_ilonly"
            rec["error"] = f"clr_header:{type(exc).__name__}"
            rec["n_instructions"] = rec["instructions"] = 0
            rec["sections_disassembled"] = 0
            return rec, []

        if any(b"UPX" in s.Name for s in pe.sections):
            rec["status"] = "upx_packed"
            rec["n_instructions"] = rec["instructions"] = 0
            rec["sections_disassembled"] = 0
            return rec, []

        if machine == 0x14C:
            mode, arch = CS_MODE_32, "x86"
        elif machine == 0x8664:
            mode, arch = CS_MODE_64, "x64"
        else:
            rec["status"] = f"arch_unsupported:{hex(machine)}"
            rec["n_instructions"] = rec["instructions"] = 0
            rec["sections_disassembled"] = 0
            return rec, []
        rec["arch"] = arch

        md = Cs(CS_ARCH_X86, mode)
        base = pe.OPTIONAL_HEADER.ImageBase
        lines: list[str] = []
        found_code = False
        n_sec = 0

        for section in pe.sections:
            if not (section.Characteristics & IMAGE_SCN_MEM_EXECUTE):
                continue
            found_code = True
            va = base + section.VirtualAddress
            before = len(lines)
            # No skipdata, by design: this sweep stops at the first byte it
            # cannot decode, exactly as extract.py does.
            for insn in md.disasm(section.get_data(), va):
                lines.append(f"0x{insn.address:08x}:  {insn.mnemonic}\t{insn.op_str}")
                if max_insn and len(lines) >= max_insn:
                    n_sec += 1
                    rec["status"] = "truncated"
                    rec["instructions"] = rec["n_instructions"] = len(lines)
                    rec["sections_disassembled"] = n_sec
                    return rec, (lines if want_lines else [])
            if len(lines) > before:
                n_sec += 1

        rec["sections_disassembled"] = n_sec
        if not found_code:
            rec["status"] = "no_exec_section"
            rec["instructions"] = rec["n_instructions"] = 0
            return rec, []
        if not lines:
            rec["status"] = "empty_disassembly"
            rec["instructions"] = rec["n_instructions"] = 0
            return rec, []
        rec["status"] = "ok"
        rec["instructions"] = rec["n_instructions"] = len(lines)
        return rec, (lines if want_lines else [])
    finally:
        try:
            pe.close()
        except Exception:
            pass


def disassemble(path: Path, max_insn: int) -> tuple[str, list[str], str, int]:
    """Returns (status, lines, arch, n_instructions).

    Kept as the stable four-tuple API: tests/test_asm_parse.py and
    asm_tool/consistency_check.py are written against it.
    """
    rec, lines = analyze(Path(path), max_insn)
    n = rec["n_instructions"]
    return rec["status"], lines, rec["arch"], (n if isinstance(n, int) else 0)


def _count_lines(path: Path) -> int | str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return ""


def _resume_record(src: Path, dest: Path, max_insn: int, prior: dict | None) -> dict:
    """Row for a file that already has a .asm and is not being redone.

    The whole point: a resumed run must reproduce the row the original run
    wrote, never a placeholder. Status comes from the prior manifest when one
    exists; otherwise it is re-derived from the file (a .asm at exactly the cap
    was truncated, anything shorter completed). Header-derived metadata is
    always recomputed, so re-running also *upgrades* a manifest written by an
    older version of this script.
    """
    rec = blank_record()
    rec["source"] = "resumed"
    rec["sha256"] = sha256_of(src)

    try:
        pe = pefile.PE(str(src), fast_load=True)
        try:
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]])
            machine = pe.FILE_HEADER.Machine
            rec["machine"] = f"0x{machine:x}"
            rec["arch"] = {0x14C: "x86", 0x8664: "x64"}.get(machine, "")
            rec["bitness"] = {0x14C: 32, 0x8664: 64}.get(machine, "")
            try:
                cd = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]]
                rec["is_dotnet"] = int(bool(cd.VirtualAddress and cd.Size))
            except (IndexError, AttributeError, KeyError):
                rec["is_dotnet"] = 0
            flag, guess, _ = packing_profile(pe)
            rec["packed_flag"], rec["packer_guess"] = flag, guess
        finally:
            pe.close()
    except Exception as exc:
        rec["error"] = f"resume_header:{type(exc).__name__}"

    n = _count_lines(dest)
    rec["instructions"] = rec["n_instructions"] = n

    prior_status = (prior or {}).get("status", "")
    if prior_status and prior_status not in ("already_done", ""):
        rec["status"] = prior_status
        if not rec["error"]:
            rec["error"] = (prior or {}).get("error", "") or ""
        prior_sec = (prior or {}).get("sections_disassembled", "")
        if prior_sec not in ("", None):
            rec["sections_disassembled"] = prior_sec
    else:
        rec["status"] = "truncated" if (max_insn and n == max_insn) else "ok"
    return rec


def process(args_tuple) -> dict:
    src, rel, out_dir, max_insn, mode, prior = args_tuple
    src, out_dir = Path(src), Path(out_dir)
    # Append .asm rather than replacing the extension: foo.exe and foo.dll
    # both live in the same class folder and would otherwise collide on
    # foo.asm. Keeping the original extension also makes the .asm basename
    # equal the source PE filename, which is what asm_parser.py matches
    # against when it builds labels.
    dest = out_dir / (rel + ".asm")

    if mode == "resume" and dest.exists():
        rec = _resume_record(src, dest, max_insn, prior)
        rec["rel_path"] = rel
        rec["asm_path"] = str(dest.relative_to(out_dir))
        return rec

    rec, lines = analyze(src, max_insn)
    rec["rel_path"] = rel
    rec["asm_path"] = ""

    if mode == "refresh":
        # Manifest-only: recompute every field from the binary but never touch
        # the .asm tree. The disassembly we just produced in memory is compared
        # against what is on disk, so a refresh also proves the tree still
        # matches the code that wrote it.
        rec["source"] = "refreshed"
        if lines:
            if dest.exists():
                rec["asm_path"] = str(dest.relative_to(out_dir))
                on_disk = dest.read_text(encoding="utf-8", errors="replace")
                if on_disk != "\n".join(lines):
                    rec["error"] = (rec["error"] + ";" if rec["error"] else "") + \
                                   "asm_mismatch"
            else:
                rec["error"] = (rec["error"] + ";" if rec["error"] else "") + \
                               "asm_missing"
        elif dest.exists():
            rec["asm_path"] = str(dest.relative_to(out_dir))
            rec["error"] = (rec["error"] + ";" if rec["error"] else "") + \
                           "asm_unexpected"
        return rec

    rec["source"] = "disassembled"
    if lines:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines), encoding="utf-8")
        rec["asm_path"] = str(dest.relative_to(out_dir))
    return rec


def load_prior_manifest(path: Path) -> dict[str, dict]:
    """Existing manifest keyed by rel_path, for the resume path."""
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return {r["rel_path"]: r for r in csv.DictReader(fh)
                    if r.get("rel_path")}
    except (OSError, csv.Error):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="tree of PE files")
    ap.add_argument("--out-dir", required=True, help="where .asm files go")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (0 = all)")
    ap.add_argument("--max-instructions", type=int, default=DEFAULT_MAX_INSN,
                    help=f"cap per file (0 = uncapped; default {DEFAULT_MAX_INSN}). "
                         "asm_parser.py consumes only ~21,800 instructions.")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--overwrite", action="store_true",
                    help="redo files that already have a .asm")
    ap.add_argument("--refresh-manifest", action="store_true",
                    help="rewrite the manifest only: re-read every binary, "
                         "recompute every column and verify the existing .asm "
                         "matches, without writing a single .asm file")
    args = ap.parse_args()

    if args.overwrite and args.refresh_manifest:
        sys.exit("--overwrite and --refresh-manifest are mutually exclusive")
    mode = ("refresh" if args.refresh_manifest
            else "overwrite" if args.overwrite else "resume")

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not in_dir.is_dir():
        sys.exit(f"input directory not found: {in_dir}")
    if mode == "refresh" and not out_dir.is_dir():
        sys.exit(f"--refresh-manifest needs an existing output tree: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / MANIFEST_NAME
    prior = load_prior_manifest(manifest) if mode == "resume" else {}

    tasks = []
    for path in sorted(in_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(in_dir)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        tasks.append((str(path), str(rel), str(out_dir),
                      args.max_instructions, mode, prior.get(str(rel))))
        if args.limit and len(tasks) >= args.limit:
            break

    print(f"{len(tasks)} candidate files under {in_dir}")
    if mode == "refresh":
        print(f"refreshing {manifest} only - no .asm file will be written\n")
    else:
        print(f"writing .asm to {out_dir} (mirrored structure, "
              f"cap {args.max_instructions or 'none'} instructions, "
              f"{args.jobs} workers, mode={mode})\n")

    results, done = [], 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process, t): t[1] for t in tasks}
        for fut in as_completed(futures):
            rel = futures[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:
                rec = blank_record()
                rec.update({"rel_path": rel,
                            "status": f"crash:{type(exc).__name__}",
                            "error": str(exc)[:200], "source": mode})
            results.append(rec)
            if done % 100 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)

    results.sort(key=lambda r: r["rel_path"])
    tmp = manifest.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in results:
            w.writerow(rec)
    os.replace(tmp, manifest)

    counts: dict[str, int] = {}
    for rec in results:
        key = rec["status"].split(":")[0]
        counts[key] = counts.get(key, 0) + 1

    print(f"\nmanifest: {manifest}  ({len(results)} rows, {len(FIELDS)} columns)")
    print("results:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20}{v:>6}")
    packed = sum(1 for r in results if r["packed_flag"] == 1)
    dotnet = sum(1 for r in results if r["is_dotnet"] == 1)
    errs = sum(1 for r in results if r["error"])
    print(f"  {'packed_flag=1':<20}{packed:>6}")
    print(f"  {'is_dotnet=1':<20}{dotnet:>6}")
    print(f"  {'error non-empty':<20}{errs:>6}")
    written = sum(1 for r in results if r["asm_path"])
    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.asm"))
    print(f"\n{written} .asm files, {total_bytes/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

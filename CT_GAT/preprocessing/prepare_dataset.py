"""Interactive tool:
  (1) Disassemble goodware/ + ransomware/ exes into data/asm/*.asm
  (2) Build metadata.csv from .asm files + folder labels
Both modes accept a per-class sample limit.
"""

from __future__ import annotations


import argparse
import csv
import re
from pathlib import Path


import pefile
from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs
from capstone.x86 import (
    X86_OP_IMM,
    X86_OP_MEM,
    X86_REG_EIP,
    X86_REG_INVALID,
    X86_REG_RIP,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # CT_GAT/
REPO_ROOT = PROJECT_ROOT.parent  # extract/


GOODWARE_DIR = REPO_ROOT / "goodware"
RANSOMWARE_DIR = REPO_ROOT / "ransomware"
ASM_DIR = PROJECT_ROOT / "data" / "asm"
OUT_CSV = PROJECT_ROOT / "data" / "metadata.csv"




def iter_pe_files(directory: Path):
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.read_bytes()[:2] != b"MZ":
                continue
        except OSError:
            continue
        yield path




_API_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_@]*")




def _canon_addr(addr: int, bits: int) -> int:
    return addr & ((1 << bits) - 1)




def _api_token(raw) -> str | None:
    """Keep a single operand token. Comments are stripped by the parsers."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
    name = str(raw).strip()
    if _API_TOKEN.fullmatch(name):
        return name
    return None




def collect_import_names(pe, bits: int) -> dict[int, str]:
    """Map each import-address-table slot (virtual address) to its function name."""
    names: dict[int, str] = {}
    for attr in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        table = getattr(pe, attr, None)
        if not table:
            continue
        for entry in table:
            for imp in getattr(entry, "imports", ()):
                token = _api_token(getattr(imp, "name", None))
                address = getattr(imp, "address", None)
                if token is None or address is None:
                    continue
                names[_canon_addr(int(address), bits)] = token
    return names




def _mem_target(insn, bits: int) -> int | None:
    """Absolute address of a simple [disp] or [rip+disp] operand."""
    operands = insn.operands
    if not operands or operands[0].type != X86_OP_MEM:
        return None
    mem = operands[0].mem
    if mem.index != X86_REG_INVALID:
        return None
    if mem.base in (X86_REG_RIP, X86_REG_EIP):
        addr = insn.address + insn.size + mem.disp
    elif mem.base == X86_REG_INVALID:
        addr = mem.disp
    else:
        return None
    return _canon_addr(addr, bits)




def _call_immediate(insn, bits: int) -> int | None:
    operands = insn.operands
    if not operands or operands[0].type != X86_OP_IMM:
        return None
    return _canon_addr(operands[0].imm, bits)




def disassemble_pe_to_asm(path: Path) -> str:
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories()


    machine = pe.FILE_HEADER.Machine
    if machine == 0x14C:
        mode = CS_MODE_32
        bits = 32
    elif machine == 0x8664:
        mode = CS_MODE_64
        bits = 64
    else:
        raise ValueError(f"Unsupported machine: 0x{machine:x}")


    try:
        imports = collect_import_names(pe, bits)
    except Exception:
        imports = {}


    md = Cs(CS_ARCH_X86, mode)
    md.detail = True
    base = pe.OPTIONAL_HEADER.ImageBase
    instructions = []


    for section in pe.sections:
        if not (section.Characteristics & 0x20):
            continue
        code = section.get_data()
        va = base + section.VirtualAddress
        instructions.extend(md.disasm(code, va))


    # MSVC calls a one-instruction thunk: jmp dword ptr [IAT].
    thunks: dict[int, str] = {}
    for insn in instructions:
        if insn.mnemonic != "jmp":
            continue
        target = _mem_target(insn, bits)
        if target is not None and target in imports:
            thunks[_canon_addr(insn.address, bits)] = imports[target]


    lines: list[str] = []
    for insn in instructions:
        operand = insn.op_str
        if insn.mnemonic == "jmp":
            target = _mem_target(insn, bits)
            if target is not None and target in imports:
                operand = imports[target]
        elif insn.mnemonic == "call":
            mem_target = _mem_target(insn, bits)
            if mem_target is not None and mem_target in imports:
                operand = imports[mem_target]
            else:
                immediate = _call_immediate(insn, bits)
                if immediate is not None and immediate in thunks:
                    operand = thunks[immediate]
        lines.append(f"0x{insn.address:08x}:  {insn.mnemonic}\t{operand}")


    return "\n".join(lines)




def mode_exes_to_asm(limit_per_class: int | None) -> None:
    """Mode 1: goodware/ransomware exes -> data/asm/*.asm"""
    ASM_DIR.mkdir(parents=True, exist_ok=True)
    total = 0


    for directory, name in (
        (GOODWARE_DIR, "goodware"),
        (RANSOMWARE_DIR, "ransomware"),
    ):
        if not directory.is_dir():
            print(f"Missing folder: {directory}")
            continue


        print(f"\n=== {name} -> .asm (limit={limit_per_class}) ===")
        count = 0
        failed = 0


        for path in iter_pe_files(directory):
            if limit_per_class is not None and count >= limit_per_class:
                break
            try:
                text = disassemble_pe_to_asm(path)
                if not text.strip():
                    print(f"Skipped empty: {path}")
                    continue
                out_file = ASM_DIR / f"{path.stem}.asm"
                out_file.write_text(text, encoding="utf-8")
                print(f"Wrote {out_file.name}")
                count += 1
                total += 1
            except Exception as error:
                print(f"Failed: {path}")
                print("Error:", error)
                failed += 1


        print(f"{name}: wrote {count}, failed {failed}")


    print(f"\nDone. Wrote {total} .asm files to {ASM_DIR}")




def index_executables(directory: Path, label: int) -> dict[str, tuple[Path, int]]:
    index: dict[str, tuple[Path, int]] = {}
    for path in directory.rglob("*"):
        if path.is_file():
            index[path.stem] = (path, label)
    return index




def mode_asm_plus_labels(limit_per_class: int | None) -> None:
    """Mode 2: data/asm/*.asm + folder labels -> metadata.csv"""
    if not ASM_DIR.is_dir():
        raise FileNotFoundError(
            f"ASM directory not found: {ASM_DIR}\n"
            "Run mode 1 first, or place .asm files there."
        )


    exe_index: dict[str, tuple[Path, int]] = {}
    exe_index.update(index_executables(GOODWARE_DIR, 0))
    exe_index.update(index_executables(RANSOMWARE_DIR, 1))


    rows: list[dict] = []
    missing: list[str] = []
    count_by_label = {0: 0, 1: 0}


    for asm_path in sorted(ASM_DIR.glob("*.asm")):
        file_id = asm_path.stem
        hit = exe_index.get(file_id)
        if hit is None:
            missing.append(file_id)
            continue


        exe_path, label = hit
        if (
            limit_per_class is not None
            and count_by_label[label] >= limit_per_class
        ):
            continue


        rows.append(
            {
                "file_id": file_id,
                "exe_path": str(exe_path.resolve()),
                "asm_path": str(asm_path.resolve()),
                "label": label,
            }
        )
        count_by_label[label] += 1


    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_id", "exe_path", "asm_path", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)


    print(
        f"Wrote {len(rows)} rows to {OUT_CSV} "
        f"({count_by_label[0]} benign, {count_by_label[1]} ransomware)"
    )
    if missing:
        print(
            f"Skipped {len(missing)} asm files with no matching exe "
            f"(examples: {missing[:5]})"
        )




def prompt_mode() -> int:
    print("Select mode:")
    print("  1) Exe folders -> .asm          (goodware/ + ransomware/)")
    print("  2) .asm + labels -> metadata.csv")
    while True:
        raw = input("Mode [1/2]: ").strip()
        if raw in {"1", "2"}:
            return int(raw)
        print("Please enter 1 or 2.")




def prompt_limit() -> int | None:
    raw = input("Samples per class (Enter = all): ").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit("Limit must be an integer.")
    if value <= 0:
        raise SystemExit("Limit must be > 0.")
    return value




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare .asm files and/or labeled metadata for Transformer/GAT."
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2],
        help="1 = exes to .asm, 2 = .asm + labels to metadata.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per class (default: all)",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    if args.mode is None:
        mode = prompt_mode()
        limit = prompt_limit()
    else:
        mode = args.mode
        limit = args.limit


    print(f"\nMode: {mode} | Limit per class: {limit}")
    print(f"Goodware:    {GOODWARE_DIR}")
    print(f"Ransomware:  {RANSOMWARE_DIR}")
    print(f"ASM dir:     {ASM_DIR}")
    print(f"Metadata:    {OUT_CSV}")


    if mode == 1:
        mode_exes_to_asm(limit)
    else:
        mode_asm_plus_labels(limit)




if __name__ == "__main__":
    main()





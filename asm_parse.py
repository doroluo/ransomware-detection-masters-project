from pathlib import Path

import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64


# ---------------------------------------------------------------------
# 1. Modify the folder path to parse the executables and save the 
#    disassembly results to the output directory.
# ---------------------------------------------------------------------

# Goodware Executables
# IN_DIR = Path("/home/yl/quarantine/extract/goodware")
# OUT_DIR = Path("/home/yl/quarantine/extract/dataset_asm")
# LIMIT = 50

# Ransomware Executables
# IN_DIR = Path("/home/yl/quarantine/extract/ransomware")
# OUT_DIR = Path("/home/yl/quarantine/extract/dataset_asm")
# LIMIT = 50

OUT_DIR.mkdir(parents=True, exist_ok=True)
count = 0

for path in IN_DIR.rglob("*"):
    if count >= LIMIT:
        break
    if not path.is_file():
        continue
    if path.read_bytes()[:2] != b"MZ":
        continue

    pe = pefile.PE(str(path))
    if pe.FILE_HEADER.Machine == 0x14c:
        mode = CS_MODE_32
    elif pe.FILE_HEADER.Machine == 0x8664:
        mode = CS_MODE_64
    else:
        continue

    md = Cs(CS_ARCH_X86, mode)
    lines = []
    base = pe.OPTIONAL_HEADER.ImageBase

    for section in pe.sections:
        if not (section.Characteristics & 0x20):
            continue
        code = section.get_data()
        va = base + section.VirtualAddress
        for insn in md.disasm(code, va):
            lines.append(f"0x{insn.address:08x}:  {insn.mnemonic}\t{insn.op_str}")

    out_file = OUT_DIR / f"{path.stem}.asm"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_file}")
    count += 1

print(f"Done. Wrote {count} files.")

#!/usr/bin/env python3
"""
opcode_sequence_builder.py

Purpose:
- Build a small dataset from local malware/goodware assembly or disassembly text files.
- Convert opcodes into:
    1) raw opcode sequences
    2) grouped opcode tokens (MOVE, JUMP, CALL, etc.)
    3) coarse behavior tokens (MEMORY_RW, FILE_OPEN_LIKE, CONTROL_FLOW, etc.)
- Export tokenized samples for downstream ML / graph2vec experiments.

Expected folder layout:
    dataset_root/
        malware/
            sample1.asm
            ...
        goodware/
            benign1.asm
            ...

Typical usage:
    python tokenizer.py ./dataset_root --output ./out
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Opcode grouping
# ----------------------------

OPCODE_GROUPS = {
    "MOVE": {
        "mov", "movzx", "movsx", "lea", "xchg", "push", "pop", "pushad", "popad",
        "pushfd", "popfd", "cmovz", "cmovnz", "cmova", "cmovb", "cmovg", "cmovl"
    },
    "ARITH": {
        "add", "sub", "inc", "dec", "mul", "imul", "div", "idiv", "adc", "sbb",
        "neg", "cmp"
    },
    "LOGIC": {
        "and", "or", "xor", "not", "test"
    },
    "SHIFT_ROTATE": {
        "shl", "shr", "sar", "sal", "rol", "ror", "rcl", "rcr"
    },
    "JUMP": {
        "jmp", "je", "jne", "jz", "jnz", "ja", "jb", "jg", "jl", "jge", "jle",
        "jo", "jno", "js", "jns", "jc", "jnc", "loop", "loope", "loopne"
    },
    "CALL": {
        "call"
    },
    "RET": {
        "ret", "retn", "retf"
    },
    "STRING": {
        "movs", "movsb", "movsw", "movsd", "movsq",
        "stos", "stosb", "stosw", "stosd", "stosq",
        "lods", "lodsb", "lodsw", "lodsd", "lodsq",
        "scas", "scasb", "scasw", "scasd", "scasq",
        "cmps", "cmpsb", "cmpsw", "cmpsd", "cmpsq"
    },
    "FLAGS": {
        "clc", "stc", "cmc", "cld", "std", "cli", "sti", "lahf", "sahf"
    },
    "STACK_FRAME": {
        "enter", "leave"
    },
    "SYSTEM": {
        "int", "syscall", "sysenter", "sysexit", "cpuid", "hlt", "nop", "ud2"
    },
    "FLOAT_SIMD": {
        "fld", "fst", "fstp", "fadd", "fsub", "fmul", "fdiv",
        "pxor", "movdqa", "movdqu", "movaps", "movups", "paddb", "paddw", "paddd"
    },
    "CRYPTO": {
        "aesenc", "aesenclast", "aesdec", "aesdeclast", "aesimc", "aeskeygenassist",
        "rdrand", "rdseed"
    },
}

OPCODE_TO_GROUP: Dict[str, str] = {}
for group_name, ops in OPCODE_GROUPS.items():
    for op in ops:
        OPCODE_TO_GROUP[op] = group_name

# ----------------------------
# Heuristic behavior grouping
# ----------------------------

# These are coarse, static-analysis-like buckets.
GROUP_TO_BEHAVIOR = {
    "MOVE": "MEMORY_RW",
    "ARITH": "DATA_TRANSFORM",
    "LOGIC": "DATA_TRANSFORM",
    "SHIFT_ROTATE": "DATA_TRANSFORM",
    "JUMP": "CONTROL_FLOW",
    "CALL": "API_OR_SUBROUTINE",
    "RET": "CONTROL_FLOW",
    "STRING": "BUFFER_OR_STRING_OP",
    "FLAGS": "CONTROL_FLOW",
    "STACK_FRAME": "FUNCTION_BOUNDARY",
    "SYSTEM": "SYSTEM_INTERACTION",
    "FLOAT_SIMD": "COMPUTE",
    "CRYPTO": "CRYPTO_OP",
}

# API-ish names and imported symbol hints often seen in disassembly text.
FILE_API_HINTS = {
    "createfile", "readfile", "writefile", "setfilepointer", "closehandle",
    "findfirstfile", "findnextfile", "deletefile", "copyfile", "movefile",
    "ntcreatefile", "ntreadfile", "ntwritefile", "fopen", "fread", "fwrite",
    "open", "read", "write", "close"
}

MEMORY_API_HINTS = {
    "virtualalloc", "virtualfree", "virtualprotect", "heapalloc", "heapfree",
    "malloc", "free", "memcpy", "memmove", "memset", "rtlmovememory",
    "mapviewoffile", "unmapviewoffile"
}

PROCESS_API_HINTS = {
    "createprocess", "openprocess", "terminateprocess", "writeprocessmemory",
    "readprocessmemory", "createremotethread", "winexec", "shellexecute",
    "loadlibrary", "getprocaddress"
}

REGISTRY_API_HINTS = {
    "regopenkey", "regsetvalue", "regcreatekey", "regqueryvalue", "regdeletekey"
}

NETWORK_API_HINTS = {
    "socket", "connect", "send", "recv", "wsastartup", "internetopen",
    "internetconnect", "httpopenrequest", "httpsendrequest", "urlopen"
}

CRYPTO_API_HINTS = {
    "cryptencrypt", "cryptdecrypt", "cryptacquirecontext", "bcryptencrypt",
    "bcryptdecrypt"
}

ALL_API_HINTS = (
    FILE_API_HINTS
    | MEMORY_API_HINTS
    | PROCESS_API_HINTS
    | REGISTRY_API_HINTS
    | NETWORK_API_HINTS
    | CRYPTO_API_HINTS
)

# ----------------------------
# Parsing helpers
# ----------------------------

LABEL_RE = re.compile(r"^\s*[A-Za-z_.$?@][\w.$?@]*:\s*$")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_@?$.:]*")
MNEMONIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
HEX_OR_ADDR_RE = re.compile(r"^(0x[0-9a-f]+|[0-9a-f]+h|[0-9a-f]{2,})$", re.I)

NON_OPCODE_TOKENS = {
    "db", "dw", "dd", "dq", "dt",
    "align", "assume", "end", "ends", "segment", "proc", "endp",
    "public", "extrn", "extern", "model", "include", "equ",
}

VALID_EXTENSIONS = {".asm", ".txt", ".s", ".lst"}

def normalize_text(s: str) -> str:
    return s.strip().lower()

def group_opcode(opcode: str) -> str:
    return OPCODE_TO_GROUP.get(opcode, "OTHER")

def group_to_behavior(group_name: str) -> str:
    return GROUP_TO_BEHAVIOR.get(group_name, "OTHER_BEHAVIOR")

def extract_symbols_from_line(line: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(line)]

def infer_api_behavior(line: str) -> Optional[str]:
    low = line.lower()

    def contains_any(hints: set[str]) -> bool:
        return any(h in low for h in hints)

    if contains_any(FILE_API_HINTS):
        return "FILE_OPEN_LIKE"
    if contains_any(MEMORY_API_HINTS):
        return "MEMORY_MANAGEMENT"
    if contains_any(PROCESS_API_HINTS):
        return "PROCESS_INTERACTION"
    if contains_any(REGISTRY_API_HINTS):
        return "REGISTRY_INTERACTION"
    if contains_any(NETWORK_API_HINTS):
        return "NETWORK_INTERACTION"
    if contains_any(CRYPTO_API_HINTS):
        return "CRYPTO_API"
    return None

def is_probable_opcode(tok: str) -> bool:
    tok = tok.lower().strip().rstrip(":")
    if not tok:
        return False
    if tok in NON_OPCODE_TOKENS:
        return False
    if HEX_OR_ADDR_RE.match(tok):
        return False
    if not MNEMONIC_RE.match(tok):
        return False
    return True

def extract_opcode_from_line(line: str) -> Optional[str]:
    # remove comments
    line = line.split(";", 1)[0].split("#", 1)[0].strip()
    if not line:
        return None
    if LABEL_RE.match(line):
        return None

    parts = line.split()
    if not parts:
        return None

    # Try first handful of tokens since disassembly often includes addresses/bytes.
    for raw in parts[:6]:
        tok = raw.strip().rstrip(":").lower()
        if not tok:
            continue
        if HEX_OR_ADDR_RE.match(tok):
            continue
        if ":" in tok and tok.count(":") == 1:
            # skip text:00401000-like prefixes
            continue
        if is_probable_opcode(tok):
            return tok
    return None

# ----------------------------
# Sequence builder
# ----------------------------

def parse_asm_file(path: Path) -> Dict:
    raw_opcodes: List[str] = []
    group_tokens: List[str] = []
    behavior_tokens: List[str] = []

    edges_opcode: Counter = Counter()
    edges_group: Counter = Counter()
    edges_behavior: Counter = Counter()

    api_behavior_hits: Counter = Counter()
    opcode_counts: Counter = Counter()
    group_counts: Counter = Counter()
    behavior_counts: Counter = Counter()

    prev_opcode = None
    prev_group = None
    prev_behavior = None

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                api_behavior = infer_api_behavior(line)
                opcode = extract_opcode_from_line(line)

                if opcode is None and api_behavior is None:
                    continue

                if opcode is not None:
                    group_name = group_opcode(opcode)
                    behavior_name = group_to_behavior(group_name)

                    raw_opcodes.append(opcode)
                    group_tokens.append(group_name)
                    behavior_tokens.append(behavior_name)

                    opcode_counts[opcode] += 1
                    group_counts[group_name] += 1
                    behavior_counts[behavior_name] += 1

                    if prev_opcode is not None:
                        edges_opcode[(prev_opcode, opcode)] += 1
                    if prev_group is not None:
                        edges_group[(prev_group, group_name)] += 1
                    if prev_behavior is not None:
                        edges_behavior[(prev_behavior, behavior_name)] += 1

                    prev_opcode = opcode
                    prev_group = group_name
                    prev_behavior = behavior_name

                # If line mentions a likely API/import/function name, append a more specific behavior token.
                if api_behavior is not None:
                    behavior_tokens.append(api_behavior)
                    behavior_counts[api_behavior] += 1
                    api_behavior_hits[api_behavior] += 1

                    if prev_behavior is not None:
                        edges_behavior[(prev_behavior, api_behavior)] += 1
                    prev_behavior = api_behavior

    except Exception as exc:
        return {
            "error": str(exc),
            "raw_opcodes": [],
            "group_tokens": [],
            "behavior_tokens": [],
            "opcode_counts": {},
            "group_counts": {},
            "behavior_counts": {},
            "edges_opcode": [],
            "edges_group": [],
            "edges_behavior": [],
        }

    return {
        "raw_opcodes": raw_opcodes,
        "group_tokens": group_tokens,
        "behavior_tokens": behavior_tokens,
        "opcode_counts": dict(opcode_counts),
        "group_counts": dict(group_counts),
        "behavior_counts": dict(behavior_counts),
        "api_behavior_hits": dict(api_behavior_hits),
        "edges_opcode": [[a, b, w] for (a, b), w in edges_opcode.items()],
        "edges_group": [[a, b, w] for (a, b), w in edges_group.items()],
        "edges_behavior": [[a, b, w] for (a, b), w in edges_behavior.items()],
    }

# ----------------------------
# Dataset builder
# ----------------------------

def build_vocab(samples: List[Dict], key: str, min_freq: int = 1) -> Dict[str, int]:
    freq = Counter()
    for sample in samples:
        for tok in sample.get(key, []):
            freq[tok] += 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for tok, count in freq.most_common():
        if count >= min_freq:
            vocab[tok] = idx
            idx += 1
    return vocab

def encode_sequence(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<PAD>"]] * (max_len - len(ids)))
    return ids

def collect_files(root: Path) -> List[Tuple[Path, int]]:
    files: List[Tuple[Path, int]] = []
    malware_dir = root / "malware"
    goodware_dir = root / "goodware"

    if malware_dir.exists():
        for path in malware_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                files.append((path, 1))

    if goodware_dir.exists():
        for path in goodware_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                files.append((path, 0))

    return files

def graph2vec_style_graph(sample: Dict, mode: str = "group") -> Dict:
    """
    Produce a simple node/edge graph object for later graph embedding work.

    mode:
        - opcode
        - group
        - behavior
    """
    if mode == "opcode":
        seq = sample["raw_opcodes"]
        edge_list = sample["edges_opcode"]
    elif mode == "behavior":
        seq = sample["behavior_tokens"]
        edge_list = sample["edges_behavior"]
    else:
        seq = sample["group_tokens"]
        edge_list = sample["edges_group"]

    nodes = sorted(set(seq))
    node_to_id = {n: i for i, n in enumerate(nodes)}

    edges = []
    for src, dst, weight in edge_list:
        if src in node_to_id and dst in node_to_id:
            edges.append({
                "src": node_to_id[src],
                "dst": node_to_id[dst],
                "weight": weight
            })

    return {
        "nodes": [{"id": node_to_id[n], "label": n} for n in nodes],
        "edges": edges,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Root directory containing malware/ and goodware/")
    parser.add_argument("--output", default="opcode_out", help="Output directory")
    parser.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum token frequency for vocab")
    parser.add_argument("--graph-mode", choices=["opcode", "group", "behavior"], default="group")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(dataset_root)
    if not files:
        raise SystemExit("No files found. Expected ./malware and/or ./goodware with .asm/.txt/.s/.lst files.")

    samples: List[Dict] = []
    label_counts = Counter()

    for path, label in files:
        parsed = parse_asm_file(path)
        rel_path = str(path.relative_to(dataset_root))

        sample = {
            "id": rel_path.replace(os.sep, "__"),
            "path": rel_path,
            "label": label,  # 1=malware, 0=goodware
            "raw_opcodes": parsed["raw_opcodes"],
            "group_tokens": parsed["group_tokens"],
            "behavior_tokens": parsed["behavior_tokens"],
            "opcode_counts": parsed["opcode_counts"],
            "group_counts": parsed["group_counts"],
            "behavior_counts": parsed["behavior_counts"],
            "api_behavior_hits": parsed["api_behavior_hits"],
            "edges_opcode": parsed["edges_opcode"],
            "edges_group": parsed["edges_group"],
            "edges_behavior": parsed["edges_behavior"],
        }

        if sample["group_tokens"] or sample["behavior_tokens"] or sample["raw_opcodes"]:
            samples.append(sample)
            label_counts[label] += 1

    if not samples:
        raise SystemExit("Files were found, but no opcode/token data could be extracted.")

    vocab_raw = build_vocab(samples, "raw_opcodes", min_freq=args.min_freq)
    vocab_group = build_vocab(samples, "group_tokens", min_freq=args.min_freq)
    vocab_behavior = build_vocab(samples, "behavior_tokens", min_freq=args.min_freq)

    for s in samples:
        s["raw_ids"] = encode_sequence(s["raw_opcodes"], vocab_raw, args.max_len)
        s["group_ids"] = encode_sequence(s["group_tokens"], vocab_group, args.max_len)
        s["behavior_ids"] = encode_sequence(s["behavior_tokens"], vocab_behavior, args.max_len)
        s["graph"] = graph2vec_style_graph(s, mode=args.graph_mode)

    with (out_dir / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    vocab_bundle = {
        "raw_vocab": vocab_raw,
        "group_vocab": vocab_group,
        "behavior_vocab": vocab_behavior,
    }
    with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab_bundle, f, indent=2)

    overall_group_counts = Counter()
    overall_behavior_counts = Counter()
    overall_opcode_counts = Counter()

    for s in samples:
        overall_opcode_counts.update(s["opcode_counts"])
        overall_group_counts.update(s["group_counts"])
        overall_behavior_counts.update(s["behavior_counts"])

    stats = {
        "num_samples": len(samples),
        "num_malware": label_counts[1],
        "num_goodware": label_counts[0],
        "top_opcodes": overall_opcode_counts.most_common(30),
        "top_groups": overall_group_counts.most_common(30),
        "top_behaviors": overall_behavior_counts.most_common(30),
        "graph_mode": args.graph_mode,
        "max_len": args.max_len,
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Built {len(samples)} samples")
    print(f"Malware: {label_counts[1]}")
    print(f"Goodware: {label_counts[0]}")
    print("Top grouped tokens:")
    for token, count in overall_group_counts.most_common(10):
        print(f"  {token:<18} {count}")
    print("Top behavior tokens:")
    for token, count in overall_behavior_counts.most_common(10):
        print(f"  {token:<18} {count}")
    print(f"Outputs written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

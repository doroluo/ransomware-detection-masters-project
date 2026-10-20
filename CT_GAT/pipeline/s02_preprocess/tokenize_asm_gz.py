"""Tokenize Mendeley IDA .asm.gz listings into transformer sequences.

Each .pt file is:

    {file_id, sequence: LongTensor[N, 5], label, asm_path,
     instruction_metadata, import_inventory}

sequence rows are [opcode_id, operand1_id, operand2_id, api_id, behavior_id].
Instructions that do not name an API use api_id NO_API (0). A call through
an import slot whose function name cannot be read uses UNK_API (75). Every
other imported or called API is added to the vocabulary automatically.
The saved vocabulary is a JSON list of names; the id is 76 plus the list index.

``instruction_metadata`` is one entry per sequence row. ``address`` and
``rva`` are the RVA of the nearest preceding IDA label, relative to Imagebase,
so they can meet a CFG block's start_rva. ``behavior_tag`` is the existing
behavior-channel name. ``behavior_tags`` are auxiliary labels and do not
replace that channel.

``import_inventory`` holds imported_apis, called_apis, and unused_imports.
``runtime_executed_apis`` stays empty: static disassembly does not prove that 
an API ran. extrn lines are imports, not sequence rows.

Train and test membership comes from each collection's splits/*.csv.
Unassigned samples are left out.

Examples, from CT_GAT/pipeline/s02_preprocess:

    python tokenize_asm_gz.py --limit 2
    python tokenize_asm_gz.py --architecture x86
    python tokenize_asm_gz.py --split train
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch

# Opcode, operand, API, and behavior ids used by the tokenizer and the models.
TOKEN_MAP = {
    "PADDING": 0,
    "UNKNOWN_OPCODE": 1,
    "NOP_SLED": 2,
    "JMP": 10,
    "JZ": 11,
    "JNZ": 12,
    "JE": 13,
    "JNE": 14,
    "JS": 15,
    "JNS": 16,
    "JG": 17,
    "JL": 18,
    "JGE": 19,
    "JLE": 20,
    "CALL": 21,
    "RET": 22,
    "RETN": 23,
    "MOV": 25,
    "PUSH": 26,
    "POP": 27,
    "LEA": 28,
    "XCHG": 29,
    "ADD": 30,
    "SUB": 31,
    "INC": 32,
    "DEC": 33,
    "CMP": 34,
    "TEST": 35,
    "IMUL": 36,
    "MUL": 37,
    "IDIV": 38,
    "DIV": 39,
    "XOR": 40,
    "OR": 41,
    "AND": 42,
    "NOT": 43,
    "SHL": 44,
    "SHR": 45,
    "ROL": 46,
    "ROR": 47,
    "SAR": 48,
    "SAL": 49,
    "NEG": 50,
    "FSTP": 55,
    "FLD": 56,
    "PADD": 57,
    "PXOR": 58,
    "MOVDQA": 59,
    "INT": 60,
    "SYSCALL": 61,
    "SYSENTER": 62,
    "CPUID": 63,
    "RDTSC": 64,
    "REG_DATA": 70,
    "REG_STACK": 71,
    "MEM_STACK_REF": 72,
    "MEM_GLOBAL_REF": 73,
    "IMM_VALUE": 74,
}

NO_API = 0
# 75 stays clear of opcode and operand ids (those stop at 74). The Capstone
# four-id row stores an API id in the opcode channel, so API ids start at 76.
UNK_API = 75
FIRST_API_ID = 76

BEHAVIOR_MAP = {
    "NONE": 0,
    "CRYPTO_OPCODE": 1,
    "DECRYPT": 2,
    "ENCRYPT": 3,
    "KEY_SETUP": 4,
    "FILE_ENUM": 5,
    "FILE_READ": 6,
    "FILE_WRITE": 7,
    "FILE_DELETE": 8,
}
BEHAVIOR_VOCAB_SIZE = len(BEHAVIOR_MAP)
HEAD_BEHAVIOR_NAMES = (
    "CRYPTO_OPCODE",
    "ENCRYPT",
    "DECRYPT",
    "FILE_READ",
    "FILE_WRITE",
)
HEAD_BEHAVIOR_IDS = tuple(BEHAVIOR_MAP[name] for name in HEAD_BEHAVIOR_NAMES)
SUSPICIOUS_BEHAVIOR_IDS = frozenset(BEHAVIOR_MAP[name] for name in BEHAVIOR_MAP if name != "NONE")
CRYPTO_BEHAVIOR_IDS = frozenset({
    BEHAVIOR_MAP["CRYPTO_OPCODE"],
    BEHAVIOR_MAP["DECRYPT"],
    BEHAVIOR_MAP["ENCRYPT"],
    BEHAVIOR_MAP["KEY_SETUP"],
})
FILE_BEHAVIOR_IDS = frozenset({
    BEHAVIOR_MAP["FILE_ENUM"],
    BEHAVIOR_MAP["FILE_READ"],
    BEHAVIOR_MAP["FILE_WRITE"],
    BEHAVIOR_MAP["FILE_DELETE"],
})

CRYPTO_OPCODES = frozenset({
    "XOR", "PXOR", "ROL", "ROR", "SHL", "SHR", "SAL", "SAR",
    "MOVDQA", "MOVDQU", "AESENC", "AESENCLAST", "AESDEC", "AESDECLAST",
    "AESKEYGENASSIST", "AESIMC",
})

API_BEHAVIOR = {
    "cryptencrypt": BEHAVIOR_MAP["ENCRYPT"],
    "cryptdecrypt": BEHAVIOR_MAP["DECRYPT"],
    "cryptacquirecontexta": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptacquirecontextw": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptcreatehash": BEHAVIOR_MAP["KEY_SETUP"],
    "crypthashdata": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptderivekey": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptdestroykey": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptdestroyhash": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptreleasecontext": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptgenrandom": BEHAVIOR_MAP["KEY_SETUP"],
    "cryptstringtobinarya": BEHAVIOR_MAP["KEY_SETUP"],
    "bcreptopenalgorithmprovider": BEHAVIOR_MAP["KEY_SETUP"],
    "bcryptexecute": BEHAVIOR_MAP["KEY_SETUP"],
    "rtlcomputecrc32": BEHAVIOR_MAP["KEY_SETUP"],
    "findfirstfilea": BEHAVIOR_MAP["FILE_ENUM"],
    "findfirstfilew": BEHAVIOR_MAP["FILE_ENUM"],
    "findnextfilea": BEHAVIOR_MAP["FILE_ENUM"],
    "findnextfilew": BEHAVIOR_MAP["FILE_ENUM"],
    "getlogicaldrives": BEHAVIOR_MAP["FILE_ENUM"],
    "getdriveclass": BEHAVIOR_MAP["FILE_ENUM"],
    "getdiskfreespaceexa": BEHAVIOR_MAP["FILE_ENUM"],
    "readfile": BEHAVIOR_MAP["FILE_READ"],
    "getfilesize": BEHAVIOR_MAP["FILE_READ"],
    "setfilepointer": BEHAVIOR_MAP["FILE_READ"],
    "createfilea": BEHAVIOR_MAP["FILE_READ"],
    "createfilew": BEHAVIOR_MAP["FILE_READ"],
    "writefile": BEHAVIOR_MAP["FILE_WRITE"],
    "copyfilea": BEHAVIOR_MAP["FILE_WRITE"],
    "copyfilew": BEHAVIOR_MAP["FILE_WRITE"],
    "deletefilea": BEHAVIOR_MAP["FILE_DELETE"],
    "deletefilew": BEHAVIOR_MAP["FILE_DELETE"],
    "movefileexa": BEHAVIOR_MAP["FILE_WRITE"],
    "movefileexw": BEHAVIOR_MAP["FILE_WRITE"],
}

API_ALIASES = {
    "readfileex": "readfile",
    "ntreadfile": "readfile",
    "zwreadfile": "readfile",
    "fread": "readfile",
    "read": "readfile",
    "writefileex": "writefile",
    "ntwritefile": "writefile",
    "zwwritefile": "writefile",
    "fwrite": "writefile",
    "write": "writefile",
    "fopen": "createfilea",
    "open": "createfilea",
    "rename": "movefileexw",
    "movefile": "movefileexw",
    "movefilea": "movefileexw",
    "movefilew": "movefileexw",
    "replacefile": "movefileexw",
    "replacefilea": "movefileexw",
    "replacefilew": "movefileexw",
}

OPCODE_PREFIXES = {"LOCK", "REP", "REPE", "REPZ", "REPNE", "REPNZ"}


class ApiVocab:
    """API ids discovered from imports and calls.

    The saved file is a JSON list of names. The first name is id 76, the next
    is 77, and so on. Spelling differences such as CreateFileW and
    createfilew share one id. MoveFileW does not share an id with MoveFileExW.
    """

    def __init__(self) -> None:
        self.names: list[str] = []
        self._ids: dict[str, int] = {}

    def id_for(self, name: str) -> int:
        key = name.lower()
        api_id = self._ids.get(key)
        if api_id is None:
            api_id = FIRST_API_ID + len(self.names)
            self.names.append(name)
            self._ids[key] = api_id
        return api_id

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._ids

_API_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_@]*")
_REG_DATA = re.compile(
    r"\b(rax|rbx|rcx|rdx|rsi|rdi|r8|r9|r10|r11|r12|r13|r14|r15|"
    r"eax|ebx|ecx|edx|esi|edi|ax|bx|cx|dx|al|bl|cl|dl)\b",
    re.IGNORECASE,
)
_REG_STACK = re.compile(r"\b(rbp|rsp|ebp|esp|bp|sp)\b", re.IGNORECASE)
_MEM_STACK = re.compile(r"\[\s*(rbp|rsp|ebp|esp|bp|sp)\s*[\+\-].*\]", re.IGNORECASE)
_MEM = re.compile(r"\[.*\]")
_IMM = re.compile(r"\b(0x[0-9a-fA-F]+|[0-9]+h?)\b")


def _mnemonic(opcode: str) -> str:
    """Drop lock/rep so 'rep stosd' is categorized as STOSD."""
    parts = opcode.upper().split()
    while parts and parts[0] in OPCODE_PREFIXES:
        parts.pop(0)
    return parts[-1] if parts else opcode.upper()


def _operand_id(operand: str) -> int:
    if not operand:
        return TOKEN_MAP["PADDING"]
    operand = operand.strip()
    if _MEM.search(operand):
        if _MEM_STACK.search(operand):
            return TOKEN_MAP["MEM_STACK_REF"]
        return TOKEN_MAP["MEM_GLOBAL_REF"]
    if _REG_DATA.search(operand):
        return TOKEN_MAP["REG_DATA"]
    if _REG_STACK.search(operand):
        return TOKEN_MAP["REG_STACK"]
    if _IMM.search(operand):
        return TOKEN_MAP["IMM_VALUE"]
    return TOKEN_MAP["MEM_GLOBAL_REF"]


def _matched_api(operands: list[str]) -> str | None:
    """Exact Capstone token match. Substring matches are not APIs.

    ``connect`` must not match inside ``WNetGetConnectionW``. Registers are
    not APIs. ``_accept_symbol`` is defined with the IDA parser below.
    """
    if not operands:
        return None
    tokens = list(dict.fromkeys(_API_IDENT.findall(" ".join(operands))))
    for token in sorted(tokens, key=len, reverse=True):
        if _REG_DATA.fullmatch(token) or _REG_STACK.fullmatch(token):
            continue
        if _accept_symbol(token):
            return token
    return None


def _behavior_key(api_name: str) -> str:
    key = api_name.lower()
    return API_ALIASES.get(key, key)


def _behavior_id(opcode: str, api_name: str | None) -> int:
    if api_name is not None:
        return API_BEHAVIOR.get(_behavior_key(api_name), BEHAVIOR_MAP["NONE"])
    if opcode in CRYPTO_OPCODES:
        return BEHAVIOR_MAP["CRYPTO_OPCODE"]
    return BEHAVIOR_MAP["NONE"]


def encode_instruction_channels(
    opcode: str,
    raw_operands: list[str],
    api_token: str | None = None,
    *,
    unresolved: bool = False,
    explicit: bool = False,
    api_ids: ApiVocab | None = None,
) -> list[int]:
    """Return [opcode_id, operand1_id, operand2_id, api_id, behavior_id].

    The Capstone path leaves ``explicit`` false and matches whole tokens.
    The IDA path passes the resolved function name, or ``unresolved`` when a
    call uses an import slot with no readable name.
    """
    opcode_name = _mnemonic(opcode)
    if explicit:
        if unresolved:
            api_name = None
            api_id = UNK_API
        elif api_token:
            table = api_ids or ApiVocab()
            api_name = api_token
            api_id = table.id_for(api_token)
        else:
            api_name = None
            api_id = NO_API
    else:
        api_name = _matched_api(raw_operands)
        api_id = (api_ids or ApiVocab()).id_for(api_name) if api_name is not None else NO_API
    if opcode_name == "NOP":
        opcode_id = TOKEN_MAP["NOP_SLED"]
    else:
        opcode_id = TOKEN_MAP.get(opcode_name, TOKEN_MAP["UNKNOWN_OPCODE"])
    operand_1 = _operand_id(raw_operands[0]) if len(raw_operands) >= 1 else TOKEN_MAP["PADDING"]
    operand_2 = _operand_id(raw_operands[1]) if len(raw_operands) >= 2 else TOKEN_MAP["PADDING"]
    return [opcode_id, operand_1, operand_2, api_id, _behavior_id(opcode_name, api_name)]


def encode_instruction(
    opcode: str,
    raw_operands: list[str],
    api_ids: ApiVocab | None = None,
) -> list[int]:
    """Four ids for the older Capstone listings.

    [opcode_or_api_id, operand_1_id, operand_2_id, behavior_id]
    A known API call stores the API id in the opcode channel.
    """
    mnemonic = _mnemonic(opcode)
    opcode_id, operand_1, operand_2, api_id, behavior_id = encode_instruction_channels(
        opcode, raw_operands, api_ids=api_ids
    )
    if mnemonic == "CALL" and api_id not in (NO_API, UNK_API):
        opcode_id = api_id
    return [opcode_id, operand_1, operand_2, behavior_id]


def parse_capstone_insn(
    mnemonic: str,
    op_str: str = "",
    api_ids: ApiVocab | None = None,
) -> list[int]:
    """Encode one Capstone instruction (mnemonic + op_str) into four token ids."""
    raw_operands = [operand.strip() for operand in op_str.split(",") if operand.strip()]
    return encode_instruction(mnemonic, raw_operands, api_ids)


def parse_asm_line(line: str, api_ids: ApiVocab | None = None) -> list[int] | None:
    """Parse one Capstone line ``00401000: 55 push ebp`` into four token ids."""
    line = line.strip()
    if not line or line.startswith(";"):
        return None
    if ";" in line:
        line = line.split(";", 1)[0].strip()
    tokens = line.split()
    if len(tokens) < 2 or ":" not in tokens[0]:
        return None
    if any(keyword in line.lower() for keyword in (
        "public", "assume", "proc", "endp", "segment", "ends", "unicode", "extrn",
    )):
        return None
    if any(directive in tokens for directive in ("db", "dw", "dd", "align")):
        return None

    code_tokens = [
        token
        for token in tokens[1:]
        if not re.fullmatch(r"[0-9A-Fa-f]{2}", token)
        and not re.fullmatch(r"[0-9A-Fa-f]{4}", token)
    ]
    if not code_tokens:
        return None
    prefix_index = 0
    while prefix_index < len(code_tokens) and code_tokens[prefix_index].upper() in OPCODE_PREFIXES:
        prefix_index += 1
    if prefix_index >= len(code_tokens):
        return None
    opcode = code_tokens[prefix_index]
    operand_text = " ".join(code_tokens[prefix_index + 1 :])
    raw_operands = [operand.strip() for operand in operand_text.split(",") if operand.strip()]
    return encode_instruction(opcode, raw_operands, api_ids)


def parse_asm_file(asm_path, api_ids: ApiVocab | None = None) -> list[list[int]]:
    """Parse a plain Capstone .asm file into four-id rows."""
    table = api_ids or ApiVocab()
    parsed: list[list[int]] = []
    with Path(asm_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parsed_line = parse_asm_line(line, table)
            if parsed_line is not None:
                parsed.append(parsed_line)
    return parsed

SCRIPT_DIR = Path(__file__).resolve().parent
# This file lives in CT_GAT/pipeline/s02_preprocess. data/ and processed/ stay at CT_GAT.
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ROOTS = [
    DATA_DIR / "Mendeley_Ransomware",
    DATA_DIR / "Mendeley_Goodware",
]
OUTPUT_DIR = PROJECT_ROOT / "processed" / "mendeley_sequences"
SPLIT_DIR = PROJECT_ROOT / "processed" / "mendeley_splits"
API_VOCAB_PATH = PROJECT_ROOT / "processed" / "api_vocab.json"

# IDA stack-frame definitions look like: var_4 = dword ptr -4
# Data and listing directives are not instructions.
_SKIP_TOKENS = {
    "proc",
    "endp",
    "segment",
    "ends",
    "assume",
    "public",
    "extrn",
    "include",
    "org",
    "struc",
    "endstruc",
    "struct",
    "db",
    "dw",
    "dd",
    "dq",
    "dt",
    "align",
    "unicode",
}
_JUMP_HINTS = {"short", "near", "far"}
_ADDR_PREFIX = re.compile(
    r"^(?P<segment>[\w.$]+):(?P<address>[0-9A-Fa-f]+?)(?P<suffix>h)?$",
    re.IGNORECASE,
)
_LABEL_ADDR = re.compile(r"(?:sub|loc|locret|unk)_([0-9A-Fa-f]+)", re.IGNORECASE)
_IMAGEBASE = re.compile(r"\bImagebase\s*:\s*([0-9A-Fa-f]+)\b", re.IGNORECASE)
_ORG = re.compile(r"\borg\s+([0-9A-Fa-f]+?)h\b", re.IGNORECASE)
_IDA_IMM = re.compile(r"\b[0-9][0-9A-Fa-f]*h\b", re.IGNORECASE)
_IMP_PREFIX = re.compile(r"__imp_+|j_(?=[A-Za-z])")
_SEGMENT_PREFIX = re.compile(r"\b(?:cs|ds|es|fs|gs|ss):", re.IGNORECASE)
_BEHAVIOR_NAME = {value: name for name, value in BEHAVIOR_MAP.items()}
_IMPORT_HEADER = re.compile(
    r";\s*(?P<kind>Delay\s+imports|Imports)\s+from\s+(?P<dll>\S+)",
    re.IGNORECASE,
)
_EXTRN = re.compile(
    r"\bextrn\s+(?:__imp_+)?(?P<symbol>[A-Za-z_?][A-Za-z0-9_@?]*)\s*:",
    re.IGNORECASE,
)
_CALL_NAME = re.compile(
    r"(?:__imp_+|(?:cs|ds|es|fs|gs|ss):(?:__imp_+)?)([A-Za-z_?][A-Za-z0-9_@?]*)",
    re.IGNORECASE,
)
_J_NAME = re.compile(r"\bj_([A-Za-z_][A-Za-z0-9_@]*)")
_SLOT_REF = re.compile(r"\b(?:cs|ds|es|fs|gs|ss):|__imp_+", re.IGNORECASE)
_COMMENT_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TAG_SUFFIX = re.compile(r"^(?:exw|exa|ex|w|a)?$")
_NON_API_NAMES = {
    "dword", "qword", "word", "byte", "offset", "ptr", "near", "short", "far",
    "flat", "large", "xmmword", "tbyte",
}
_KNOWN_DLLS = {
    "kernel32", "kernelbase", "ntdll", "user32", "advapi32", "shell32", "ole32",
    "oleaut32", "ws2_32", "wininet", "mpr", "gdi32", "crypt32", "bcrypt",
    "shlwapi", "iphlpapi", "urlmon", "winhttp", "combase",
}
_AUX_BEHAVIOR_RULES = (
    ("FILE_ENUMERATION", ("findfirstfile", "findnextfile", "getlogicaldrives", "getdrivetype", "getdiskfreespace")),
    ("FILE_OPEN", ("createfile", "ntcreatefile", "zwcreatefile", "ntopenfile")),
    ("FILE_READ", ("readfile", "ntreadfile", "zwreadfile")),
    ("FILE_WRITE", ("writefile", "ntwritefile", "zwwritefile")),
    ("FILE_RENAME_DELETE", ("movefile", "deletefile", "replacefile")),
    ("ENCRYPTION_API", ("cryptencrypt", "cryptdecrypt", "bcryptencrypt", "bcryptdecrypt")),
    ("KEY_GENERATION", (
        "cryptgenrandom", "cryptderivekey", "cryptsetkeyparam", "cryptdestroykey",
        "cryptacquirecontext", "cryptcreatehash", "bcryptgenerate",
    )),
    ("CRYPTO_PRIMITIVE", ("crypthashdata", "cryptdestroyhash")),
    ("SERVICE_CONTROL", ("controlservice", "openservice", "startservice", "createservice", "closeservicehandle")),
    ("PROCESS_EXECUTION", ("createprocess", "shellexecute", "winexec", "ntcreateuserprocess")),
    ("NETWORK", (
        "wsastartup", "socket", "connect", "send", "recv", "internetopen",
        "internetconnect", "httpsendrequest", "wsasend", "wsarecv", "getaddrinfo",
    )),
    ("SYNCHRONIZATION", (
        "initializeslisthead", "entercriticalsection", "leavecriticalsection",
        "waitforsingleobject", "createmutex", "openmutex",
    )),
)


def _ida_operand_text(text: str) -> str:
    """Normalize IDA syntax so the existing operand and API rules still match.

    ``ds:__imp_CreateFileW`` and ``j_CreateFileW`` become ``CreateFileW``.
    ``40h`` and ``0Ch`` become ``0x40`` and ``0x0C``, which ``_operand_id``
    already treats as one immediate id.
    """
    text = _SEGMENT_PREFIX.sub("", text)
    text = _IMP_PREFIX.sub("", text)

    def hex_immediate(match: re.Match[str]) -> str:
        return "0x" + match.group(0)[:-1]

    text = _IDA_IMM.sub(hex_immediate, text)
    return re.sub(r"\s+", " ", text).strip()


def _va_from_token(token: str) -> int | None:
    """Read a virtual address from an IDA label or a ``seg:address`` prefix."""
    prefixed = _ADDR_PREFIX.fullmatch(token)
    if prefixed:
        return int(prefixed.group("address"), 16)
    label = _LABEL_ADDR.fullmatch(token.rstrip(":"))
    if label:
        return int(label.group(1), 16)
    return None


def _note_location(line: str, state: dict[str, int | None]) -> None:
    """Remember Imagebase and the virtual address of the latest IDA label."""
    imagebase = _IMAGEBASE.search(line)
    if imagebase:
        state["imagebase"] = int(imagebase.group(1), 16)
    org = _ORG.search(line)
    if org:
        state["va"] = int(org.group(1), 16)
    code = line.split(";", 1)[0].strip()
    if not code or code.startswith("."):
        return
    tokens = code.split()
    if len(tokens) == 2 and tokens[1].lower() in ("endp", "ends"):
        # ``sub_401E80 endp`` closes a function. Its name is the function start, not a new label address.
        return
    va = _va_from_token(tokens[0])
    if va is not None:
        state["va"] = va


def _current_rva(state: dict[str, int | None]) -> int | None:
    """Convert the latest label VA to the RVA stored on CFG blocks."""
    va = state.get("va")
    if va is None:
        return None
    imagebase = state.get("imagebase")
    if imagebase is not None and va >= imagebase:
        return va - imagebase
    return va


def parse_ida_instruction_line(line: str) -> dict[str, Any] | None:
    """Parse one IDA line into an opcode and operands, or None to skip it.

    ``extrn`` declarations are not instructions. The comment is kept so a
    call such as ``call eax ; CreateFileW`` can still name the API.
    """
    raw = line.strip()
    if not raw or raw.startswith(";") or raw.startswith("."):
        return None
    comment = ""
    if ";" in raw:
        raw, comment = raw.split(";", 1)
        raw = raw.strip()
        comment = comment.strip()
    if not raw or "=" in raw.split():
        return None

    tokens = raw.split()
    if tokens and _ADDR_PREFIX.fullmatch(tokens[0]):
        tokens = tokens[1:]
    if tokens and tokens[0].endswith(":"):
        tokens = tokens[1:]
    if not tokens:
        return None

    # ``name proc`` and ``label db ...`` keep the directive off the first token.
    lowered = [token.lower().rstrip(":") for token in tokens]
    if any(token in _SKIP_TOKENS or token == "equ" for token in lowered):
        return None

    prefix_end = 0
    while prefix_end < len(tokens) and tokens[prefix_end].upper() in OPCODE_PREFIXES:
        prefix_end += 1
    if prefix_end >= len(tokens) or not tokens[prefix_end][:1].isalpha():
        return None

    opcode = " ".join(tokens[: prefix_end + 1])
    operand_tokens = tokens[prefix_end + 1 :]
    while operand_tokens and operand_tokens[0].lower() in _JUMP_HINTS:
        operand_tokens.pop(0)
        if operand_tokens and operand_tokens[0].lower() == "ptr":
            operand_tokens.pop(0)

    source_operands = " ".join(operand_tokens)
    operand_text = _ida_operand_text(source_operands)
    raw_operands = [operand.strip() for operand in operand_text.split(",") if operand.strip()]
    return {
        "opcode": opcode,
        "operands": raw_operands,
        "operand_text": source_operands,
        "comment": comment,
    }


def _decode_pe_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
    else:
        text = str(value)
    text = text.strip().rstrip("\x00")
    return text or None


def imports_from_pe(path: Path) -> list[dict[str, Any]]:
    """Read DLL, API name, ordinal, IAT RVA, and delay-import flag with pefile.

    ``thunk_rva`` is the import-address-table slot. pefile's ``address`` is
    that slot as a virtual address (RVA plus ImageBase), so it is only a
    fallback. Ordinal-only imports keep ``name`` as None.
    """
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ]
        )
        imagebase = int(pe.OPTIONAL_HEADER.ImageBase)
        rows: list[dict[str, Any]] = []
        for delay, attr in (
            (False, "DIRECTORY_ENTRY_IMPORT"),
            (True, "DIRECTORY_ENTRY_DELAY_IMPORT"),
        ):
            for entry in getattr(pe, attr, None) or []:
                dll = _decode_pe_text(getattr(entry, "dll", None))
                for imp in getattr(entry, "imports", ()) or ():
                    name = _decode_pe_text(getattr(imp, "name", None))
                    if name:
                        name = re.sub(r"^__imp_+", "", name)
                    ordinal = getattr(imp, "ordinal", None)
                    iat_rva = getattr(imp, "thunk_rva", None)
                    if iat_rva is None:
                        address = getattr(imp, "address", None)
                        if address is not None and int(address) >= imagebase:
                            iat_rva = int(address) - imagebase
                    if name is None and not ordinal:
                        continue
                    rows.append(
                        {
                            "name": name,
                            "dll": dll,
                            "ordinal": int(ordinal) if ordinal else None,
                            "iat_rva": int(iat_rva) if iat_rva is not None else None,
                            "delay": delay,
                        }
                    )
        return rows
    finally:
        pe.close()


def collect_ida_imports(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Read ``extrn`` names and the ``Imports from DLL`` header above them.

    These rows are file-level imports. They are not execution evidence.
    """
    rows: list[dict[str, Any]] = []
    dll: str | None = None
    delay = False
    for line in lines:
        header = _IMPORT_HEADER.search(line)
        if header:
            delay = header.group("kind").lower().startswith("delay")
            dll = header.group("dll").rstrip(".,")
            continue
        code = line.split(";", 1)[0]
        if re.search(r"\b(?:proc|segment)\b", code, re.IGNORECASE):
            dll = None
            delay = False
        match = _EXTRN.search(code)
        if match is None:
            continue
        rows.append(
            {
                "name": re.sub(r"^__imp_+", "", match.group("symbol")),
                "dll": dll,
                "ordinal": None,
                "iat_rva": None,
                "delay": delay,
            }
        )
    return rows


def _merge_imports(
    pe_rows: list[dict[str, Any]],
    ida_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """PE fields win. IDA fills a DLL or name the PE record left empty."""
    merged: dict[str, dict[str, Any]] = {}
    for row in [*pe_rows, *ida_rows]:
        name = row.get("name")
        if name:
            name = re.sub(r"^__imp_+", "", str(name))
            key = name.lower()
        else:
            key = f"ordinal:{row.get('dll')}:{row.get('ordinal')}"
        current = merged.get(key)
        if current is None:
            merged[key] = {
                "name": name,
                "dll": row.get("dll"),
                "ordinal": row.get("ordinal"),
                "iat_rva": row.get("iat_rva"),
                "delay": bool(row.get("delay")),
            }
            continue
        for field in ("dll", "ordinal", "iat_rva"):
            if current.get(field) is None and row.get(field) is not None:
                current[field] = row[field]
        if row.get("delay"):
            current["delay"] = True
    return merged


def _function_name(token: str) -> str:
    """Drop IDA import and thunk prefixes from one symbol."""
    name = re.sub(r"^__imp_+", "", token)
    if name.lower().startswith("j_") and not name.lower().startswith("jpt_"):
        name = name[2:]
    return name


def _accept_symbol(name: str) -> bool:
    low = name.lower()
    if not name or name.startswith("?"):
        return False
    if low in _NON_API_NAMES:
        return False
    # IDA's own labels and compiler helpers are not imported API names.
    # ``__imp_CreateFileW`` is stripped before this check.
    if low.startswith("__"):
        return False
    if low.startswith((
        "sub_", "loc_", "locret_", "unk_", "off_", "jpt_", "algn_",
        "dword_", "word_", "byte_", "qword_", "xmmword_", "tbyte_",
        "flt_", "dbl_", "stru_", "asc_", "def_",
    )):
        return False
    return True


def _mixed_case(name: str) -> bool:
    return any(char.islower() for char in name) and any(char.isupper() for char in name)


def _name_from_operand(text: str, imports: dict[str, dict[str, Any]]) -> tuple[str | None, bool]:
    """Return a called function name, and whether a non-API symbol was seen."""
    rejected = False
    for match in _CALL_NAME.finditer(text):
        name = _function_name(match.group(1))
        if _accept_symbol(name):
            return name, False
        rejected = True
    for match in _J_NAME.finditer(text):
        name = match.group(1)
        known = (
            name.lower() in imports
            or name.lower() in API_BEHAVIOR
            or name.lower() in API_ALIASES
        )
        if _accept_symbol(name) and (known or _mixed_case(name)):
            return name, False
        rejected = True
    return None, rejected


def _is_dll_token(token: str) -> bool:
    low = token.lower().rstrip(".")
    return (
        low.endswith(".dll")
        or low.startswith(("api-ms-", "ext-ms-"))
        or low in _KNOWN_DLLS
    )


def _comment_dll(comment: str) -> str | None:
    for token in re.findall(r"[A-Za-z0-9_.\-]+", comment):
        if _is_dll_token(token):
            return token.rstrip(".")
    return None


def _comment_api(comment: str, imports: dict[str, dict[str, Any]]) -> str | None:
    """Use a comment only when it names an imported or already mapped API."""
    for token in _COMMENT_IDENT.findall(comment):
        if _is_dll_token(token):
            continue
        key = token.lower()
        if key in imports or key in API_BEHAVIOR or key in API_ALIASES:
            return token
    return None


def resolve_api_reference(
    operand_text: str,
    comment: str,
    imports: dict[str, dict[str, Any]],
    mnemonic: str,
) -> dict[str, Any]:
    """Tell a real API reference apart from a plain instruction.

    ``called_api`` comes from ``ds:``, ``cs:``, ``__imp_``, or ``j_`` on a
    call or jump. ``ida_comment`` is used only when the operand itself has
    no function name. An import slot with no readable name is unresolved.
    """
    name, rejected = _name_from_operand(operand_text, imports)
    source: str | None = None
    unresolved = False
    if name is not None:
        source = "called_api" if mnemonic in {"CALL", "JMP"} else "operand"
    else:
        comment_name = _comment_api(comment, imports) if mnemonic in {"CALL", "JMP"} else None
        if comment_name is not None:
            name = comment_name
            source = "ida_comment"
        elif (
            mnemonic in {"CALL", "JMP"}
            and _SLOT_REF.search(operand_text)
            and not rejected
        ):
            unresolved = True
            source = "unresolved"
    dll = None
    if name is not None:
        info = imports.get(name.lower())
        dll = info.get("dll") if info is not None else None
        if dll is None:
            dll = _comment_dll(comment)
    return {"name": name, "dll": dll, "source": source, "unresolved": unresolved}


def _stem_match(name: str, stem: str) -> bool:
    if not name.startswith(stem):
        return False
    return _TAG_SUFFIX.fullmatch(name[len(stem):]) is not None


def auxiliary_behavior_tags(api_name: str | None, opcode: str) -> list[str]:
    """Auxiliary tags. They do not change the numeric behavior channel."""
    tags: list[str] = []
    if _mnemonic(opcode) in CRYPTO_OPCODES:
        tags.append("CRYPTO_PRIMITIVE")
    if not api_name:
        return tags
    folded = api_name.lower()
    matched = False
    for tag, stems in _AUX_BEHAVIOR_RULES:
        if any(_stem_match(folded, stem) for stem in stems):
            if tag not in tags:
                tags.append(tag)
            matched = True
    if not matched:
        tags.append("OTHER")
    return tags


def _metadata_row(
    record: dict[str, Any],
    parsed: list[int],
    va: int | None,
    rva: int | None,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    """Keep the old behavior channel and store the resolved API beside it."""
    api_id = parsed[3]
    behavior_id = parsed[4]
    api_name = None if api_id in (NO_API, UNK_API) else resolved.get("name")
    return {
        "address": rva,
        "va": va,
        "rva": rva,
        "opcode": record["opcode"],
        "operands": record["operands"],
        "api_name": api_name,
        "dll_name": resolved.get("dll") if api_name else None,
        "api_source": resolved.get("source") if api_id != NO_API else None,
        "behavior_tag": _BEHAVIOR_NAME[behavior_id],
        "behavior_tags": auxiliary_behavior_tags(api_name, record["opcode"]),
    }


def _import_source(pe_rows: list[dict[str, Any]], ida_rows: list[dict[str, Any]]) -> str:
    if pe_rows and ida_rows:
        return "pefile+ida_extrn"
    if pe_rows:
        return "pefile"
    if ida_rows:
        return "ida_extrn"
    return "none"


def _inventory(
    imports: dict[str, dict[str, Any]],
    called: dict[str, dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    imported = [
        {
            "name": info.get("name"),
            "dll": info.get("dll"),
            "ordinal": info.get("ordinal"),
            "iat_rva": info.get("iat_rva"),
            "delay": bool(info.get("delay")),
        }
        for info in imports.values()
    ]
    imported.sort(key=lambda row: (row["name"] or "").lower())
    called_rows = [
        {
            "name": item["name"],
            "dll": item.get("dll"),
            "api_id": item["api_id"],
            "count": item["count"],
        }
        for item in sorted(called.values(), key=lambda item: item["name"].lower())
    ]
    called_names = {item["name"].lower() for item in called.values()}
    unused = [
        row for row in imported
        if not row["name"] or row["name"].lower() not in called_names
    ]
    return {
        "imported_apis": imported,
        "called_apis": called_rows,
        "unused_imports": unused,
        "runtime_executed_apis": [],
        "import_source": source,
    }


def encode_parsed_record(
    record: dict[str, Any],
    imports: dict[str, dict[str, Any]],
    api_ids: ApiVocab,
) -> tuple[list[int], dict[str, Any]]:
    """Encode one parsed IDA instruction with the shared API vocabulary.

    ``imports`` is the file-level table from ``_merge_imports``. Names already
    in ``api_ids`` keep their ids. A new called or imported name is appended
    to that same vocabulary.
    """
    resolved = resolve_api_reference(
        record["operand_text"],
        record["comment"],
        imports,
        _mnemonic(record["opcode"]),
    )
    if resolved["unresolved"]:
        parsed = encode_instruction_channels(
            record["opcode"],
            record["operands"],
            unresolved=True,
            explicit=True,
            api_ids=api_ids,
        )
    elif resolved["name"]:
        parsed = encode_instruction_channels(
            record["opcode"],
            record["operands"],
            resolved["name"],
            explicit=True,
            api_ids=api_ids,
        )
    else:
        parsed = encode_instruction_channels(
            record["opcode"],
            record["operands"],
            explicit=True,
            api_ids=api_ids,
        )
    return parsed, resolved


def instruction_api_kind(api_id: int, api_source: str | None) -> str:
    """Separate a missing API, an unresolved slot, an unseen name, and a known API.

    Imported APIs that are never called are not instruction rows. They stay in
    the sample import inventory.
    """
    if api_source == "unseen":
        return "unseen"
    if api_id == UNK_API or api_source == "unresolved":
        return "unresolved"
    if api_id == NO_API or not api_source:
        return "none"
    return "known"


def _remember_call(
    called: dict[str, dict[str, Any]],
    resolved: dict[str, Any],
    api_id: int,
    mnemonic: str,
) -> None:
    if not resolved.get("name") or resolved.get("unresolved"):
        return
    if mnemonic not in {"CALL", "JMP"} or resolved.get("source") not in {"called_api", "ida_comment"}:
        return
    key = resolved["name"].lower()
    slot = called.get(key)
    if slot is None:
        called[key] = {
            "name": resolved["name"],
            "dll": resolved["dll"],
            "api_id": api_id,
            "count": 1,
        }
        return
    slot["count"] += 1
    if slot.get("dll") is None:
        slot["dll"] = resolved["dll"]


def tokenize_lines(
    lines: Iterable[str],
    ida_imports: list[dict[str, Any]],
    pe_imports: list[dict[str, Any]] | None = None,
    api_ids: ApiVocab | None = None,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    """Tokenize IDA lines whose import table was already collected."""
    pe_rows = pe_imports or []
    table = api_ids or ApiVocab()
    imports = _merge_imports(pe_rows, ida_imports)
    for info in imports.values():
        if info.get("name"):
            table.id_for(info["name"])
    instructions: list[list[int]] = []
    metadata: list[dict[str, Any]] = []
    called: dict[str, dict[str, Any]] = {}
    state: dict[str, int | None] = {"imagebase": None, "va": None}
    for line in lines:
        _note_location(line, state)
        record = parse_ida_instruction_line(line)
        if record is None:
            continue
        parsed, resolved = encode_parsed_record(record, imports, table)
        _remember_call(called, resolved, parsed[3], _mnemonic(record["opcode"]))
        instructions.append(parsed)
        metadata.append(
            _metadata_row(record, parsed, state.get("va"), _current_rva(state), resolved)
        )
    inventory = _inventory(imports, called, _import_source(pe_rows, ida_imports))
    return instructions, metadata, inventory


def tokenize_listing(
    lines: Iterable[str],
    api_ids: ApiVocab | None = None,
    pe_imports: list[dict[str, Any]] | None = None,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    """Tokenize an in-memory IDA listing. Used by the tests."""
    listing = list(lines)
    return tokenize_lines(listing, collect_ida_imports(listing), pe_imports, api_ids)


def tokenize_raw_instruction(
    raw_line: str,
    imports: dict[str, dict[str, Any]],
    api_ids: ApiVocab | None = None,
    *,
    va: int | None = None,
    rva: int | None = None,
    freeze_vocab: bool = False,
) -> dict[str, Any] | None:
    """Tokenize one selected assembly line with a shared API vocabulary.

    When ``freeze_vocab`` is set, a name that is not already in ``api_ids``
    stays out of the vocabulary and uses ``UNK_API``. The API name is still
    kept on the row, with ``api_kind`` ``unseen``. An import slot with no
    readable name stays ``unresolved``. ``NO_API`` means the instruction does
    not name an API.
    """
    record = parse_ida_instruction_line(raw_line)
    if record is None:
        return None
    table = api_ids or ApiVocab()
    resolved = resolve_api_reference(
        record["operand_text"],
        record["comment"],
        imports,
        _mnemonic(record["opcode"]),
    )
    unseen_name = None
    if (
        freeze_vocab
        and resolved.get("name")
        and not resolved.get("unresolved")
        and resolved["name"] not in table
    ):
        unseen_name = resolved["name"]
        parsed = encode_instruction_channels(
            record["opcode"],
            record["operands"],
            unresolved=True,
            explicit=True,
            api_ids=table,
        )
        # The id stays UNK_API so the shared vocabulary does not grow.
        # The behavior channel still follows the named API.
        parsed[4] = _behavior_id(_mnemonic(record["opcode"]), unseen_name)
        resolved = {
            "name": unseen_name,
            "dll": resolved.get("dll"),
            "source": "unseen",
            "unresolved": True,
        }
    else:
        parsed, resolved = encode_parsed_record(record, imports, table)
    metadata = _metadata_row(record, parsed, va, rva, resolved)
    if unseen_name is not None:
        metadata["api_name"] = unseen_name
        metadata["api_source"] = "unseen"
        metadata["dll_name"] = resolved.get("dll")
        metadata["behavior_tag"] = _BEHAVIOR_NAME[parsed[4]]
        metadata["behavior_tags"] = auxiliary_behavior_tags(unseen_name, record["opcode"])
    metadata["raw_text"] = raw_line.strip()
    metadata["token_ids"] = parsed
    metadata["api_kind"] = instruction_api_kind(parsed[3], metadata.get("api_source"))
    return metadata


def parse_asm_gz(
    path: Path,
    pe_path: Path | None = None,
    api_ids: ApiVocab | None = None,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    """Stream an IDA listing twice: imports first, then instruction rows.

    Pass ``pe_path`` when the original PE is available. This dataset stores
    IDA text only, so the Mendeley run leaves ordinal and IAT RVA empty and
    takes import names from ``extrn`` lines.
    """
    path = Path(path)
    pe_rows = imports_from_pe(Path(pe_path)) if pe_path else []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        ida_rows = collect_ida_imports(handle)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        return tokenize_lines(handle, ida_rows, pe_rows, api_ids)


def _pe_for_sample(sample_id: str, pe_root: Path | None) -> Path | None:
    """Find a PE whose filename is the sample SHA-256, if one was supplied."""
    if pe_root is None:
        return None
    root = pe_root.resolve()
    for suffix in (".exe", ".dll", ".pe"):
        candidate = root / f"{sample_id}{suffix}"
        if candidate.is_file() and root in candidate.parents:
            return candidate
    return None


def load_api_vocab(path: Path) -> ApiVocab:
    """Load the name list. A missing file starts an empty vocabulary."""
    vocab = ApiVocab()
    if not path.is_file():
        return vocab
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        names = payload.get("names")
        if names is None and "extra" in payload:
            names = [
                name
                for name, _api_id in sorted(payload["extra"].items(), key=lambda item: int(item[1]))
            ]
        names = names or []
    else:
        names = payload
    for name in names:
        vocab.id_for(str(name))
    return vocab


def save_api_vocab(vocab: ApiVocab, path: Path) -> None:
    """Write the vocabulary as a JSON list of API names, in id order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.names, indent=2), encoding="utf-8")


def _selected_rows(roots: list[Path], split: str, architecture: str | None, sample_id: str | None):
    splits = ["train", "test"] if split == "all" else [split]
    seen: dict[str, dict[str, str]] = {}
    for root in roots:
        root = root.resolve()
        for split_name in splits:
            manifest = root / "splits" / f"{split_name}.csv"
            with manifest.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if architecture and row["architecture"] != architecture:
                        continue
                    if sample_id and row["sample_id"] != sample_id:
                        continue
                    if row["asm_status"] != "ok" or not row["asm_path"]:
                        continue
                    sample = row["sample_id"]
                    prior = seen.get(sample)
                    if prior is not None:
                        if prior["label"] != row["label"]:
                            raise ValueError(f"conflicting labels for SHA-256: {sample}")
                        continue
                    asm_path = (root / row["asm_path"]).resolve()
                    if root not in asm_path.parents:
                        raise ValueError(f"unsafe assembly path: {row['asm_path']}")
                    seen[sample] = {
                        "sample_id": sample,
                        "label": row["label"],
                        "architecture": row["architecture"],
                        "collection": row["collection"],
                        "split": split_name,
                        "asm_path": str(asm_path),
                        "asm_bytes": row.get("asm_bytes") or "",
                        "num_blocks": row.get("num_blocks") or "",
                        "num_instructions": row.get("num_instructions") or "",
                    }
    return list(seen.values())


def _write_split_csv(split_name: str, rows: list[dict[str, str]]) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"{split_name}.csv"
    fieldnames = [
        "file_id",
        "file_path",
        "label",
        "architecture",
        "collection",
        "num_instructions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        dest="roots",
        help="Collection folder. Repeat to mix collections. Defaults to both Mendeley folders.",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--architecture", choices=["x86", "x64"])
    parser.add_argument("--sample-id", help="Tokenize one SHA-256 only.")
    parser.add_argument("--limit", type=int, help="Stop after this many saved or reused files.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--pe-root",
        type=Path,
        help="Folder of original PE files named <sha256>.exe, .dll, or .pe. "
        "Omitted for the Mendeley listings, which have no binaries.",
    )
    args = parser.parse_args()

    roots = [path.resolve() for path in (args.roots or DEFAULT_ROOTS)]
    for root in roots:
        if not (root / "splits").is_dir():
            parser.error(f"missing splits folder: {root}")
    if args.pe_root is not None and not args.pe_root.is_dir():
        parser.error(f"missing PE folder: {args.pe_root}")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_rows(roots, args.split, args.architecture, args.sample_id)
    if args.limit is not None:
        selected = selected[: args.limit]

    saved = 0
    reused = 0
    skipped_empty = 0
    failed = 0
    parsed_any = False
    full_run = args.limit is None and args.sample_id is None
    api_ids = load_api_vocab(API_VOCAB_PATH) if full_run else ApiVocab()
    manifests: dict[str, list[dict[str, str]]] = {"train": [], "test": []}

    for row in selected:
        sample_id = row["sample_id"]
        asm_path = Path(row["asm_path"])
        output_path = output_dir / f"{sample_id}.pt"
        try:
            payload = None
            if output_path.is_file() and not args.overwrite:
                payload = torch.load(output_path, weights_only=False)
                sequence = payload.get("sequence")
                inventory = payload.get("import_inventory")
                current = (
                    getattr(sequence, "ndim", 0) == 2
                    and sequence.shape[1] == 5
                    and payload.get("instruction_metadata")
                    and isinstance(inventory, dict)
                    and "imported_apis" in inventory
                )
                if current:
                    num_instructions = int(sequence.shape[0])
                    reused += 1
                    print(f"Reused {sample_id}: {num_instructions} instructions")
                else:
                    payload = None
            if payload is None:
                if not asm_path.is_file():
                    print(f"Missing asm: {asm_path}")
                    failed += 1
                    continue
                instructions, instruction_metadata, import_inventory = parse_asm_gz(
                    asm_path,
                    pe_path=_pe_for_sample(sample_id, args.pe_root),
                    api_ids=api_ids,
                )
                if not instructions:
                    print(f"Skipped empty file: {asm_path.name}")
                    skipped_empty += 1
                    continue
                sequence = torch.tensor(instructions, dtype=torch.long)
                torch.save(
                    {
                        "file_id": sample_id,
                        "sequence": sequence,
                        "label": int(row["label"]),
                        "asm_path": str(asm_path),
                        "architecture": row["architecture"],
                        "collection": row["collection"],
                        "instruction_metadata": instruction_metadata,
                        "import_inventory": import_inventory,
                    },
                    output_path,
                )
                parsed_any = True
                num_instructions = int(sequence.shape[0])
                saved += 1
                print(
                    f"Saved {sample_id}: {num_instructions} instructions, "
                    f"label={row['label']}, {row['architecture']}"
                )

            manifests[row["split"]].append(
                {
                    "file_id": sample_id,
                    "file_path": str(output_path),
                    "label": row["label"],
                    "architecture": row["architecture"],
                    "collection": row["collection"],
                    "num_instructions": str(num_instructions),
                }
            )
        except Exception as error:
            print(f"Failed: {asm_path}")
            print("Error:", error)
            failed += 1

    if full_run:
        for split_name, rows in manifests.items():
            if rows:
                _write_split_csv(split_name, rows)
        if parsed_any:
            save_api_vocab(api_ids, API_VOCAB_PATH)
            print(f"Wrote {API_VOCAB_PATH}")

    print("\nFinished.")
    print("Saved:", saved)
    print("Reused:", reused)
    print("Skipped empty:", skipped_empty)
    print("Failed:", failed)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()

import re
from pathlib import Path


TOKEN_MAP = {
    'PADDING': 0, 'UNKNOWN_OPCODE': 1, 'NOP_SLED': 2,
    
    # Control Flow / Jumps
    'JMP': 10, 'JZ': 11, 'JNZ': 12, 'JE': 13, 'JNE': 14, 
    'JS': 15, 'JNS': 16, 'JG': 17, 'JL': 18, 'JGE': 19, 'JLE': 20,
    'CALL': 21, 'RET': 22, 'RETN': 23,
    
    # Data Movement
    'MOV': 25, 'PUSH': 26, 'POP': 27, 'LEA': 28, 'XCHG': 29,
    
    # Arithmetic Processing
    'ADD': 30, 'SUB': 31, 'INC': 32, 'DEC': 33, 'CMP': 34, 'TEST': 35,
    'IMUL': 36, 'MUL': 37, 'IDIV': 38, 'DIV': 39,
    
    # Logical / Bitwise Algebra
    'XOR': 40, 'OR': 41, 'AND': 42, 'NOT': 43, 
    'SHL': 44, 'SHR': 45, 'ROL': 46, 'ROR': 47, 'SAR': 48, 'SAL': 49,
    'NEG': 50,
    
    # Floating Point & Vectorized/Cryptographic Mixing
    'FSTP': 55, 'FLD': 56, 'PADD': 57, 'PXOR': 58, 'MOVDQA': 59,
    
    # System Level Interruption & Profiling
    'INT': 60, 'SYSCALL': 61, 'SYSENTER': 62,    
    'CPUID': 63, 'RDTSC': 64,
    
    # Operand Structural Tokens
    'REG_DATA': 70,       # Data registers (EAX, EBX, etc.)
    'REG_STACK': 71,      # Stack pointers (EBP, ESP)     
    'MEM_STACK_REF': 72,  # Memory references via EBP/ESP
    'MEM_GLOBAL_REF': 73, # Pure pointer brackets [0x401000]
    'IMM_VALUE': 74,       # Constants/Immediate values
    'UNKNOWN_API': 75     # Fallback for unlisted APIs
}


# Paste your complete existing API_MAP here
API_MAP = {
    'amsiscanbuffer': 76, 'etweventwrite': 77, 'nttraceevent': 78,
    'impersonatedloggedonuser': 79, 'duplicatetokenex': 80, 'adjusttokenprivileges': 81,
    'openprocesstoken': 82, 'deviceiocontrol': 83, 'ntdeviceiocontrolfile': 84,
    'cocreateinstance': 85, 'coinitalizeex': 86, 'clsidfromstring': 87,
    'ntopenprocesstoken': 88, 'openmutexa': 89, 'createmutexa': 90,
    'createprocesswithlogonw': 91, 'setthreadplaceholder': 92, 'switchtothread': 93,
    'registerwindowmessagea': 94, 'tpallocwork': 95, 'tppostwork': 96,
    'lookupprivilegevaluea': 97, 'ntadjustprivilegestoken': 98,  'ntcreatesection': 99,
    'virtualalloc': 100, 'virtualallocex': 101, 'writeprocessmemory': 102, 
    'createthread': 103, 'createremotethread': 104, 'ntmapviewofsection': 105,
    'virtualprotect': 106, 'virtualprotectex': 107, 'openprocess': 108,
    'ntallocatevirtualmemory': 109, 'ntwritevirtualmemory': 110, 'ntprotectvirtualmemory': 111,
    'ntcreatethreadex': 112, 'queueuserapc': 113, 'rtlcreateremotethread': 114,
    'setthreadcontext': 115, 'getthreadcontext': 116, 'ntopenprocess': 117,
    'ntsetcontextthread': 118, 'ntgetcontextthread': 119,
    'getprocaddress': 120, 'loadlibrarya': 121, 'loadlibraryw': 122, 
    'loadlibraryex': 123, 'ldrloaddll': 124, 'getmodulehandlea': 125, 'getmodulehandlew': 126,
    'getmodulefilenamea': 127, 'getmodulefilenamew': 128, 'ldrgetprocedureaddress': 129,
    'freelibrary': 130, 'exitprocess': 131, 'terminateprocess': 132,
    'ntterminateprocess': 133, 'ntqueryinformationprocess': 134, 'rtlgetversion': 135,
    'iswow64process': 136, 'getstartuptime': 137, 'getcommandlinea': 138, 'getcommandlinew': 139,
    'createfilea': 140, 'createfilew': 141, 'writefile': 142, 'readfile': 143, 
    'copyfilea': 144, 'copyfilew': 145, 'deletefilea': 146, 'deletefilew': 147,
    'getsystemdirectorya': 148, 'getsystemdirectoryw': 149, 'getwindowsdirectorya': 150,
    'getwindowsdirectoryw': 151, 'movefileexw': 152, 'movefileexa': 153,
    'getfilesize': 154, 'setfilepointer': 155, 'findfirstfilea': 156,
    'findnextfilea': 157, 'findfirstfilew': 158, 'findnextfilew': 159,
    'regsetvalueex': 160, 'regopenkeyex': 161, 'regcreatekeyex': 162, 
    'regqueryvalueex': 163, 'regclosekey': 164, 'regdeletevaluea': 165,
    'regdeletevaluew': 166, 'regenumkeyexw': 167, 'regenumkeyexa': 168,
    'ntopenkey': 169, 'ntsetvaluekey': 170, 'ntqueryvaluekey': 171,
    'shreggetvaluew': 172, 'shreggetvaluea': 173, 'regsavekeya': 174,
    'internetopena': 175, 'internetopenw': 176, 'internetconnecta': 177,
    'internetconnectw': 178, 'httpsendrequesta': 179, 'httpsendrequestw': 180,
    'internetreadfile': 181, 'internetwritefile': 182, 'wsastartup': 183,
    'socket': 184, 'connect': 185, 'send': 186, 'recv': 187,
    'wsasend': 188, 'wsarecv': 189, 'getaddrinfo': 190, 'gethostbyname': 191,
    'closesocket': 192, 'internetopenurla': 193, 'internetopenurlw': 194,
    'isdebuggerpresent': 195, 'checkremotedebuggerpresent': 196, 'gettickcount': 197,
    'gettickcount64': 198, 'sleep': 199, 'ntdelayexecution': 200,
    'outputdebugstringa': 201, 'outputdebugstringw': 202, 'ntquerysysteminformation': 203,
    'getsystemtime': 204, 'getlocaltime': 205, 'getasynckeystate': 206,
    'setwindowshookexa': 207, 'setwindowshookexw': 208, 'callnexthookex': 209,
    'process32first': 210, 'process32next': 211, 'createtoolhelp32snapshot': 212,
    'getcomputernamea': 213, 'getcomputernamew': 214, 'getusernamea': 215,
    'getusernamew': 216, 'getnativeid': 217, 'enumprocesses': 218,
    'enumprocessmodules': 219, 'getcurrentprocessid': 220, 'getcurrentthreadid': 221,
    'getlogicaldrives': 222, 'getdriveclass': 223, 'getdiskfreespaceexa': 224,
    'cryptacquirecontexta': 225, 'cryptacquirecontextw': 226, 'cryptcreatehash': 227, 
    'crypthashdata': 228, 'cryptderivekey': 229, 'cryptencrypt': 230,
    'cryptdecrypt': 231, 'cryptdestroykey': 232, 'cryptdestroyhash': 233,
    'cryptreleasecontext': 234, 'bcreptopenalgorithmprovider': 235, 'bcryptexecute': 236,
    'cryptgenrandom': 237, 'rtlcomputecrc32': 238, 'cryptstringtobinarya': 239,
    'shellexecutea': 240, 'shellexecutew': 241, 'createprocessa': 242, 
    'createprocessw': 243, 'resumethread': 244, 'suspendthread': 245,
    'unmapviewofsection': 246, 'findresourcea': 247, 'findresourcew': 248, 
    'loadresource': 249, 'lockresource': 250, 'gettemppatha': 251, 
    'gettemppathw': 252, 'winexec': 253, 'ntcreateuserprocess': 254, 'createprocessasuserw': 255
}


# Fourth token channel: what the instruction is doing (not the binary label).
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

# Behavior-head / explainability labels (subset of BEHAVIOR_MAP).
HEAD_BEHAVIOR_NAMES = (
    "CRYPTO_OPCODE",
    "ENCRYPT",
    "DECRYPT",
    "FILE_READ",
    "FILE_WRITE",
)
HEAD_BEHAVIOR_IDS = tuple(BEHAVIOR_MAP[name] for name in HEAD_BEHAVIOR_NAMES)

SUSPICIOUS_BEHAVIOR_IDS = frozenset(
    {
        BEHAVIOR_MAP["CRYPTO_OPCODE"],
        BEHAVIOR_MAP["DECRYPT"],
        BEHAVIOR_MAP["ENCRYPT"],
        BEHAVIOR_MAP["KEY_SETUP"],
        BEHAVIOR_MAP["FILE_ENUM"],
        BEHAVIOR_MAP["FILE_READ"],
        BEHAVIOR_MAP["FILE_WRITE"],
        BEHAVIOR_MAP["FILE_DELETE"],
    }
)
CRYPTO_BEHAVIOR_IDS = frozenset(
    {
        BEHAVIOR_MAP["CRYPTO_OPCODE"],
        BEHAVIOR_MAP["DECRYPT"],
        BEHAVIOR_MAP["ENCRYPT"],
        BEHAVIOR_MAP["KEY_SETUP"],
    }
)
FILE_BEHAVIOR_IDS = frozenset(
    {
        BEHAVIOR_MAP["FILE_ENUM"],
        BEHAVIOR_MAP["FILE_READ"],
        BEHAVIOR_MAP["FILE_WRITE"],
        BEHAVIOR_MAP["FILE_DELETE"],
    }
)

# Mixing-style opcodes often used in software crypto routines.
CRYPTO_OPCODES = frozenset(
    {
        "XOR",
        "PXOR",
        "ROL",
        "ROR",
        "SHL",
        "SHR",
        "SAL",
        "SAR",
        "MOVDQA",
        "MOVDQU",
        "AESENC",
        "AESENCLAST",
        "AESDEC",
        "AESDECLAST",
        "AESKEYGENASSIST",
        "AESIMC",
    }
)

# Known API name (lowercase) -> behavior category.
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

# Extra file APIs that share an existing opcode id so the embedding stays valid.
# The value is the API_MAP key whose behavior tag should be used.
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

_API_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_@]*")


REG_DATA_REGEX = re.compile(
    r"\b(rax|rbx|rcx|rdx|rsi|rdi|r8|r9|r10|r11|r12|r13|r14|r15|"
    r"eax|ebx|ecx|edx|esi|edi|ax|bx|cx|dx|al|bl|cl|dl)\b",
    re.IGNORECASE,
)

REG_STACK_REGEX = re.compile(
    r"\b(rbp|rsp|ebp|esp|bp|sp)\b",
    re.IGNORECASE,
)

MEM_STACK_REGEX = re.compile(
    r"\[\s*(rbp|rsp|ebp|esp|bp|sp)\s*[\+\-].*\]",
    re.IGNORECASE,
)

MEM_REGEX = re.compile(r"\[.*\]")

IMM_REGEX = re.compile(
    r"\b(0x[0-9a-fA-F]+|[0-9]+h?)\b"
)


OPCODE_PREFIXES = {"LOCK", "REP", "REPE", "REPZ", "REPNE", "REPNZ"}


def strip_opcode_prefixes(opcode: str) -> str:
    """Drop lock/rep prefixes so 'rep stosb' tokenizes as STOSB."""
    parts = opcode.upper().split()
    while parts and parts[0] in OPCODE_PREFIXES:
        parts.pop(0)
    return parts[-1] if parts else opcode.upper()


def categorize_operand(operand):
    if not operand:
        return TOKEN_MAP["PADDING"]

    operand = operand.strip()

    if MEM_REGEX.search(operand):
        if MEM_STACK_REGEX.search(operand):
            return TOKEN_MAP["MEM_STACK_REF"]

        return TOKEN_MAP["MEM_GLOBAL_REF"]

    if REG_DATA_REGEX.search(operand):
        return TOKEN_MAP["REG_DATA"]

    if REG_STACK_REGEX.search(operand):
        return TOKEN_MAP["REG_STACK"]

    if IMM_REGEX.search(operand):
        return TOKEN_MAP["IMM_VALUE"]

    return TOKEN_MAP["MEM_GLOBAL_REF"]


def _match_api_in_operands(raw_operands: list[str]) -> str | None:
    if not raw_operands:
        return None
    tokens = [token.lower() for token in _API_IDENT.findall(" ".join(raw_operands))]
    # Whole-name match first, so "read" does not steal "ReadFile" or "OpenProcess".
    for token in sorted(set(tokens), key=len, reverse=True):
        if token in API_MAP:
            return token
        alias = API_ALIASES.get(token)
        if alias is not None:
            return alias
    target = " ".join(raw_operands).lower()
    for api_name in sorted(API_MAP, key=len, reverse=True):
        if api_name in target:
            return api_name
    return None


def categorize_behavior(opcode: str, opcode_id: int, matched_api: str | None) -> int:
    """Map one instruction to a behavior category id (fourth token channel)."""
    if matched_api is not None:
        behavior = API_BEHAVIOR.get(matched_api)
        if behavior is not None:
            return behavior
        return BEHAVIOR_MAP["NONE"]

    if opcode in CRYPTO_OPCODES:
        return BEHAVIOR_MAP["CRYPTO_OPCODE"]

    return BEHAVIOR_MAP["NONE"]


def encode_instruction(opcode: str, raw_operands: list[str]):
    """
    Map mnemonic + operands to
    [opcode_or_api_id, operand_1_id, operand_2_id, behavior_id].

    Capstone CALLs are usually addresses / mem refs — keep CALL unless
    the target string actually contains a known API name.
    """
    opcode = strip_opcode_prefixes(opcode)
    matched_api = _match_api_in_operands(raw_operands)

    if opcode == "CALL":
        opcode_id = TOKEN_MAP["CALL"]
        if matched_api is not None:
            opcode_id = API_MAP[matched_api]
    elif opcode == "NOP":
        opcode_id = TOKEN_MAP["NOP_SLED"]
    else:
        opcode_id = TOKEN_MAP.get(opcode, TOKEN_MAP["UNKNOWN_OPCODE"])

    operand_1_id = (
        categorize_operand(raw_operands[0])
        if len(raw_operands) >= 1
        else TOKEN_MAP["PADDING"]
    )
    operand_2_id = (
        categorize_operand(raw_operands[1])
        if len(raw_operands) >= 2
        else TOKEN_MAP["PADDING"]
    )
    behavior_id = categorize_behavior(opcode, opcode_id, matched_api)

    return [opcode_id, operand_1_id, operand_2_id, behavior_id]


def parse_capstone_insn(mnemonic: str, op_str: str = ""):
    """Encode one Capstone instruction (mnemonic + op_str) into token IDs."""
    raw_operands = [
        operand.strip()
        for operand in op_str.split(",")
        if operand.strip()
    ]
    return encode_instruction(mnemonic, raw_operands)


def parse_asm_line(line):
    """
    Parse one assembly line.

    Returns:
        [opcode_or_api_id, operand_1_id, operand_2_id, behavior_id]

    Returns None for comments, directives, or invalid lines.
    """

    line = line.strip()

    # Ignore empty lines and full-line comments
    if not line or line.startswith(";"):
        return None

    # Remove inline comments
    if ";" in line:
        line = line.split(";", 1)[0].strip()

    tokens = line.split()

    # Expected format begins with an address such as:
    # 00401000: 55 push ebp
    if len(tokens) < 2 or ":" not in tokens[0]:
        return None

    ignored_keywords = [
        "public",
        "assume",
        "proc",
        "endp",
        "segment",
        "ends",
        "unicode",
        "extrn",
    ]

    if any(keyword in line.lower() for keyword in ignored_keywords):
        return None

    ignored_directives = ["db", "dw", "dd", "align"]

    if any(directive in tokens for directive in ignored_directives):
        return None

    # Remove the address
    code_tokens = tokens[1:]

    # Remove hexadecimal machine-code bytes
    code_tokens = [
        token
        for token in code_tokens
        if not re.fullmatch(r"[0-9A-Fa-f]{2}", token)
        and not re.fullmatch(r"[0-9A-Fa-f]{4}", token)
    ]

    if not code_tokens:
        return None

    prefix_index = 0
    while (
        prefix_index < len(code_tokens)
        and code_tokens[prefix_index].upper() in OPCODE_PREFIXES
    ):
        prefix_index += 1
    if prefix_index >= len(code_tokens):
        return None

    opcode = code_tokens[prefix_index]
    operand_text = " ".join(code_tokens[prefix_index + 1:])
    raw_operands = [
        operand.strip()
        for operand in operand_text.split(",")
        if operand.strip()
    ]

    return encode_instruction(opcode, raw_operands)


def parse_asm_file(asm_path):
    """
    Parse an entire ASM file.

    Returns:
        A list of instructions, where each instruction is:
        [opcode_or_api_id, operand_1_id, operand_2_id, behavior_id]
    """

    asm_path = Path(asm_path)
    parsed_instructions = []

    with asm_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:
            parsed_line = parse_asm_line(line)

            if parsed_line is not None:
                parsed_instructions.append(parsed_line)

    return parsed_instructions


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("python token_mapping.py path/to/file.asm")
        sys.exit(1)

    asm_path = sys.argv[1]
    instructions = parse_asm_file(asm_path)

    print("ASM file:", asm_path)
    print("Parsed instructions:", len(instructions))
    print("\nFirst 10 instructions:")

    for instruction in instructions[:10]:
        print(instruction)

    tagged = [row for row in instructions if row[3] != BEHAVIOR_MAP["NONE"]]
    print(f"\nBehavior-tagged instructions: {len(tagged)}")
    for instruction in tagged[:10]:
        print(instruction)
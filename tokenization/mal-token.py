#!/usr/bin/env python3
"""
Goal:
- reads local malware assembly/disassembly text files
- extracts raw opcodes
- maps opcodes to grouped tokens like MOVE, JUMP, CALL
- maps grouped/API-hint tokens to coarse behavior tokens
- builds vocabularies and integer token sequences
- writes dataset.jsonl, vocab.json, and stats.json

Usage:
    python3 mal-token.py ./malware --output ./tokenized_out
"""
import os
import re
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer, SentencePieceBPETokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace


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
    "CALL": {"call"},
    "RET": {"ret", "retn", "retf"},
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
    "STACK_FRAME": {"enter", "leave"},
    "SYSTEM": {"int", "syscall", "sysenter", "sysexit", "cpuid", "hlt", "nop", "ud2"},
    "FLOAT_SIMD": {
        "fld", "fst", "fstp", "fadd", "fsub", "fmul", "fdiv",
        "pxor", "movdqa", "movdqu", "movaps", "movups", "paddb", "paddw", "paddd"
    },
    "CRYPTO": {
        "aesenc", "aesenclast", "aesdec", "aesdeclast", "aesimc", "aeskeygenassist",
        "rdrand", "rdseed"
    },
}

OPCODE_TO_GROUP = {}
for group_name, opcodes in OPCODE_GROUPS.items():
    for opcode in opcodes:
        OPCODE_TO_GROUP[opcode] = group_name


def map_opcode_token(token):
    token = str(token).strip().lower()
    return OPCODE_TO_GROUP.get(token, "OTHER")


def group_opcode_sequence(text):
    if pd.isna(text):
        return ""
    tokens = str(text).split()
    mapped = [map_opcode_token(tok) for tok in tokens]
    return " ".join(mapped)


# ----------------------------
# ASM parsing helpers
# ----------------------------

VALID_EXTENSIONS = {".asm", ".s", ".lst", ".txt"}

LABEL_RE = re.compile(r"^\s*[A-Za-z_.$?@][\w.$?@]*:\s*$")
MNEMONIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
HEX_OR_ADDR_RE = re.compile(r"^(0x[0-9a-f]+|[0-9a-f]+h|[0-9a-f]{2,})$", re.I)

NON_OPCODE_TOKENS = {
    "db", "dw", "dd", "dq", "dt",
    "align", "assume", "end", "ends", "segment", "proc", "endp",
    "public", "extrn", "extern", "model", "include", "equ",
}


def is_probable_opcode(tok):
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


def extract_opcode_from_line(line):
    line = line.split(";", 1)[0].split("#", 1)[0].strip()
    if not line:
        return None
    if LABEL_RE.match(line):
        return None

    parts = line.split()
    for raw in parts[:6]:
        tok = raw.strip().rstrip(":").lower()
        if not tok:
            continue
        if HEX_OR_ADDR_RE.match(tok):
            continue
        if ":" in tok and tok.count(":") == 1:
            continue
        if is_probable_opcode(tok):
            return tok
    return None


def extract_opcodes_from_file(path):
    opcodes = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                opcode = extract_opcode_from_line(line)
                if opcode:
                    opcodes.append(opcode)
    except Exception as e:
        print(f"Skipping {path}: {e}")

    return " ".join(opcodes)


def build_dataset_from_asm_folder(root_dir):
    rows = []

    for root, _, files in os.walk(root_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            full_path = os.path.join(root, filename)
            opcode_text = extract_opcodes_from_file(full_path)

            if opcode_text.strip():
                rows.append({
                    "File": os.path.relpath(full_path, root_dir),
                    "Opcodes": opcode_text
                })

    return pd.DataFrame(rows)


# ----------------------------
# Cleaning helpers
# ----------------------------

def removeNonVocab(vocab, series):
    vocab = set(vocab)
    rows = []
    for row in series:
        tokens = str(row).split()
        cleaned = [tok for tok in tokens if tok in vocab]
        rows.append(cleaned)
    return rows


def batch_iterator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data["Opcodes"].iloc[i:i + batch_size].astype(str).tolist()


def count_unigrams(df, c):
    for row in df["Opcodes"]:
        data = str(row).split()
        c.update(data)


def make_bigram_strings(tokens):
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def count_bigrams(df, c, name):
    bigrams_list = []

    for row in df["Opcodes"]:
        data = str(row).split()
        bigrms = make_bigram_strings(data)
        bigrams_list.append(" ".join(bigrms))

        if name == "train":
            c.update(bigrms)

    return bigrams_list


# ----------------------------
# Tokenizer setup
# ----------------------------

def prepare_tokenizer_trainer(alg, v_size, unk_token, spl_tokens):
    if alg == "BPE":
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=v_size)
    elif alg == "UNI":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            unk_token=unk_token,
            special_tokens=spl_tokens,
            vocab_size=v_size
        )
    elif alg == "WPC":
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(
            special_tokens=spl_tokens,
            vocab_size=v_size
        )
    else:
        raise ValueError(f"Unsupported algorithm: {alg}")

    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer


def train_tokenizer(alg, train_data, v_size, unk_token, spl_tokens, batch_size):
    if alg in {"BPE", "UNI", "WPC"}:
        tokenizer, trainer = prepare_tokenizer_trainer(
            alg, v_size, unk_token, spl_tokens
        )
        tokenizer.train_from_iterator(
            batch_iterator(train_data, batch_size),
            trainer=trainer
        )
        return tokenizer

    if alg == "SPC":
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(
            batch_iterator(train_data, batch_size),
            vocab_size=v_size,
            special_tokens=spl_tokens
        )
        return tokenizer

    raise ValueError(f"Unsupported algorithm: {alg}")


def encode(tokenizer, df):
    encoded = tokenizer.encode_batch(df["Opcodes"].astype(str).tolist())
    return [encoding.tokens for encoding in encoded]


# ----------------------------
# Tokenization modes
# ----------------------------

def single_words(train, test, out_prefix):
    size = 31

    sw_train = train.copy()
    sw_test = test.copy()

    count_total = Counter()
    count_unigrams(sw_train, count_total)

    count_list = [x[0] for x in count_total.most_common(size)]
    print("Top single tokens:", count_list)

    sw_train["Opcodes"] = removeNonVocab(count_list, sw_train["Opcodes"])
    sw_test["Opcodes"] = removeNonVocab(count_list, sw_test["Opcodes"])

    sw_train.to_pickle(f"TokenizeResults/{out_prefix}_SW_train.pkl")
    sw_test.to_pickle(f"TokenizeResults/{out_prefix}_SW_test.pkl")


def word_pairs(train, test, out_prefix):
    size = 30

    wp_train = train.copy()
    wp_test = test.copy()

    count_total = Counter()
    bigrams_train = count_bigrams(wp_train, count_total, "train")
    bigrams_test = count_bigrams(wp_test, count_total, "test")

    count_list = [x[0] for x in count_total.most_common(size)]
    print("Top word-pair tokens:", count_list)

    wp_train["Opcodes"] = bigrams_train
    wp_test["Opcodes"] = bigrams_test

    wp_train["Opcodes"] = removeNonVocab(count_list, wp_train["Opcodes"])
    wp_test["Opcodes"] = removeNonVocab(count_list, wp_test["Opcodes"])

    wp_train.to_pickle(f"TokenizeResults/{out_prefix}_WP_train.pkl")
    wp_test.to_pickle(f"TokenizeResults/{out_prefix}_WP_test.pkl")


def BPE_tokens(train, test, out_prefix):
    batch_size = 1000
    unk_token = "<UNK>"
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
    v_size = 1000

    tokenizer = train_tokenizer("BPE", train, v_size, unk_token, spl_tokens, batch_size)
    tokenizer.save(f"TokenizerData/{out_prefix}_BPE-trained.json")

    bpe_train = train.copy()
    bpe_test = test.copy()

    bpe_train["Opcodes"] = encode(tokenizer, bpe_train)
    bpe_test["Opcodes"] = encode(tokenizer, bpe_test)

    bpe_train.to_pickle(f"TokenizeResults/{out_prefix}_BPE_train.pkl")
    bpe_test.to_pickle(f"TokenizeResults/{out_prefix}_BPE_test.pkl")


def WPC_tokens(train, test, out_prefix):
    batch_size = 1000
    unk_token = "<UNK>"
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
    v_size = 500

    tokenizer = train_tokenizer("WPC", train, v_size, unk_token, spl_tokens, batch_size)
    tokenizer.save(f"TokenizerData/{out_prefix}_WPC-trained.json")

    wpc_train = train.copy()
    wpc_test = test.copy()

    wpc_train["Opcodes"] = encode(tokenizer, wpc_train)
    wpc_test["Opcodes"] = encode(tokenizer, wpc_test)

    wpc_train.to_pickle(f"TokenizeResults/{out_prefix}_WPC_train.pkl")
    wpc_test.to_pickle(f"TokenizeResults/{out_prefix}_WPC_test.pkl")


def SPC_tokens(train, test, out_prefix):
    batch_size = 1000
    unk_token = "<UNK>"
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
    v_size = 500

    tokenizer = train_tokenizer("SPC", train, v_size, unk_token, spl_tokens, batch_size)
    tokenizer.save(f"TokenizerData/{out_prefix}_SPC-trained.json")

    spc_train = train.copy()
    spc_test = test.copy()

    spc_train["Opcodes"] = encode(tokenizer, spc_train)
    spc_test["Opcodes"] = encode(tokenizer, spc_test)

    spc_train.to_pickle(f"TokenizeResults/{out_prefix}_SPC_train.pkl")
    spc_test.to_pickle(f"TokenizeResults/{out_prefix}_SPC_test.pkl")


def UNI_tokens(train, test, out_prefix):
    batch_size = 1000
    unk_token = "<UNK>"
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
    v_size = 500

    tokenizer = train_tokenizer("UNI", train, v_size, unk_token, spl_tokens, batch_size)
    tokenizer.save(f"TokenizerData/{out_prefix}_UNI-trained.json")

    uni_train = train.copy()
    uni_test = test.copy()

    uni_train["Opcodes"] = encode(tokenizer, uni_train)
    uni_test["Opcodes"] = encode(tokenizer, uni_test)

    uni_train.to_pickle(f"TokenizeResults/{out_prefix}_UNI_train.pkl")
    uni_test.to_pickle(f"TokenizeResults/{out_prefix}_UNI_test.pkl")


def main():
    asm_dir = "/home/doroluo/CMPE295/program-files/ransomware"

    os.makedirs("TokenizeResults", exist_ok=True)
    os.makedirs("TokenizerData", exist_ok=True)

    print("Reading ransomware asm files")
    dataset = build_dataset_from_asm_folder(asm_dir)

    if dataset.empty:
        raise ValueError(f"No opcode data extracted from folder: {asm_dir}")

    print(f"Loaded {len(dataset)} files")

    train = dataset.sample(frac=0.7, random_state=42)
    test = dataset.drop(train.index)

    print("Starting RAW opcode tokenization")
    single_words(train, test, "RAW")
    word_pairs(train, test, "RAW")
    BPE_tokens(train, test, "RAW")
    WPC_tokens(train, test, "RAW")
    SPC_tokens(train, test, "RAW")
    UNI_tokens(train, test, "RAW")

    print("Converting opcodes to grouped tokens")
    grouped_train = train.copy()
    grouped_test = test.copy()

    grouped_train["Opcodes"] = grouped_train["Opcodes"].apply(group_opcode_sequence)
    grouped_test["Opcodes"] = grouped_test["Opcodes"].apply(group_opcode_sequence)

    print("Starting GROUP opcode tokenization")
    single_words(grouped_train, grouped_test, "GROUP")
    word_pairs(grouped_train, grouped_test, "GROUP")
    BPE_tokens(grouped_train, grouped_test, "GROUP")
    WPC_tokens(grouped_train, grouped_test, "GROUP")
    SPC_tokens(grouped_train, grouped_test, "GROUP")
    UNI_tokens(grouped_train, grouped_test, "GROUP")

    print("Done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Loading `seq_model/config.yaml` under either interpreter.

pyyaml is present under the 3.14 system interpreter and absent from the GPU
venv, and the runner has to read the same file under both, so this module uses
pyyaml when it is importable and otherwise parses the small YAML subset the
config file is written in: nested block mappings, block sequences of scalars,
`#` comments, and JSON-ish scalars. `tests/test_seq_model.py` checks the
fallback parser against pyyaml on the real config file, so the subset cannot
drift without a test failing.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def _scalar(s: str):
    s = s.strip()
    if s.startswith("#"):
        return None
    if s.startswith('"') and s.endswith('"'):
        try:
            return json.loads(s)          # keeps \\ and \" the way pyyaml does
        except ValueError:
            return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [] if not inner else [_scalar(x) for x in inner.split(",")]
    low = s.lower()
    if low in ("null", "~", ""):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return json.loads(s)
    except Exception:
        return s


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _parse(lines, i: int, indent: int):
    """Returns (value, next_index) for the block starting at `lines[i]`."""
    if i < len(lines) and lines[i][1].startswith("- "):
        seq = []
        while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
            seq.append(_scalar(lines[i][1][2:]))
            i += 1
        return seq, i
    out = {}
    while i < len(lines) and lines[i][0] == indent:
        ind, text = lines[i]
        key, _, rest = text.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest:
            out[key] = _scalar(rest)
            i += 1
        else:
            i += 1
            if i < len(lines) and lines[i][0] > indent:
                out[key], i = _parse(lines, i, lines[i][0])
            else:
                out[key] = None
    return out, i


def load_yaml(path=CONFIG_PATH) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        pass
    lines = []
    for raw in text.splitlines():
        body = _strip_comment(raw)
        if not body.strip():
            continue
        lines.append((len(body) - len(body.lstrip(" ")), body.strip()))
    if not lines:
        return {}
    doc, _ = _parse(lines, 0, lines[0][0])
    return doc


def load(path=CONFIG_PATH) -> dict:
    return load_yaml(path)


# ---------------------------------------------------------------------------
# ablations: the only sanctioned way to change the frozen configuration
# ---------------------------------------------------------------------------
def ablation(cfg: dict, name: str) -> dict:
    """Return a copy of `cfg` with exactly the one change `name` names.

    `single_seed` is not here: it is read off the main run's per-seed
    predictions rather than retrained, as the brief asks.
    """
    c = copy.deepcopy(cfg)
    # every ablation is a mendeley K-fold study at ONE seed, as the brief
    # asks; the seed is the main run's first, so the `single_seed` row read off
    # the main run is the like-for-like comparison for all of them.
    c["seeds"] = {"kfold": [cfg["seeds"]["kfold"][0]], "lofo": []}
    if name == "no_pretrain":
        c["pretrain"]["enabled"] = False
    elif name == "first_window_only":
        # one window of ~21,845 tokens = the slice CNN-ViT/asm_parser.py's
        # 256x256 canvas could hold, rounded down to the stem's stride so the
        # conv stack divides it exactly.
        c["windows"]["n_windows"] = 1
        c["windows"]["window"] = 21840
        c["training"]["batch_size"] = 6
        c["training"]["eval_batch_size"] = 6
    elif name == "no_arch_balance":
        c["sampler"]["arch_balanced"] = False
    else:
        raise ValueError(f"unknown ablation {name!r}")
    c["ablation"] = name
    return c


ABLATIONS = ("no_pretrain", "first_window_only", "no_arch_balance")

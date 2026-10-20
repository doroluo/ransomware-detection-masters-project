"""Why the listing-to-CFG match of a pipeline-B build fails, per label, per family and per cause.

python -m experiments.team_pipelines.match_diagnosis --build RUNS/team/b_sig500 [--source gnn-331ce0b|main-c7899f0]
       [--replay-worst 30] [--replay-goodware 10] [--replay-partial 20] [--max-listing-mb 60] [--among DIR] [--cache FILE]
       [--replay-cache DIR] [--json out.json] [--markdown out.md] [--csv out.csv]
       (run from the block_level_counting root; a leading RUNS/ is graph_transformer_runs)

No listing or CFG parser is written here. Listing lines are parsed and aligned by the teammates' own `parse_asm_instructions` and
`align_cfg_to_asm` / `align_sampled_blocks`, called through `import_team`. Two helpers do read raw listing text, and only to count
and to quote it: `scan_labels` counts label, `proc`, `endp` lines by their first tokens (using their `_va_from_token`), and
`excerpt` finds COLLAPSED FUNCTION markers with the same regex their module uses. Everything else comes from what the build
saved. Four parts.

1. Build-wide, from the saved tensors (`<id>.pt`: label, family, arch, `instruction_matched`, `block_keys`, `insn_to_block`):
   the match rate per sample (mean of `instruction_matched`), joined to label, family and arch; per label and per family the
   share below 50% and below 90%, the mean and the 10 worst families (families of at least --min-family-n samples); the
   AUC of `1 - match_rate` as a ransomware score with a 95% percentile bootstrap interval over clusters, drawn as
   evaluation/stats.py draws them (`fam:<family>` for ransomware, `id:<sample_id>` for goodware); and the lost share by
   function length (selected instructions of one function), which is a proxy for span length.
2. Build-wide, from the manifests' gap reports (`mismatch_report`): the counts of CFG instructions without a listing line
   (`missing_asm_count`), listing lines without a CFG instruction (`missing_json_count`), duplicate RVAs and address problems,
   per label, and the reasons of the examples a report keeps (at most 32 per kind). Scope differs by source: the first source
   (gnn-331ce0b) aligns the whole CFG, so its counts cover every CFG instruction; the second (main-c7899f0) aligns only the
   sampled blocks, and its counts are inflated by the `endp` reset (review section 11).
3. Replay of single samples (the --replay-worst lowest-match samples, the --replay-goodware lowest-match goodware, and
   --replay-partial random samples per label that lose some but not all of their instructions): the
   teammates' own `load_cfg`, `parse_asm_instructions` and alignment run on that sample, through `import_team` with the
   example limit of their mismatch report lifted in memory (their files are not touched). The selected blocks are the
   build's `block_keys`. Every selected instruction that stays unmatched gets one cause, in this order:
     iv   duplicate RVA: more than one CFG instruction at that address, which the pairing skips on purpose;
     i    no listing label at or below its RVA (`rva_before_any_asm_label`);
     vi   matched by the first source's alignment, unmatched by the build's (the `endp` reset of review section 11);
     iii  the CFG instruction's function has no listing lines: the function is a COLLAPSED FUNCTION in the listing (name
          found in the collapsed list), its entry RVA is no listing label, or its label span holds no listing line;
     v    a label kind her address rule does not read: the function's `proc` line is in the listing under another spelling
          of its name (so no anchor is set), or the loss follows a mnemonic disagreement in a label span that also holds a
          label outside her prefixes (`def_`, `nullsub_`, `j_`, other names);
     ii   lost after the first mnemonic disagreement of its label span; the (listing, CFG) mnemonic pair of each
          disagreement is counted, with its instructions lost, and sorted into prefixes, `retn`/`ret`, fillers and
          data printed as code, else other;
     vii  anything else (the listing's lines of the span ran out before the CFG's rows did).
   Also counted: listing code lines without a label address (the listing side of category i), label lines by kind, `endp`
   lines. Replayed counts are of selected instructions, the same ones `instruction_matched` covers; the replay of the build's
   own source must reproduce the build's matched count, and `consistent` says so.
4. The lost share by label-span length (CFG instructions between two listing labels) over the replayed samples.
"""
import argparse
import gc
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.team_pipelines.build_b import resolve_dir

LABELS = {1: 'ransomware', 0: 'goodware'}
CATEGORIES = (
    ('i', 'listing label missing below the RVA'),
    ('ii', 'mnemonic disagreement'),
    ('iii', 'function without listing lines (collapsed or absent)'),
    ('iv', 'duplicate RVA'),
    ('v', 'disagreement span with an unanchored label'),
    ('vi', '`endp` reset'),
    ('vii', 'other'),
)
LENGTH_BINS = ((1, 9), (10, 49), (50, 199), (200, 999), (1000, None))
RET = {'ret', 'retn', 'retf'}
PREFIXES = {'rep', 'repe', 'repne', 'repz', 'repnz', 'lock', 'bnd', 'notrack'}
FILLERS = {'nop', 'int3', 'int', 'align', 'pause'}
DATA_AS_CODE = {'db', 'dw', 'dd', 'dq', 'dt', 'offset', 'unicode', 'text', 'guid', 'add', 'adc', 'or', 'sbb'}
LABEL_PREFIX = re.compile(r'(sub|loc|locret|unk)_[0-9A-Fa-f]+', re.IGNORECASE)
COLLAPSED = re.compile(r'COLLAPSED FUNCTION\s+(.+?)\]', re.IGNORECASE)


# ----------------------------------------------------------------------------------------------------------------- build

def build_ids(build):
    ckpt = Path(build) / 'checkpoint.json'
    if ckpt.is_file():
        return sorted(json.loads(ckpt.read_text(encoding='utf-8'))['done'])
    return sorted(p.stem for p in Path(build).glob('*.pt') if p.name != 'channel_embedder.pt')


def build_source(build, default='gnn-331ce0b'):
    """The teammate source a build was made with (settings.json, else its checkpoint, else the first source)."""
    for name in ('settings.json', 'checkpoint.json'):
        path = Path(build) / name
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            source = data.get('source') or data.get('settings', {}).get('source')
            if source:
                return source
    return default


def cluster_of(label, family, sample_id):
    """evaluation/stats.py `pool`: one cluster per ransomware family, one per goodware binary."""
    return f'fam:{family}' if label == 1 else f'id:{sample_id}'


def function_sizes(payload):
    """[(selected instructions, matched ones)] per function of a saved sample."""
    fids = np.array([k['function_id'] for k in payload['block_keys']])
    per_instr = fids[payload['insn_to_block'].numpy()] if payload['insn_to_block'].numel() else np.array([], dtype=int)
    matched = payload['instruction_matched'].numpy().astype(bool)
    out = []
    for fid in np.unique(per_instr):
        sel = per_instr == fid
        out.append((int(sel.sum()), int(matched[sel].sum())))
    return out


def load_samples(build, ids=None):
    """[{sample_id, label, family, arch, cluster, n, matched, rate, functions}] from the `.pt` files of a build."""
    rows = []
    for sample_id in (build_ids(build) if ids is None else ids):
        p = torch.load(Path(build) / f'{sample_id}.pt', map_location='cpu', weights_only=False)
        meta = p.get('dataset_metadata') or {}
        m = p['instruction_matched']
        label = int(p['label'])
        family = meta.get('family', 'goodware' if label == 0 else 'unknown')
        rows.append({'sample_id': sample_id, 'label': label, 'family': family, 'arch': meta.get('arch', ''),
                     'cluster': cluster_of(label, family, sample_id), 'n': int(m.numel()), 'matched': int(m.sum()),
                     'rate': float(m.float().mean()) if m.numel() else 1.0, 'functions': function_sizes(p)})
    return rows


def share(values, pred):
    return sum(1 for v in values if pred(v)) / len(values) if values else float('nan')


def group_stats(rates):
    return {'n': len(rates), 'mean_match': float(np.mean(rates)) if rates else float('nan'),
            'n_below_50': sum(1 for r in rates if r < 0.5), 'n_below_90': sum(1 for r in rates if r < 0.9),
            'share_below_50': share(rates, lambda r: r < 0.5), 'share_below_90': share(rates, lambda r: r < 0.9),
            'share_zero': share(rates, lambda r: r == 0.0)}


def label_table(samples):
    return {name: group_stats([s['rate'] for s in samples if s['label'] == label]) for label, name in LABELS.items()}


def family_table(samples):
    by = defaultdict(list)
    for s in samples:
        by[s['family']].append(s['rate'])
    return {fam: group_stats(rates) for fam, rates in by.items()}


def worst_families(table, k=10, min_n=3):
    rows = [(fam, st) for fam, st in table.items() if st['n'] >= min_n]
    return sorted(rows, key=lambda r: (r[1]['mean_match'], r[0]))[:k]


def auc_of_loss(samples, resamples=1000, seed=0):
    """AUC of `1 - match_rate` as a ransomware score, with a 95% percentile interval over clusters. A resample draws
    clusters with replacement; its weight on a sample is how often the sample's cluster was drawn (stats.py)."""
    from sklearn.metrics import roc_auc_score
    y = np.array([s['label'] for s in samples])
    score = 1.0 - np.array([s['rate'] for s in samples])
    clusters, cidx = np.unique(np.array([s['cluster'] for s in samples]), return_inverse=True)
    point = float(roc_auc_score(y, score)) if len(set(y)) == 2 else float('nan')
    draws = np.random.default_rng(seed).integers(0, len(clusters), size=(resamples, len(clusters)))
    values = []
    for draw in draws:
        w = np.bincount(draw, minlength=len(clusters))[cidx]
        if len(set(y[w > 0])) == 2:
            values.append(roc_auc_score(y, score, sample_weight=w))
    lo, hi = np.percentile(values, [2.5, 97.5]) if values else (float('nan'), float('nan'))
    return {'auc': point, 'lo': float(lo), 'hi': float(hi), 'n_clusters': int(len(clusters)),
            'resamples_used': len(values)}


def length_bin(n):
    for lo, hi in LENGTH_BINS:
        if n >= lo and (hi is None or n <= hi):
            return (lo, hi)
    return LENGTH_BINS[0]


def bin_name(b):
    return f'{b[0]}+' if b[1] is None else (f'{b[0]}' if b[0] == b[1] else f'{b[0]}-{b[1]}')


def lost_by_function_length(samples):
    """{label name: {bin name: {functions, selected, lost, lost_share, fully_lost}}} from the saved tensors."""
    out = {name: {bin_name(b): {'functions': 0, 'selected': 0, 'lost': 0, 'fully_lost': 0} for b in LENGTH_BINS}
           for name in LABELS.values()}
    for s in samples:
        for n, matched in s['functions']:
            cell = out[LABELS[s['label']]][bin_name(length_bin(n))]
            cell['functions'] += 1
            cell['selected'] += n
            cell['lost'] += n - matched
            cell['fully_lost'] += int(matched == 0)
    for cells in out.values():
        for cell in cells.values():
            cell['lost_share'] = cell['lost'] / cell['selected'] if cell['selected'] else float('nan')
    return out


# ------------------------------------------------------------------------------------------------------------ manifests

def read_gap_report(path):
    """The counts of one manifest's `mismatch_report` and the reasons of its retained examples."""
    report = json.loads(Path(path).read_text(encoding='utf-8')).get('mismatch_report') or {}
    asm = report.get('missing_asm_instructions', [])
    return {'missing_asm_count': report.get('missing_asm_count', 0), 'missing_json_count': report.get('missing_json_count', 0),
            'duplicate_count': report.get('duplicate_count', 0), 'address_format_count': report.get('address_format_count', 0),
            'asm_reasons': dict(Counter(x.get('reason', '?') for x in asm)),
            'json_reasons': dict(Counter(x.get('reason', '?') for x in report.get('missing_json_instructions', [])))}


def gap_reports(build, ids, cache=None):
    cached = json.loads(Path(cache).read_text(encoding='utf-8')) if cache and Path(cache).is_file() else {}
    out, dirty = {}, False
    for sample_id in ids:
        if sample_id not in cached:
            path = Path(build) / f'{sample_id}.manifest.json'
            if not path.is_file():
                continue
            cached[sample_id] = read_gap_report(path)
            dirty = True
        out[sample_id] = cached[sample_id]
    if cache and dirty:
        Path(cache).write_text(json.dumps(cached), encoding='utf-8')
    return out


def gap_table(samples, gaps):
    """{label name: {n, selected, missing_asm, missing_json, duplicates, address problems, example reasons}}."""
    out = {}
    for label, name in LABELS.items():
        rows = [(s, gaps[s['sample_id']]) for s in samples if s['label'] == label and s['sample_id'] in gaps]
        reasons = Counter()
        for _, g in rows:
            reasons.update(g['asm_reasons'])
        out[name] = {'n': len(rows), 'selected': sum(s['n'] for s, _ in rows),
                     'missing_asm': sum(g['missing_asm_count'] for _, g in rows),
                     'missing_json': sum(g['missing_json_count'] for _, g in rows),
                     'duplicates': sum(g['duplicate_count'] for _, g in rows),
                     'address_problems': sum(g['address_format_count'] for _, g in rows),
                     'example_reasons': dict(reasons)}
    return out


# ---------------------------------------------------------------------------------------------------------------- replay

def pair_kind(listing, cfg):
    """Sort a (listing mnemonic, CFG mnemonic) disagreement."""
    if listing in RET and cfg in RET:
        return 'retn/ret'
    if listing in PREFIXES or cfg in PREFIXES:
        return 'prefix'
    if listing in FILLERS or cfg in FILLERS:
        return 'filler'
    if listing in DATA_AS_CODE or cfg in DATA_AS_CODE:
        return 'data as code'
    return 'other'


def label_kind(token):
    """Kind of a code label (a first token ending in ':') that her address rule does not read, else None."""
    name = token.rstrip(':')
    if LABEL_PREFIX.fullmatch(name):
        return None
    lowered = name.lower()
    for prefix in ('def_', 'nullsub_', 'j_'):
        if lowered.startswith(prefix):
            return prefix
    return 'named'


def canon(name):
    """A function name as the listing spells it: IDA turns every character that is no identifier character into '_'
    (the Go name `unicode.utf8.DecodeRune` is `unicode_utf8_DecodeRune` in the listing)."""
    return re.sub(r'[^0-9a-z_?@$]', '_', name.lower())


def collapsed_names(entries):
    """Canonical function names of the COLLAPSED FUNCTION markers (the captured text ends with '. PRESS CTRL-NUMPAD+ TO EXPAND')."""
    return {canon(e.split('. PRESS')[0].strip()) for e in entries}


def scan_labels(lines, imagebase, function_rvas, va_of):
    """Census of the listing's label lines, no instruction parsing: {anchor RVA: {kind: count}} of code labels (a first token
    ending in ':') that carry no address, the `endp` lines, and the canonical names of the `name proc` lines whose name is no
    CFG function name (those procedures set no anchor). `va_of` is the teammates' `_va_from_token`; the anchor tracks their
    rule (a label with an address, or a `name proc` of a known function)."""
    kinds, endp, anchor, renamed = defaultdict(Counter), 0, None, set()
    for line in lines:
        code = line.split(';', 1)[0].strip()
        if not code or code.startswith('.'):
            continue
        parts = code.split()
        first = parts[0]
        if len(parts) > 1 and parts[1].lower() == 'proc' and first in function_rvas:
            anchor = function_rvas[first]
            continue
        if len(parts) > 1 and parts[1].lower() == 'proc':
            renamed.add(canon(first))
        if len(parts) > 1 and parts[1].lower() == 'endp':
            endp += 1
        va = va_of(first)
        if va is not None:
            anchor = va - imagebase if va >= imagebase else va
        elif first.endswith(':') and len(first) > 1:
            kind = label_kind(first)
            if kind:
                kinds[anchor][kind] += 1
    return kinds, endp, renamed


def selected_instructions(cfg, keys):
    return [(key, index, ins) for key in keys for index, ins in enumerate(cfg.blocks[key].instructions)]


def run_alignment(pp, source_kind, cfg, keys, asm_path):
    """Run the pinned alignment of `source_kind` ('whole' = align_cfg_to_asm over the whole CFG, 'sampled' =
    align_sampled_blocks over the keys) and return (parsed asm items, collapsed names, lines, report).
    The parse is `parse_asm_instructions` in both cases; `lines` is the listing as a list."""
    reader = getattr(pp, 'iter_text_lines', None) or pp.read_text_lines
    lines = list(reader(asm_path))
    problems = []
    parsed, collapsed = pp.parse_asm_instructions(lines, cfg.imagebase, cfg.function_rvas, problems)
    if source_kind == 'whole':
        report = pp.align_cfg_to_asm(cfg, parsed, collapsed)
    else:
        report = pp.align_sampled_blocks(cfg, keys, lines, [])
    return parsed, collapsed, lines, report


def matched_set(cfg, keys):
    return frozenset((key, index) for key, index, ins in selected_instructions(cfg, keys) if ins.matched)


def replay_sample(sample_id, keys, build_src, cfg_path, asm_path, imports=None):
    """Replay one sample; `imports(source)` returns the pinned `preprocess_full` of a source (default: import_team with the
    example limit lifted). Returns the per-sample diagnosis dict (see the module doc, part 3)."""
    if imports is None:
        imports = _import_pinned
    keys = [tuple(k) for k in keys]
    ref_src = 'gnn-331ce0b'
    pp = imports(ref_src)
    cfg = pp.load_cfg(cfg_path, file_id=sample_id)
    parsed, collapsed, lines, report = run_alignment(pp, 'whole', cfg, keys, asm_path)
    ref_matched = matched_set(cfg, keys)
    sel = selected_instructions(cfg, keys)
    grouped = Counter(ins.rva for block in cfg.blocks.values() for ins in block.instructions)
    anchors = sorted({p.anchor_rva for p in parsed if p.anchor_rva is not None})
    lines_at = defaultdict(list)
    for p in parsed:
        if p.anchor_rva is not None:
            lines_at[p.anchor_rva].append(p)
    rows_at = defaultdict(list)
    for block in cfg.blocks.values():
        for ins in block.instructions:
            if grouped[ins.rva] == 1:
                rows_at[pp._anchor_bucket(anchors, ins.rva)].append(ins)
    for rows in rows_at.values():
        rows.sort(key=lambda i: (i.rva, i.function_id, i.block_id, i.index_in_block))
    first_bad = {a: next((n for n, r in enumerate(rows) if not r.matched), len(rows)) for a, rows in rows_at.items()}
    entry = {}
    for block in cfg.blocks.values():
        if not block.external:
            entry[block.function_id] = min(entry.get(block.function_id, block.start_rva), block.start_rva)
    for block in cfg.blocks.values():
        entry.setdefault(block.function_id, block.start_rva)
    keyset = set(keys)
    anchor_set = set(anchors)
    coll = collapsed_names(collapsed)
    label_kinds, endp_lines, renamed_procs = scan_labels(lines, cfg.imagebase, cfg.function_rvas, pp.tok._va_from_token)

    own_matched = ref_matched
    if build_src != ref_src:
        pp2 = imports(build_src)
        cfg2 = pp2.load_cfg(cfg_path, file_id=sample_id)
        run_alignment(pp2, 'sampled', cfg2, keys, asm_path)
        own_matched = matched_set(cfg2, keys)

    categories, subcategories, spans = Counter(), Counter(), {}
    pairs, events = defaultdict(Counter), set()
    bucket_of = {id(r): a for a, rows in rows_at.items() for r in rows}
    for key, index, ins in sel:
        a = bucket_of.get(id(ins))
        if a is not None:
            cell = spans.setdefault(a, {'cfg_rows': len(rows_at[a]), 'listing_lines': len(lines_at[a]),
                                        'selected': 0, 'lost': 0})
            cell['selected'] += 1
            cell['lost'] += int((key, index) not in own_matched)
        if (key, index) in own_matched:
            continue
        if (key, index) in ref_matched:
            categories['vi'] += 1
            continue
        if grouped[ins.rva] != 1:
            categories['iv'] += 1
            continue
        if a is None:
            categories['i'] += 1
            continue
        name = ins.function_name.lower()
        n_lines, k = len(lines_at[a]), first_bad[a]
        if canon(ins.function_name) in coll:
            categories['iii'] += 1
            subcategories['iii: function collapsed in the listing'] += 1
        elif cfg.function_rvas.get(ins.function_name, entry[ins.function_id]) not in anchor_set:
            if canon(ins.function_name) in renamed_procs:
                categories['v'] += 1
                subcategories['v: procedure is in the listing under another spelling of its name'] += 1
            else:
                categories['iii'] += 1
                subcategories['iii: function entry is no listing label'] += 1
        elif n_lines == 0:
            categories['iii'] += 1
            subcategories['iii: label span without listing lines'] += 1
        elif k < min(n_lines, len(rows_at[a])):
            pair = (lines_at[a][k].mnemonic, pp.tok._mnemonic(rows_at[a][k].mnemonic).lower())
            pairs[pair]['lost'] += 1
            if a not in events:
                events.add(a)
                pairs[pair]['events'] += 1
            if label_kinds.get(a):
                categories['v'] += 1
                subcategories['v: ' + ', '.join(sorted(label_kinds[a]))] += 1
            else:
                categories['ii'] += 1
            subcategories['disagreement ' + pair_kind(*pair)] += 1
        else:
            categories['vii'] += 1
            subcategories['vii: listing lines of the span ran out'] += 1
    return {
        'sample_id': sample_id, 'build_source': build_src, 'n_selected': len(sel), 'matched_build': len(own_matched),
        'matched_reference': len(ref_matched), 'categories': dict(categories), 'subcategories': dict(subcategories),
        'pairs': {f'{l} | {c}': dict(v) for (l, c), v in pairs.items()},
        'spans': list(spans.values()),
        'listing': {'code_lines': len(parsed), 'lines_without_label': sum(1 for p in parsed if p.anchor_rva is None),
                    'anchors': len(anchors), 'collapsed_markers': len(collapsed), 'endp_lines': endp_lines,
                    'renamed_procs': len(renamed_procs), 'unanchored_labels': dict(sum((Counter(c) for c in label_kinds.values()), Counter())),
                    'cfg_functions': len(entry), 'cfg_functions_with_label': sum(1 for e in entry.values() if e in anchor_set),
                    'duplicate_rvas': sum(1 for n in grouped.values() if n > 1)},
        'excerpt': excerpt(lines, collapsed, sel, own_matched, coll),
    }


def excerpt(lines, collapsed, sel, matched, coll, width=8):
    """At most `width` listing lines (text only) that show a lost function: the COLLAPSED FUNCTION marker of the first function
    with an unmatched selected instruction that has one (with the lines around it), else the lines that name the first such
    function, else only its name."""
    lost, seen = [], set()
    for key, index, ins in sel:
        if (key, index) not in matched and ins.function_name not in seen:
            seen.add(ins.function_name)
            lost.append(ins.function_name)
    if not lost:
        return {'function': None, 'kind': 'none', 'lines': []}
    markers = {}
    for n, line in enumerate(lines):
        found = COLLAPSED.search(line)
        if found:
            markers.setdefault(canon(found.group(1).split('. PRESS')[0].strip()), n)
    for name in lost:
        n = markers.get(canon(name))
        if n is not None:
            lo = max(0, n - 3)
            return {'function': name, 'kind': 'collapsed', 'lines': [x.rstrip()[:110] for x in lines[lo:lo + width]]}
    for name in lost:
        for n, line in enumerate(lines):
            if line.lstrip().startswith(name):
                return {'function': name, 'kind': 'named', 'lines': [x.rstrip()[:110] for x in lines[n:n + width]]}
    return {'function': lost[0], 'kind': 'absent', 'lines': []}


def _import_pinned(source):
    from experiments.team_pipelines.common import import_team
    pp = import_team('preprocess_full', source=source)
    pp.MISMATCH_EXAMPLE_LIMIT = 10 ** 9     # in memory only: keep every gap of the report
    return pp


def pick_replay(samples, worst, goodware, partial=0, seed=0):
    """The `worst` lowest-match samples, the `goodware` lowest-match goodware samples, then `partial` random samples per label
    among those that lose some but not all (0 < rate < 0.999; random.Random(seed))."""
    ranked = sorted(samples, key=lambda s: (s['rate'], -s['n'], s['sample_id']))
    chosen = [s['sample_id'] for s in ranked[:worst]]
    chosen += [s['sample_id'] for s in ranked if s['label'] == 0 and s['sample_id'] not in chosen][:goodware]
    rng = random.Random(seed)
    for label in (1, 0):
        pool = [s['sample_id'] for s in ranked if s['label'] == label and 0 < s['rate'] < 0.999 and s['sample_id'] not in chosen]
        chosen += sorted(rng.sample(pool, min(partial, len(pool))))
    return chosen


def summarize_replays(replays, labels):
    """Counts per category and per label, the pair histogram, and the lost share by label-span length."""
    by_label = {name: Counter() for name in LABELS.values()}
    sub = {name: Counter() for name in LABELS.values()}
    pairs = defaultdict(Counter)
    span_cells = defaultdict(lambda: {'spans': 0, 'selected': 0, 'lost': 0})
    for r in replays:
        name = LABELS[labels[r['sample_id']]]
        by_label[name]['selected'] += r['n_selected']
        by_label[name]['lost'] += r['n_selected'] - r['matched_build']
        by_label[name].update(r['categories'])
        sub[name].update(r['subcategories'])
        for pair, v in r['pairs'].items():
            pairs[pair].update(v)
        for sp in r['spans']:
            cell = span_cells[bin_name(length_bin(sp['cfg_rows']))]
            cell['spans'] += 1
            cell['selected'] += sp['selected']
            cell['lost'] += sp['lost']
    for cell in span_cells.values():
        cell['lost_share'] = cell['lost'] / cell['selected'] if cell['selected'] else float('nan')
    kinds = Counter()
    for pair, v in pairs.items():
        listing, cfg = (x.strip() for x in pair.split('|'))
        kinds[pair_kind(listing, cfg)] += v['lost']
    return {'by_label': {k: dict(v) for k, v in by_label.items()}, 'subcategories': {k: dict(v) for k, v in sub.items()},
            'pairs': {k: dict(v) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]['lost'])}, 'pair_kinds': dict(kinds),
            'span_length': {bin_name(b): span_cells.get(bin_name(b), {'spans': 0, 'selected': 0, 'lost': 0,
                                                                       'lost_share': float('nan')}) for b in LENGTH_BINS}}


def replay_all(build, samples, ids, source, max_mb, imports=None, log=print, paths_of=None, cache_dir=None):
    """Replay `ids`; a listing above `max_mb` (compressed) is skipped. `cache_dir` keeps one JSON per replayed sample, so an
    interrupted run resumes."""
    if paths_of is None:
        from experiments.team_pipelines.common import paths_for as paths_of
    by_id = {s['sample_id']: s for s in samples}
    skipped, done, todo = [], {}, []
    for sample_id in ids:
        _, asm_path = paths_of(sample_id)
        if Path(asm_path).stat().st_size > max_mb * 1e6:
            skipped.append(sample_id)
            continue
        entry = Path(cache_dir) / f'{source}__{sample_id}.json' if cache_dir else None
        if entry is not None and entry.is_file():
            done[sample_id] = json.loads(entry.read_text(encoding='utf-8'))
        else:
            todo.append(sample_id)

    def keep(r):
        done[r['sample_id']] = r
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            (Path(cache_dir) / f"{source}__{r['sample_id']}.json").write_text(json.dumps(r), encoding='utf-8')
        s = by_id[r['sample_id']]
        log(f"replay {len(done)}/{len(ids) - len(skipped)} {r['sample_id'][:12]} {s['family']} rate {s['rate']:.3f} "
            f"{r['categories']}", flush=True)

    for sample_id in todo:
        cfg_path, asm_path = paths_of(sample_id)
        p = torch.load(Path(build) / f'{sample_id}.pt', map_location='cpu', weights_only=False)
        keys = [(k['function_id'], k['block_id']) for k in p['block_keys']]
        keep(replay_sample(sample_id, keys, source, cfg_path, asm_path, imports=imports))
        gc.collect()
    out = []
    for sample_id in ids:
        if sample_id in done:
            r = done[sample_id]
            r['consistent'] = r['matched_build'] == by_id[sample_id]['matched']
            r['family'], r['label'] = by_id[sample_id]['family'], by_id[sample_id]['label']
            out.append(r)
    return out, skipped


# --------------------------------------------------------------------------------------------------------------- output

def pct(x):
    return 'n/a' if x != x else f'{100 * x:.1f}%'


def format_report(result):
    L = [f"match diagnosis of {result['build']} ({result['n_samples']} samples, source {result['source']})", '']
    L += ['Match rate per label', '', '| label | binaries | mean match | below 50% | below 90% | exactly 0 |', '|---|---|---|---|---|---|']
    for name, s in result['labels'].items():
        L.append(f"| {name} | {s['n']} | {s['mean_match']:.3f} | {s['n_below_50']} ({pct(s['share_below_50'])}) | "
                 f"{s['n_below_90']} ({pct(s['share_below_90'])}) | {pct(s['share_zero'])} |")
    a = result['auc']
    L += ['', f"AUC of 1 - match rate as a ransomware score: {a['auc']:.3f}, 95% interval over {a['n_clusters']} clusters "
              f"[{a['lo']:.3f}, {a['hi']:.3f}]", '']
    L += ['Per family (all families, lowest mean first)', '', '| family | binaries | mean match | below 50% | below 90% |',
          '|---|---|---|---|---|']
    for fam, s in sorted(result['families'].items(), key=lambda kv: (kv[1]['mean_match'], kv[0])):
        L.append(f"| {fam} | {s['n']} | {s['mean_match']:.3f} | {pct(s['share_below_50'])} | {pct(s['share_below_90'])} |")
    L += ['', 'Ten worst families (mean match rate, families of at least 3 binaries)', '',
          '| family | binaries | mean match | below 50% | below 90% |', '|---|---|---|---|---|']
    for fam, s in result['worst_families']:
        L.append(f"| {fam} | {s['n']} | {s['mean_match']:.3f} | {pct(s['share_below_50'])} | {pct(s['share_below_90'])} |")
    L += ['', 'Lost share by function length (selected instructions of one function; saved tensors)', '',
          '| label | length | functions | selected | lost | lost share | fully lost functions |', '|---|---|---|---|---|---|---|']
    for name, cells in result['function_length'].items():
        for b, c in cells.items():
            L.append(f"| {name} | {b} | {c['functions']} | {c['selected']} | {c['lost']} | {pct(c['lost_share'])} | {c['fully_lost']} |")
    if result.get('gaps'):
        L += ['', 'Gap reports of the manifests (scope: whole CFG for gnn-331ce0b, sampled blocks for main-c7899f0)', '',
              '| label | manifests | selected instr. | CFG instr. without listing line | listing lines without CFG instr. | '
              'duplicate RVAs | address problems |', '|---|---|---|---|---|---|---|']
        for name, g in result['gaps'].items():
            L.append(f"| {name} | {g['n']} | {g['selected']} | {g['missing_asm']} | {g['missing_json']} | {g['duplicates']} | "
                     f"{g['address_problems']} |")
        L += ['', 'Reasons of the retained examples (at most 32 per manifest): ' +
              '; '.join(f"{name}: {g['example_reasons']}" for name, g in result['gaps'].items())]
    rp = result.get('replay')
    if rp:
        L += ['', f"Replay of {rp['n_replayed']} samples ({rp['n_consistent']} reproduce the build's matched count; "
                  f"{len(rp['skipped'])} skipped for listing size)", '']
        L += ['| cause | ' + ' | '.join(LABELS.values()) + ' |', '|---|' + '---|' * len(LABELS)]
        for code, text in CATEGORIES:
            L.append(f'| ({code}) {text} | ' + ' | '.join(str(rp['by_label'][n].get(code, 0)) for n in LABELS.values()) + ' |')
        L.append('| selected instructions replayed | ' + ' | '.join(str(rp['by_label'][n].get('selected', 0)) for n in LABELS.values()) + ' |')
        L.append('| unmatched | ' + ' | '.join(str(rp['by_label'][n].get('lost', 0)) for n in LABELS.values()) + ' |')
        for name in LABELS.values():
            L += ['', f'Sub-causes, {name}: ' + '; '.join(f'{k} {v}' for k, v in sorted(rp['subcategories'][name].items(),
                                                                                       key=lambda kv: -kv[1]))]
        L += ['', 'Mnemonic disagreements (listing | CFG), instructions lost after them', '']
        L += [f"{v['lost']:>8} lost, {v['events']:>4} spans  {pair}" for pair, v in list(rp['pairs'].items())[:20]]
        L += ['', 'By kind: ' + '; '.join(f'{k} {v}' for k, v in rp['pair_kinds'].items())]
        L += ['', 'Lost share by label-span length (CFG instructions between two listing labels)', '',
              '| span length | spans | selected | lost | lost share |', '|---|---|---|---|---|']
        for b, c in rp['span_length'].items():
            L.append(f"| {b} | {c['spans']} | {c['selected']} | {c['lost']} | {pct(c['lost_share'])} |")
    return '\n'.join(L)


def diagnose(build, source=None, ids=None, replay_worst=30, replay_goodware=10, replay_partial=0, max_listing_mb=60, cache=None,
             manifests=True, min_family_n=3, resamples=1000, imports=None, log=print, paths_of=None, replay_cache=None):
    build = Path(build)
    samples = load_samples(build, ids)
    source = source or build_source(build)
    families = family_table(samples)
    result = {'build': str(build), 'source': source, 'n_samples': len(samples), 'labels': label_table(samples),
              'auc': auc_of_loss(samples, resamples), 'families': families,
              'worst_families': worst_families(families, 10, min_family_n),
              'function_length': lost_by_function_length(samples),
              'samples': [{k: v for k, v in s.items() if k != 'functions'} for s in samples]}
    if manifests:
        result['gaps'] = gap_table(samples, gap_reports(build, [s['sample_id'] for s in samples], cache))
    if replay_worst or replay_goodware or replay_partial:
        chosen = pick_replay(samples, replay_worst, replay_goodware, replay_partial)
        replays, skipped = replay_all(build, samples, chosen, source, max_listing_mb, imports, log, paths_of, replay_cache)
        labels = {s['sample_id']: s['label'] for s in samples}
        rp = summarize_replays(replays, labels)
        rp.update(n_replayed=len(replays), n_consistent=sum(r['consistent'] for r in replays), skipped=skipped, replays=replays)
        result['replay'] = rp
    return result


def write_csv(path, samples):
    fields = ('sample_id', 'label', 'family', 'arch', 'cluster', 'n', 'matched', 'rate')
    lines = [','.join(fields)] + [','.join(str(s[f]) for f in fields) for s in sorted(samples, key=lambda s: s['rate'])]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--source', help='teammate source the build used (default: from its settings.json)')
    ap.add_argument('--among', type=Path, help='only the ids that this build also holds')
    ap.add_argument('--replay-worst', type=int, default=30, help='replay the N lowest-match samples (0: no replay)')
    ap.add_argument('--replay-goodware', type=int, default=10, help='and the N lowest-match goodware samples')
    ap.add_argument('--replay-partial', type=int, default=20, help='and N random samples per label that lose some but not all')
    ap.add_argument('--max-listing-mb', type=float, default=60, help='skip the replay of listings larger than this (gz size)')
    ap.add_argument('--no-manifests', action='store_true', help='skip the gap reports of the manifests (slow: GBs of JSON)')
    ap.add_argument('--cache', type=Path, help='JSON cache of the gap reports (outside the build directory)')
    ap.add_argument('--replay-cache', type=Path, help='directory of one JSON per replayed sample (resumes an interrupted run)')
    ap.add_argument('--min-family-n', type=int, default=3)
    ap.add_argument('--resamples', type=int, default=1000)
    ap.add_argument('--json', type=Path)
    ap.add_argument('--markdown', type=Path)
    ap.add_argument('--csv', type=Path, help='one row per sample: id, label, family, arch, cluster, selected, matched, rate')
    args = ap.parse_args(argv)
    build = resolve_dir(args.build)
    ids = None
    if args.among:
        ids = sorted(set(build_ids(build)) & set(build_ids(resolve_dir(args.among))))
    result = diagnose(build, args.source, ids, args.replay_worst, args.replay_goodware, args.replay_partial, args.max_listing_mb, args.cache,
                      not args.no_manifests, args.min_family_n, args.resamples, replay_cache=args.replay_cache)
    text = format_report(result)
    print(text)
    if args.markdown:
        args.markdown.write_text(text, encoding='utf-8')
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    if args.csv:
        write_csv(args.csv, result['samples'])
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Step 24: `match_diagnosis.py` and `match_acceptance.py` on synthetic builds (tensors, manifests, a fake pinned module);
one integration test on a real small binary when the teammates' worktrees and the data are there."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.team_pipelines import match_acceptance as acc, match_diagnosis as md
from test_team_pipelines import needs_real_data, needs_worktrees, pinned_team_modules  # noqa: F401


# ------------------------------------------------------------------------------------------------ synthetic build

def write_sample(build, sample_id, label, family, rates_by_function, matched_total=None):
    """One saved sample: `rates_by_function` is [(instructions, matched)] per function (one block per function)."""
    insn_to_block, matched = [], []
    for block, (n, m) in enumerate(rates_by_function):
        insn_to_block += [block] * n
        matched += [True] * m + [False] * (n - m)
    payload = {'label': label, 'dataset_metadata': {'sample_id': sample_id, 'label': label, 'arch': 'x86', 'family': family},
               'instruction_matched': torch.tensor(matched, dtype=torch.bool),
               'insn_to_block': torch.tensor(insn_to_block, dtype=torch.long),
               'block_keys': [{'function_id': 100 + b, 'block_id': 0} for b in range(len(rates_by_function))]}
    torch.save(payload, Path(build) / f'{sample_id}.pt')


def write_manifest(build, sample_id, missing_asm, missing_json=0, duplicates=0, reasons=()):
    report = {'missing_asm_count': missing_asm, 'missing_json_count': missing_json, 'duplicate_count': duplicates,
              'address_format_count': 0, 'missing_json_instructions': [{'reason': 'no_cfg_instruction_in_address_span'}],
              'missing_asm_instructions': [{'reason': r} for r in reasons]}
    (Path(build) / f'{sample_id}.manifest.json').write_text(json.dumps({'mismatch_report': report}), encoding='utf-8')


def synthetic_build(tmp_path, leak=True, source='gnn-331ce0b'):
    """12 ransomware (3 families) and 8 goodware; with `leak` ransomware matches badly and goodware well."""
    build = tmp_path / 'build'
    build.mkdir(parents=True)
    rng = np.random.default_rng(0)
    ids = []
    for i in range(20):
        label = int(i < 12)
        family = ['alpha', 'beta', 'gamma'][i % 3] if label else 'goodware'
        sample_id = f'{"r" if label else "g"}{i:03d}'
        good = True if not leak else rng.random() < (0.3 if label else 0.95)
        funcs = [(40, 40 if good else 0), (10, 10 if good else 5), (5, 5)]
        write_sample(build, sample_id, label, family, funcs)
        write_manifest(build, sample_id, missing_asm=0 if good else 50, reasons=[] if good else ['collapsed_function'] * 3)
        ids.append(sample_id)
    (build / 'checkpoint.json').write_text(json.dumps({'settings': {'sample': 'signal', 'num_blocks': 500, 'seed': 0,
                                                                  'source': source}, 'done': ids}), encoding='utf-8')
    (build / 'settings.json').write_text(json.dumps({'sample': 'signal', 'num_blocks': 500, 'seed': 0, 'source': source,
                                                     'excluded': {}}), encoding='utf-8')
    return build, ids


def test_cluster_rule_matches_stats_py():
    assert md.cluster_of(1, 'alpha', 'r001') == 'fam:alpha' and md.cluster_of(0, 'goodware', 'g001') == 'id:g001'


def test_load_samples_joins_label_family_arch_and_function_sizes(tmp_path):
    build = tmp_path
    write_sample(build, 'r001', 1, 'alpha', [(4, 1), (6, 6)])
    s = md.load_samples(build, ['r001'])[0]
    assert (s['label'], s['family'], s['arch'], s['n'], s['matched'], s['cluster']) == (1, 'alpha', 'x86', 10, 7, 'fam:alpha')
    assert s['rate'] == pytest.approx(0.7) and s['functions'] == [(4, 1), (6, 6)]


def test_label_and_family_tables_and_the_worst_families(tmp_path):
    build, ids = synthetic_build(tmp_path)
    samples = md.load_samples(build)
    table = md.label_table(samples)
    assert table['ransomware']['n'] == 12 and table['goodware']['n'] == 8
    assert table['ransomware']['mean_match'] < table['goodware']['mean_match']
    assert table['ransomware']['share_below_90'] > table['goodware']['share_below_90']
    fam = md.family_table(samples)
    assert set(fam) == {'alpha', 'beta', 'gamma', 'goodware'} and fam['alpha']['n'] == 4
    worst = md.worst_families(fam, 2, min_n=3)
    assert [f for f, _ in worst][0] != 'goodware' and len(worst) == 2
    assert md.worst_families({'solo': fam['alpha'] | {'n': 1}}, 5, min_n=3) == []


def test_auc_of_loss_is_one_when_the_match_rate_separates_the_labels_and_half_when_it_does_not(tmp_path):
    samples = [{'label': 1, 'rate': 0.1 * (i % 5), 'cluster': f'fam:f{i % 4}'} for i in range(20)]
    samples += [{'label': 0, 'rate': 0.95 + 0.01 * (i % 5), 'cluster': f'id:g{i}'} for i in range(20)]
    leak = md.auc_of_loss(samples, resamples=200)
    assert leak['auc'] == 1.0 and leak['lo'] == 1.0 and leak['hi'] == 1.0 and leak['n_clusters'] == 24
    rng = np.random.default_rng(1)
    flat = [{'label': i % 2, 'rate': float(rng.random()), 'cluster': f'id:{i}'} for i in range(400)]
    none = md.auc_of_loss(flat, resamples=300)
    assert 0.4 < none['auc'] < 0.6 and none['lo'] < none['auc'] < none['hi'] and none['hi'] - none['lo'] < 0.2
    assert md.auc_of_loss(flat, resamples=300) == none                          # seeded


def test_auc_bootstrap_draws_clusters_not_samples():
    # two ransomware families of 50 binaries that match perfectly apart from one family: resampling clusters moves the
    # interval a lot more than resampling the 100 binaries would
    samples = [{'label': 1, 'rate': 0.2, 'cluster': 'fam:a'} for _ in range(50)]
    samples += [{'label': 1, 'rate': 0.99, 'cluster': 'fam:b'} for _ in range(50)]
    samples += [{'label': 0, 'rate': 0.97, 'cluster': f'id:g{i}'} for i in range(100)]
    out = md.auc_of_loss(samples, resamples=400)
    assert out['n_clusters'] == 102 and out['hi'] - out['lo'] > 0.1


def test_lost_by_function_length_bins_and_fully_lost_functions(tmp_path):
    build, _ = synthetic_build(tmp_path)
    samples = md.load_samples(build)
    cells = md.lost_by_function_length(samples)
    assert cells['ransomware']['1000+']['functions'] == 0
    assert cells['ransomware']['10-49']['functions'] == 24 and cells['ransomware']['1-9']['functions'] == 12
    total = sum(c['selected'] for c in cells['ransomware'].values())
    assert total == sum(s['n'] for s in samples if s['label'] == 1)
    assert cells['ransomware']['10-49']['fully_lost'] > 0 and cells['goodware']['10-49']['lost_share'] < \
        cells['ransomware']['10-49']['lost_share']


def test_gap_reports_cache_and_table(tmp_path):
    build, ids = synthetic_build(tmp_path)
    cache = tmp_path / 'cache' / 'gaps.json'
    cache.parent.mkdir()
    gaps = md.gap_reports(build, ids, cache)
    assert cache.is_file() and set(gaps) == set(ids)
    (build / f'{ids[0]}.manifest.json').unlink()                       # the cache answers without the manifest
    assert md.gap_reports(build, ids, cache) == gaps
    table = md.gap_table(md.load_samples(build), gaps)
    assert table['ransomware']['missing_asm'] > table['goodware']['missing_asm'] == 0
    assert set(table['ransomware']['example_reasons']) <= {'collapsed_function'}
    assert md.read_gap_report(build / f'{ids[1]}.manifest.json')['missing_json_count'] == 0


def test_pair_kinds_label_kinds_and_collapsed_names():
    assert md.pair_kind('retn', 'ret') == 'retn/ret' and md.pair_kind('rep', 'movsd') == 'prefix'
    assert md.pair_kind('nop', 'lea') == 'filler' and md.pair_kind('mov', 'add') == 'data as code'
    assert md.pair_kind('mov', 'lea') == 'other'
    assert md.label_kind('loc_401000:') is None and md.label_kind('locret_401000:') is None
    assert [md.label_kind(t) for t in ('def_401000:', 'nullsub_1:', 'j_memcpy:', '_main:')] == ['def_', 'nullsub_', 'j_', 'named']
    assert md.collapsed_names(['__input_l. PRESS CTRL-NUMPAD+ TO EXPAND', 'start']) == {'__input_l', 'start'}
    assert md.canon('unicode.utf8.DecodeRune') == md.canon('unicode_utf8_DecodeRune') == 'unicode_utf8_decoderune'
    assert md.collapsed_names(['unicode_utf8_DecodeRune. PRESS CTRL-NUMPAD+ TO EXPAND']) == {md.canon('unicode.utf8.DecodeRune')}


# ------------------------------------------------------------------------------------------------------- the replay

class Ins:
    def __init__(self, rva, mnemonic, function_id, function_name, block_id, index):
        self.rva, self.mnemonic, self.function_id, self.function_name = rva, mnemonic, function_id, function_name
        self.block_id, self.index_in_block, self.matched = block_id, index, False

    @property
    def block_key(self):
        return (self.function_id, self.block_id)


class FakePinned:
    """The part of a pinned `preprocess_full` the replay calls, with the first source's pairing rule; with
    `lose` the sampled alignment of the second source leaves those RVAs unmatched (the `endp` reset)."""
    IMAGEBASE = 0x400000

    def __init__(self, listing, functions, lose=()):
        self.tok = SimpleNamespace(_mnemonic=lambda m: m.upper().split()[-1],
                                   _va_from_token=lambda t: int(t.rstrip(':')[4:], 16) if t.startswith('sub_') else None)
        self.listing, self.functions, self.lose = listing, functions, set(lose)
        self.MISMATCH_EXAMPLE_LIMIT = 10 ** 9

    def load_cfg(self, path, file_id=None):
        blocks = {}
        for fid, (name, rva, rows) in enumerate(self.functions, 1):
            for bid, (start, mns) in enumerate(rows):
                blocks[(fid, bid)] = SimpleNamespace(
                    function_id=fid, block_id=bid, function_name=name, start_rva=start, external=False,
                    instructions=[Ins(start + i, mn, fid, name, bid, i) for i, mn in enumerate(mns)])
        return SimpleNamespace(imagebase=self.IMAGEBASE, blocks=blocks, function_rvas={n: r for n, r, _ in self.functions})

    def read_text_lines(self, path):
        return list(self.listing)

    iter_text_lines = read_text_lines

    def parse_asm_instructions(self, lines, imagebase, function_rvas, problems):
        parsed, collapsed, anchor = [], [], None
        for line in lines:
            if 'COLLAPSED FUNCTION' in line:
                collapsed.append(line.split('COLLAPSED FUNCTION')[1].strip(' ]') + '. PRESS CTRL-NUMPAD+ TO EXPAND')
            parts = line.split()
            if len(parts) > 1 and parts[1] == 'proc':
                anchor = function_rvas.get(parts[0], anchor)
            elif line.startswith('    '):
                parsed.append(SimpleNamespace(anchor_rva=anchor, mnemonic=parts[0], raw_text=line.strip()))
        return parsed, collapsed

    @staticmethod
    def _anchor_bucket(anchors, rva):
        below = [a for a in anchors if a <= rva]
        return below[-1] if below else None

    def align_cfg_to_asm(self, cfg, parsed, collapsed):
        grouped = Counter(i.rva for b in cfg.blocks.values() for i in b.instructions)
        anchors = sorted({p.anchor_rva for p in parsed if p.anchor_rva is not None})
        rows = {}
        for b in cfg.blocks.values():
            for i in b.instructions:
                if grouped[i.rva] == 1 and self._anchor_bucket(anchors, i.rva) is not None:
                    rows.setdefault(self._anchor_bucket(anchors, i.rva), []).append(i)
        for a, js in rows.items():
            js.sort(key=lambda i: (i.rva, i.function_id, i.block_id, i.index_in_block))
            ls = [p for p in parsed if p.anchor_rva == a]
            for n, (l, j) in enumerate(zip(ls, js)):
                if l.mnemonic != self.tok._mnemonic(j.mnemonic).lower():
                    break
                j.matched = True

    def align_sampled_blocks(self, cfg, keys, lines, problems):
        parsed, collapsed = self.parse_asm_instructions(list(lines), cfg.imagebase, cfg.function_rvas, [])
        self.align_cfg_to_asm(cfg, parsed, collapsed)
        for b in cfg.blocks.values():
            for i in b.instructions:
                if i.rva in self.lose:
                    i.matched = False


LISTING = [
    'sub_400064 proc near', '    push ebp', '    mov ebp, esp', '    call foo', '    retn', 'sub_400064 endp',
    '; [00000012 BYTES: COLLAPSED FUNCTION collapsed_fn]',
    'sub_4000C8 proc near', '    push ebp', '    nop', '    mov eax, 1', '    retn', 'sub_4000C8 endp',
    'sub_40012C proc near', 'def_4001A0:', '    push ebp', '    call bar', '    mov eax, 2', 'sub_40012C endp',
    'go_fn proc near', '    push ebp', '    mov ebp, esp', 'go_fn endp',
]
FUNCTIONS = [                                    # name, entry rva, [(block start, mnemonics)]
    ('sub_400064', 100, [(100, ['push', 'mov', 'call', 'retn'])]),
    ('collapsed_fn', 150, [(150, ['push', 'mov', 'retn'])]),
    ('sub_4000C8', 200, [(200, ['push', 'lea', 'mov', 'retn'])]),
    ('sub_40012C', 300, [(300, ['push', 'jmp', 'mov'])]),
    ('f5', 400, [(400, ['push', 'mov']), (400, ['push', 'mov'])]),     # the same RVAs twice: duplicates
    ('f6', 10, [(10, ['push'])]),                                      # below the first listing label
    ('go.fn', 500, [(500, ['push', 'mov'])]),                          # listed as `go_fn proc near`: no anchor is set
]
KEYS = [(i, b) for i, (_, _, rows) in enumerate(FUNCTIONS, 1) for b in range(len(rows))]


def fake_imports(lose=()):
    pinned = {'gnn-331ce0b': FakePinned(LISTING, FUNCTIONS), 'main-c7899f0': FakePinned(LISTING, FUNCTIONS, lose)}
    return lambda source: pinned[source]


def test_replay_sorts_every_unmatched_instruction_into_one_cause():
    r = md.replay_sample('s1', KEYS, 'gnn-331ce0b', 'cfg', 'asm', imports=fake_imports())
    assert r['n_selected'] == 4 + 3 + 4 + 3 + 4 + 1 + 2 and r['matched_build'] == r['matched_reference'] == 6
    assert r['categories'] == {'iii': 3, 'ii': 3, 'v': 4, 'iv': 4, 'i': 1}
    assert sum(r['categories'].values()) == r['n_selected'] - r['matched_build']
    assert r['subcategories']['iii: function collapsed in the listing'] == 3
    assert r['subcategories']['disagreement filler'] == 3 and r['subcategories']['v: def_'] == 2
    assert r['subcategories']['v: procedure is in the listing under another spelling of its name'] == 2
    assert r['pairs'] == {'nop | lea': {'lost': 3, 'events': 1}, 'call | jmp': {'lost': 2, 'events': 1}}
    assert r['listing']['unanchored_labels'] == {'def_': 1} and r['listing']['endp_lines'] == 4
    assert r['listing']['collapsed_markers'] == 1 and r['listing']['cfg_functions'] == 7 and r['listing']['renamed_procs'] == 1
    assert r['excerpt']['kind'] == 'collapsed' and r['excerpt']['function'] == 'collapsed_fn' and len(r['excerpt']['lines']) <= 8
    assert any('COLLAPSED FUNCTION collapsed_fn' in x for x in r['excerpt']['lines'])
    lengths = {(sp['cfg_rows'], sp['listing_lines']) for sp in r['spans']}
    assert (7, 4) in lengths                                           # f1 and the collapsed function share one label span


def test_replay_attributes_the_difference_between_the_sources_to_the_endp_reset():
    r = md.replay_sample('s1', KEYS, 'main-c7899f0', 'cfg', 'asm', imports=fake_imports(lose={103}))
    assert r['categories']['vi'] == 1 and r['matched_build'] == 5 and r['matched_reference'] == 6
    assert r['categories']['iii'] == 3                                 # the other causes are the same


def test_summarize_and_format_report(tmp_path):
    r = md.replay_sample('s1', KEYS, 'gnn-331ce0b', 'cfg', 'asm', imports=fake_imports())
    summary = md.summarize_replays([r], {'s1': 1})
    assert summary['by_label']['ransomware']['iii'] == 3 and summary['by_label']['ransomware']['lost'] == 15
    assert summary['pairs']['nop | lea']['lost'] == 3 and summary['pair_kinds']['filler'] == 3
    assert sum(c['selected'] for c in summary['span_length'].values()) == sum(sp['selected'] for sp in r['spans'])


def write_replay_sample(build, matched):
    """Sample 's1' for the fake pinned module: the 19 selected instructions of FUNCTIONS, `matched` of them saved as matched."""
    write_sample(build, 's1', 1, 'alpha', [(19, matched)])
    ckpt = json.loads((build / 'checkpoint.json').read_text(encoding='utf-8'))
    ckpt['done'].append('s1')
    (build / 'checkpoint.json').write_text(json.dumps(ckpt), encoding='utf-8')
    payload = torch.load(build / 's1.pt', weights_only=False)
    payload['block_keys'] = [{'function_id': f, 'block_id': b} for f, b in KEYS]
    torch.save(payload, build / 's1.pt')


def test_diagnose_end_to_end_on_a_synthetic_build(tmp_path):
    build, ids = synthetic_build(tmp_path, leak=False)
    write_replay_sample(build, 6)
    (build / 'listing.asm.gz').write_bytes(b'x')
    result = md.diagnose(build, imports=fake_imports(), replay_worst=1, replay_goodware=0, resamples=100,
                         paths_of=lambda i: ('cfg', build / 'listing.asm.gz'), log=lambda *a, **k: None,
                         cache=tmp_path / 'gaps.json')
    assert result['n_samples'] == 21 and result['source'] == 'gnn-331ce0b'
    rp = result['replay']
    assert rp['n_replayed'] == 1 and rp['n_consistent'] == 1 and rp['replays'][0]['sample_id'] == 's1'
    text = md.format_report(result)
    for needle in ('Match rate per label', 'AUC of 1 - match rate', 'Ten worst families', 'Lost share by function length',
                   'Gap reports of the manifests', 'Replay of 1 samples', '(iii) function without listing lines',
                   'Mnemonic disagreements', 'Lost share by label-span length'):
        assert needle in text
    assert json.dumps(result, default=str)
    csv_path = tmp_path / 'samples.csv'
    md.write_csv(csv_path, result['samples'])
    rows = csv_path.read_text(encoding='utf-8').splitlines()
    assert rows[0] == 'sample_id,label,family,arch,cluster,n,matched,rate' and len(rows) == 22 and rows[1].startswith('s1,1,alpha,x86,fam:alpha,19,6,')


def test_replay_all_skips_big_listings_and_flags_an_inconsistent_replay(tmp_path):
    build, ids = synthetic_build(tmp_path)
    write_replay_sample(build, 7)                                      # the fake matches 6, this build says 7
    asm = tmp_path / 'a.asm.gz'
    asm.write_bytes(b'x' * 2_000_000)
    samples = md.load_samples(build)
    paths = lambda i: ('cfg', asm)                                      # noqa: E731
    out, skipped = md.replay_all(build, samples, ['s1'], 'gnn-331ce0b', 1.0, fake_imports(), lambda *a, **k: None, paths)
    assert out == [] and skipped == ['s1']
    out, skipped = md.replay_all(build, samples, ['s1'], 'gnn-331ce0b', 5.0, fake_imports(), lambda *a, **k: None, paths)
    assert skipped == [] and out[0]['consistent'] is False


def test_pick_replay_adds_random_partial_samples_per_label_with_a_seed():
    samples = [{'sample_id': f'r{i}', 'label': 1, 'rate': 0.5 + i / 40, 'n': 10} for i in range(12)]
    samples += [{'sample_id': f'g{i}', 'label': 0, 'rate': 0.6 + i / 40, 'n': 10} for i in range(12)]
    samples += [{'sample_id': 'perfect', 'label': 1, 'rate': 1.0, 'n': 10}]
    picked = md.pick_replay(samples, 1, 1, partial=3, seed=0)
    assert picked[:2] == ['r0', 'g0'] and len(picked) == 8 and 'perfect' not in picked and len(set(picked)) == 8
    assert picked == md.pick_replay(samples, 1, 1, partial=3, seed=0) and picked != md.pick_replay(samples, 1, 1, partial=3, seed=1)


def test_pick_replay_takes_the_worst_then_the_worst_goodware():
    samples = [{'sample_id': f'r{i}', 'label': 1, 'rate': i / 10, 'n': 10} for i in range(6)]
    samples += [{'sample_id': f'g{i}', 'label': 0, 'rate': 0.5 + i / 10, 'n': 10} for i in range(4)]
    assert md.pick_replay(samples, 3, 2) == ['r0', 'r1', 'r2', 'g0', 'g1']
    assert md.pick_replay(samples, 0, 0) == []


def test_build_source_reads_settings_then_checkpoint(tmp_path):
    build, _ = synthetic_build(tmp_path, source='main-c7899f0')
    assert md.build_source(build) == 'main-c7899f0'
    (build / 'settings.json').unlink()
    assert md.build_source(build) == 'main-c7899f0'
    (build / 'checkpoint.json').write_text(json.dumps({'settings': {}, 'done': []}), encoding='utf-8')
    assert md.build_source(build) == 'gnn-331ce0b'


# ------------------------------------------------------------------------------------------------------ acceptance

def good_build(tmp_path):
    """A build that passes 2-4: every sample matches 100%."""
    build, ids = synthetic_build(tmp_path, leak=False)
    for sample_id in ids:
        s = torch.load(build / f'{sample_id}.pt', weights_only=False)
        s['instruction_matched'] = torch.ones_like(s['instruction_matched'])
        torch.save(s, build / f'{sample_id}.pt')
    return build, ids


def copy_rebuild(build, perturb=None):
    def run(ids, out):
        out.mkdir(parents=True)
        for i in ids:
            p = torch.load(Path(build) / f'{i}.pt', weights_only=False)
            if perturb == i:
                p['block_features'] = torch.zeros(2)
            torch.save(p, out / f'{i}.pt')
    return run


def test_check_complete_needs_every_expected_id_and_no_exclusions(tmp_path):
    build, ids = synthetic_build(tmp_path)
    (build / 'build_summary.json').write_text(json.dumps({'processed': 20, 'total': 20, 'excluded': {}}), encoding='utf-8')
    assert acc.check_complete(build, ids)[0]
    ok, detail = acc.check_complete(build, ids + ['missing1'])
    assert not ok and '1 missing' in detail
    (build / 'build_summary.json').write_text(json.dumps({'processed': 19, 'total': 20, 'excluded': {ids[0]: 'x'}}), encoding='utf-8')
    ok, detail = acc.check_complete(build, ids)
    assert not ok and '1 excluded' in detail
    assert not acc.check_complete(build, ids[:5])[0]                     # ids outside the expected set


def test_label_gap_tail_and_auc_criteria():
    def table(r90, g90, r50, g50):
        mk = lambda n, b90, b50: {'n': n, 'n_below_90': round(n * b90), 'share_below_90': b90,    # noqa: E731
                                  'n_below_50': round(n * b50), 'share_below_50': b50}
        return {'ransomware': mk(1000, r90, r50), 'goodware': mk(500, g90, g50)}
    assert acc.check_label_gap(table(0.05, 0.04, 0, 0))[0] and acc.check_label_gap(table(0.05, 0.03, 0, 0))[0]
    assert not acc.check_label_gap(table(0.21, 0.034, 0, 0))[0]
    assert acc.check_tail(table(0, 0, 0.005, 0.0))[0] and not acc.check_tail(table(0, 0, 0.165, 0.01))[0]
    assert not acc.check_tail(table(0, 0, 0.0, 0.01))[0]                  # exactly 1% is not below 1%
    assert acc.check_auc({'auc': 0.549, 'lo': 0.5, 'hi': 0.6, 'n_clusters': 3})[0]
    assert not acc.check_auc({'auc': 0.55, 'lo': 0.5, 'hi': 0.6, 'n_clusters': 3})[0]
    assert not acc.check_auc({'auc': float('nan'), 'lo': 0, 'hi': 0, 'n_clusters': 1})[0]


def test_label_gap_limit_is_two_points_inclusive():
    mk = lambda share: {'n': 1000, 'n_below_90': int(1000 * share), 'share_below_90': share}       # noqa: E731
    assert acc.check_label_gap({'ransomware': mk(0.05), 'goodware': mk(0.03)})[0]
    assert not acc.check_label_gap({'ransomware': mk(0.052), 'goodware': mk(0.03)})[0]
    assert not acc.check_label_gap({'ransomware': mk(0.0), 'goodware': {'n': 0}})[0]


def test_payload_diff_names_the_fields_that_differ():
    a = {'x': torch.arange(3), 'y': [1, 2], 'z': 'a', 'only_a': 1}
    b = {'x': torch.arange(3), 'y': [1, 3], 'z': 'a', 'only_b': 1}
    assert acc.payload_diff(a, b) == ['only_a', 'only_b', 'y']
    assert acc.payload_diff({'x': torch.zeros(2)}, {'x': torch.zeros(2, dtype=torch.float64)}) == ['x']
    assert acc.payload_diff({'x': torch.tensor([1.0])}, {'x': torch.tensor([1.0000001])}) == ['x']


def test_deterministic_check_passes_on_identical_tensors_and_names_the_id_that_differs(tmp_path):
    build, ids = good_build(tmp_path)
    picked = acc.pick_rebuild_ids(ids, 5, 0)
    assert picked == acc.pick_rebuild_ids(ids, 5, 0) and len(set(picked)) == 5 and picked != acc.pick_rebuild_ids(ids, 5, 1)
    ok, detail = acc.check_deterministic(build, picked, copy_rebuild(build))
    assert ok and '5 of 5' in detail
    ok, detail = acc.check_deterministic(build, picked, copy_rebuild(build, perturb=picked[2]))
    assert not ok and picked[2][:12] in detail and 'block_features' in detail


def test_evaluate_and_format_a_build_that_passes_all_five(tmp_path):
    build, ids = good_build(tmp_path)
    (build / 'build_summary.json').write_text(json.dumps({'processed': 20, 'total': 20, 'excluded': {}}), encoding='utf-8')
    rows = acc.evaluate(build, ids, rebuild=copy_rebuild(build), resamples=50)
    text, overall = acc.format_rows(build, rows)
    assert overall == 'PASS' and [v for _, v, _ in rows] == ['PASS'] * 5 and 'OVERALL  PASS' in text


def test_evaluate_fails_a_leaking_partial_build_and_never_passes_a_skipped_rebuild(tmp_path):
    build, ids = synthetic_build(tmp_path)
    rows = acc.evaluate(build, ids + ['not-built'], rebuild=None, resamples=50)
    verdicts = {name.split()[0]: v for name, v, _ in rows}
    assert verdicts == {'1': 'FAIL', '2': 'FAIL', '3': 'FAIL', '4': 'FAIL', '5': 'SKIPPED'}
    assert 'partial: 20 binaries' in rows[1][2]
    text, overall = acc.format_rows(build, rows)
    assert overall == 'FAIL' and 'OVERALL  FAIL  (0 of 5 criteria pass)' in text
    good, gids = good_build(tmp_path / 'again')
    rows = acc.evaluate(good, gids, rebuild=None, resamples=50)
    assert acc.format_rows(good, rows)[1] == 'FAIL'                       # no build summary and a skipped rebuild


def test_main_exit_status_follows_the_verdict(tmp_path, monkeypatch, capsys):
    build, ids = synthetic_build(tmp_path)
    monkeypatch.setattr(acc, 'expected_ids', lambda: ids)
    out = tmp_path / 'verdict.json'
    assert acc.main(['--build', str(build), '--skip-rebuild', '--json', str(out)]) == 1
    data = json.loads(out.read_text(encoding='utf-8'))
    assert data['overall'] == 'FAIL' and len(data['criteria']) == 5 and 'acceptance of' in capsys.readouterr().out


# ----------------------------------------------------------------------------------------- real data, one binary

@needs_worktrees
@needs_real_data
def test_replay_reproduces_the_build_of_a_real_small_binary(tmp_path):
    from experiments.team_pipelines import build_b, common as tp
    rows = sorted(tp.sample_rows().values(), key=lambda r: int(r['num_blocks']))
    small = rows[0]['sample_id']
    for source in ('gnn-331ce0b', 'main-c7899f0'):
        out = tmp_path / source
        build_b.main(['--source', source, '--out', str(out), '--ids', small])
        samples = md.load_samples(out)
        replays, skipped = md.replay_all(out, samples, [small], source, 60, log=lambda *a, **k: None)
        assert not skipped and replays[0]['consistent'], replays[0]['categories']
        assert replays[0]['matched_build'] == samples[0]['matched']
        assert sum(replays[0]['categories'].values()) == replays[0]['n_selected'] - replays[0]['matched_build']

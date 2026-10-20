"""python -m pytest tests/test_eval_report.py: the RESULTS table writer of `s04_evaluation.report`: arm labels, the AUC table, the
kfold_v3 floors, the cell-balanced section, and where the file is written."""
import json

import pytest

from eval_synth import (V3_MASKED, build_v3_ab_fake, build_v3_fake, cell_sample, fake_fold_rows, gnn, needs_combined,  # noqa: F401
                        write_v3_run)
from s04_evaluation import report
from s04_evaluation.metrics import floors, scores

def test_report_labels():
    assert report.model_label(gnn()) == 'edge-typed GNN'
    assert report.model_label(gnn(control='no_edges')) == 'GNN no-edges'
    assert report.model_label(gnn(control='shuffle')) == 'GNN shuffle'
    assert report.model_label(gnn(untyped=True)) == 'GNN untyped'
    assert report.model_label(gnn(control='no_edges', encoder='mean')) == 'GNN no-edges enc-mean'
    assert report.model_label(gnn(mask_tokens=0.15)) == 'edge-typed GNN mask-tok 15%'
    labelled = set(report.LABELLED)
    for cfg in (gnn(), gnn(control='shuffle'), gnn(encoder='mean')):
        overrides = {k for k, v in report.asdict(cfg).items() if k not in labelled and v != report.DEFAULTS[k]}
        assert not overrides, overrides


def test_report_auc_table_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'RUNS', tmp_path)
    for name, selection in (('gnn', True), ('old', False)):
        cfg = gnn(seed=0) if selection else gnn(seed=0, control='no_edges')
        d = tmp_path / cfg.name
        d.mkdir()
        cfg.save(d / 'config.json')
        test = scores([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6], [0, 1, 0, 1], ['x86'] * 4)
        m = {'test': test, 'family_recall': {}, 'floors': {}}
        if selection:
            m['selection'] = {'auc': {'test': {**test, 'macro_f1': 0.123}, 'family_recall': {}}}
        (d / 'metrics.json').write_text(json.dumps(m))
        (d / 'predictions.csv').write_text('sample_id,split,label,prob,pred,arch,family,fold,n_segments_total\n'
                                           'a,test,0,0.1,0,x86,g,0,1\nb,test,1,0.9,1,x86,f,0,1\n')
    main, main_notes = report.collect({})
    auc, auc_notes = report.collect({}, auc=True)
    assert {k[2] for k, _ in main} == {'edge-typed GNN', 'GNN no-edges'}
    assert [k[2] for k, _ in auc] == ['edge-typed GNN']
    assert auc[0][1][0][0]['macro_f1'] == 0.123
    assert any('GNN no-edges' in n and 'no AUC-selected checkpoint' in n for n in auc_notes)
    assert not main_notes


def test_report_labels_new_arms():
    labels = {report.model_label(gnn(**kw)) for kw in (
        {}, dict(direction='forward'), dict(direction='backward'), dict(direction='undirected'),
        dict(drop_edges=0.2), dict(flip_edges=0.2), dict(gnn_fn_layers=8), dict(gnn_bin_layers=4),
        dict(train_fraction=0.5), dict(train_fraction=0.25))}
    assert len(labels) == 10
    assert report.model_label(gnn(direction='forward')) == 'edge-typed GNN forward'
    assert report.model_label(gnn(drop_edges=0.2, gnn_fn_layers=8)) == 'edge-typed GNN drop-edge 20% L8'
    assert report.model_label(gnn(train_fraction=0.5)) == 'edge-typed GNN train 50%'
    labelled = set(report.LABELLED)
    cfg = gnn(direction='forward', drop_edges=0.2, flip_edges=0.1, gnn_fn_layers=8, train_fraction=0.5)
    assert not {k for k, v in report.asdict(cfg).items() if k not in labelled and v != report.DEFAULTS[k]}


def test_v3_report_rows_and_floors(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    labels = {sid: y for fold in range(5) for sid, y, *_ in fake_fold_rows(fold, 0, 1.0)}
    for fold in range(5):
        labels.update({f'tr{fold}_{i}': i % 2 for i in range(10)})
    results, notes = report.collect(labels)
    keys = {k for k, _ in results}
    assert ('kfold_v3', 'all', 'GNN no-edges mask-blk 15% cell-select') in keys
    assert ('kfold_v3', 'matched', 'edge-typed GNN mask-blk 15% cell-select') in keys
    assert ('kfold_v3', 'all', 'tfidf_sig_api_lr') in keys and ('kfold_v3', 'matched', 'structural_lr') in keys
    assert any('GNN shuffle mask-blk 15% cell-select, kfold_v3, all, seed 0: only folds [0, 1, 2] present' in n for n in notes)
    s, _, fl = dict(results)[('kfold_v3', 'all', 'edge-typed GNN mask-blk 15% cell-select')][0]
    assert s['n'] == 600 and set(fl) == {'majority', 'x86_rule'} and s['recall_R_x64'] >= 0
    auc_results, _ = report.collect(labels, auc=True)
    assert ('kfold_v3', 'matched', 'tfidf_sig_api_lr') in {k for k, _ in auc_results}
    assert not any(k[0] == 'kfold' for k in keys)


def test_report_labels_for_arch_balance():
    assert report.model_label(gnn(arch_balance=True)) == 'edge-typed GNN arch-bal'
    assert report.model_label(gnn(control='shuffle', arch_balance=True)) == 'GNN shuffle arch-bal'
    assert report.model_label(gnn(arch_balance=True, encoder='mean')) == 'edge-typed GNN arch-bal enc-mean'
    assert 'arch_balance' in report.LABELLED
    overrides = {k for k, v in report.asdict(gnn(arch_balance=True)).items()
                 if k not in set(report.LABELLED) and v != report.DEFAULTS[k]}
    assert not overrides


def test_report_cell_balanced_table_and_ab_rows(tmp_path, monkeypatch):
    build_v3_ab_fake(tmp_path, monkeypatch)
    labels = {sid: y for fold in range(5) for sid, y, *_ in fake_fold_rows(fold, 0, 1.0)}
    for fold in range(5):
        labels.update({f'tr{fold}_{i}': i % 2 for i in range(10)})
    results, _ = report.collect(labels)
    keys = {k for k, _ in results}
    assert ('kfold_v3', 'all', 'edge-typed GNN mask-blk 15% arch-bal cell-select') in keys
    assert ('kfold_v3', 'all', 'edge-typed GNN mask-blk 15% cell-select') in keys
    assert ('kfold_v3', 'all', 'tfidf_sig_api_lr arch-bal') in keys and ('kfold_v3', 'all', 'tfidf_sig_api_lr') in keys
    floor = {('kfold_v3', 'all', 'floor: x86_rule'): {0: (floors([0, 1], *cell_sample([60, 10, 20, 30]))['x86_rule'],
                                                         {}, {})}}
    lines = report.cell_table(results + sorted(floor.items()))
    assert lines[0].startswith('| config | arch | seeds | Acc | macro-F1 | AUC | PR-AUC | cells |')
    row = next(l for l in lines if l.startswith('| edge-typed GNN mask-blk 15% arch-bal cell-select | all | 1 |'))
    assert row.split(' | ')[-5] == '4' and not any('bundle' in l for l in lines)
    assert sum(l.startswith('| edge-typed GNN') for l in lines) == 3


def test_report_main_writes_the_cell_balanced_section(tmp_path, monkeypatch):
    build_v3_ab_fake(tmp_path, monkeypatch)
    labels = {f'tr{fold}_{i}': i % 2 for fold in range(5) for i in range(10)}
    monkeypatch.setattr(report, 'all_labels', lambda: labels)
    report.main(['--out', str(tmp_path / 'RESULTS.md')])
    text = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')
    head, section = text.split('## Cell-balanced (each class x arch weighted equally)')
    for title in ('## Models and baselines', '## Floors on the same test manifests', '## AUC-selected checkpoint'):
        assert title in head
    assert '### F1 checkpoint at 0.5' in section and '### AUC checkpoint at the validation threshold' in section
    assert '| edge-typed GNN mask-blk 15% arch-bal cell-select | all | 1 |' in section and '| tfidf_sig_api_lr arch-bal | all | 1 |' in section
    assert section.count('| floor: x86_rule | all | 1 |') == 2 and section.count('| floor: majority | matched | 1 |') == 2
    assert 'bundle' not in section


def test_report_writes_into_the_run_directory_unless_told_otherwise(tmp_path, monkeypatch, capsys):
    build_v3_ab_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(report, 'all_labels', lambda: {f'tr{fold}_{i}': i % 2 for fold in range(5) for i in range(10)})
    report.main([])
    text = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')                      # report.RUNS is the run directory
    assert text.startswith('# Results') and capsys.readouterr().out == text + chr(10)
    elsewhere = tmp_path / 'elsewhere' / 'table.md'
    elsewhere.parent.mkdir()
    report.main(['--out', str(elsewhere)])
    assert elsewhere.read_text(encoding='utf-8') == text


@needs_combined
def test_v3_floor_rows_in_the_report_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'RUNS', tmp_path)
    report.main(['--out', str(tmp_path / 'RESULTS.md')])
    floors_part = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8').split(
        '## Floors on the same test manifests')[1].split('## AUC-selected')[0]
    for arch in ('all', 'matched'):
        for name in ('x86_rule', 'majority'):
            assert f'| floor: {name} | kfold_v3 | {arch} | 1 |' in floors_part


@needs_combined
@pytest.mark.parametrize('arch', ['all', 'matched'])
def test_v3_floors_reproduce_the_plan_table(arch):
    report.v3_floors.cache_clear()
    pooled_floors, per_fold = report.v3_floors(arch)
    assert [round(f['x86_rule']['macro_f1'], 3) for f in per_fold] == V3_X86_RULE[arch]
    assert round(pooled_floors['x86_rule']['macro_f1'], 3) == V3_X86_RULE_POOLED[arch]
    assert pooled_floors['x86_rule']['n'] == (5652 if arch == 'all' else 4850)
    assert set(pooled_floors) == {'majority', 'x86_rule'}


def test_model_labels_for_step11_fields():
    assert report.model_label(gnn(select_on='cell')) == 'edge-typed GNN cell-select'
    assert report.model_label(gnn(select_on='cell', arch_balance=True)) == 'edge-typed GNN arch-bal cell-select'
    assert report.model_label(gnn(subgraph='cap', budget=20000)) == 'edge-typed GNN cap 20k'
    assert report.model_label(gnn(budget=5000)) == 'edge-typed GNN cap 5k'
    assert report.model_label(gnn(control='no_edges', encoder='presence')) == 'GNN no-edges enc-presence'
    assert report.model_label(gnn()) == 'edge-typed GNN'
    for kw in (dict(select_on='cell'), dict(subgraph='cap', budget=5000), dict(budget=20000)):
        assert not {k for k in kw} & {k for k in report.DEFAULTS if k not in report.LABELLED}


V3_X86_RULE = {'all': [0.607, 0.686, 0.685, 0.733, 0.715], 'matched': [0.535, 0.598, 0.588, 0.660, 0.628]}


V3_X86_RULE_POOLED = {'all': 0.690, 'matched': 0.606}


def test_the_cache_field_is_labelled_so_the_report_does_not_list_it_as_an_override():
    assert 'cache' in report.LABELLED
    cfg = gnn(protocol='kfold_v3', fold=3, arch='all', cache='D:/elsewhere')
    overrides = {k for k, v in report.asdict(cfg).items() if k not in set(report.LABELLED) and v != report.DEFAULTS[k]}
    assert not overrides and report.model_label(cfg) == report.model_label(gnn(protocol='kfold_v3', fold=3, arch='all'))

"""Keep the committed shareable workbook consistent with its recorded source data."""
import importlib.util
from pathlib import Path


def test_final_excel_export_preserves_scores_and_matches_sources():
    pipeline=Path(__file__).resolve().parents[1]
    repo=pipeline.parent.parent
    exporter=pipeline/'reporting/final_export'
    spec=importlib.util.spec_from_file_location('final_excel_verify',exporter/'verify.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result=module.check(
        repo/'reports/final_paper/Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx',
        exporter/'assets/baseline.xlsx',
        exporter/'assets/attention_records.json.gz',
        repo/'reports/final_paper/layout.json',
    )
    assert result['source_matches']
    assert result['formula_errors']==0

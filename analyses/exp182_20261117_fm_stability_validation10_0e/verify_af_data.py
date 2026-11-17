"""Check the published A-F data and B-scan aggregation without loading images."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    source = read('source/af_core_records.json')
    cfg = read('experiment.json')
    records = {(row['model'], row['case']): row for row in source['core']}
    assert len(records) == 50
    assert set(records) == {(m, c) for m in cfg['models'] for c in cfg['validation_cases']}
    summary = read('analysis/af_summary.json')
    assert summary['source_sha256'] == hashlib.sha256((ROOT/'source/af_core_records.json').read_bytes()).hexdigest()
    for tag, result in summary['models'].items():
        assert set(result['standard']) == {'volume', 'bscan2d', 'ILM_OPL', 'OPL_BM', 'FULL',
                                           'ZMAX', 'ILM_OPL_MEAN', 'OPL_BM_MEAN', 'ILM_BM_MEAN'}
    values = 0
    with (ROOT/'analysis/per_patient_standard.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1800
    for row in rows:
        case = records[row['model'], row['case']]
        treatment = row['treatment']
        if treatment == 'unadjusted':
            metrics = case['endpoint']['standard']
        else:
            assert treatment.startswith('GT_oracle_')
            metrics = case['oracle'][treatment.removeprefix('GT_oracle_')]['standard']
        for key in ('mae', 'psnr', 'ssim'):
            assert abs(float(row[key])-metrics[row['object']][key]) < 1e-12
            values += 1
    with (ROOT/'analysis/bscan_per_slice.csv').open(encoding='utf-8-sig', newline='') as stream:
        bscans = list(csv.DictReader(stream))
    assert len(bscans) == 15200
    groups = defaultdict(list)
    for row in bscans:
        groups[row['model'], row['case']].append(row)
    assert set(groups) == set(records)
    for key, rows in groups.items():
        assert len(rows) == 304 and {int(r['y']) for r in rows} == set(range(304))
        for metric in ('mae', 'mse', 'psnr', 'ssim'):
            estimate = math.fsum(float(r[metric]) for r in rows)/304
            assert abs(estimate-records[key]['endpoint']['standard']['bscan2d'][metric]) < 1e-10
    expected_rows = {'depth_profiles.csv': 25600, 'gt_region_bias.csv': 1300,
                     'intensity_distributions.csv': 100, 'trajectory.csv': 360,
                     'velocity_teacher_free.csv': 320, 'topology.csv': 50,
                     'standard_means.csv': 180, 'oracle_volume_delta.csv': 15}
    for name, expected in expected_rows.items():
        with (ROOT/'analysis'/name).open(encoding='utf-8-sig', newline='') as stream:
            assert sum(1 for _ in csv.DictReader(stream)) == expected, name
    for row in source['core']:
        if row['model'] != 'XOCT_E213':
            assert len(row['trajectory']) == 9 and len(row['velocity']) == 8
            assert max(abs(v['mse_balance_residual']) for v in row['velocity']) < 2e-8
    print(json.dumps(dict(passed=True, model_cases=50, checked_standard_values=values,
                          bscan_metric_rows=15200, bscan_aggregation_values=200,
                          table_counts=expected_rows, training=0, inference=0)))


if __name__ == '__main__':
    main()


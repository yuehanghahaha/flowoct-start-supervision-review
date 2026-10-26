"""Verify the public review directory and independent data/CSV consistency."""
import ast
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in ('src', 'href') and v)


def main():
    provenance = read(ROOT / 'provenance.json')
    assert provenance['public_publication_authorized']
    original = read(ROOT / 'source/evaluation/results/metrics.json')
    cases = {case['case_id']: case for case in original['cases']}
    data = read(ROOT / 'verification/derived_data.json')
    assert len(cases) == data['patients'] == len(data['cases']) == 50
    assert [row['case_id'] for row in data['cases']] == list(cases)
    manifest = read(ROOT / 'verification/source_manifest.json')
    for row in manifest['files']:
        assert sha(ROOT / row['shared']) == row['sha256'], row['shared']
    with (ROOT / 'report/intensity_bins_per_case.csv').open(encoding='utf-8-sig', newline='') as stream:
        bins = list(csv.DictReader(stream))
    assert len(bins) == 50 * 6
    max_delta = 0.
    for row in data['cases']:
        cid = row['case_id']
        path = ROOT / row['derived_file']
        assert sha(path) == row['derived_sha256']
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {'gt', 'pred'}
            gt, pred = [arrays[key].astype(np.float64) for key in ['gt', 'pred']]
            for key in arrays.files:
                assert hashlib.sha256(arrays[key].tobytes()).hexdigest() == row['array_sha256'][key]
            assert gt.shape == pred.shape == (304, 304)
            assert np.isfinite(gt).all() and np.isfinite(pred).all()
            assert min(gt.min(), pred.min()) >= 0 and max(gt.max(), pred.max()) <= 1
        metrics = cases[cid]['metrics']
        for calculated, saved in [(gt.mean(), metrics['ilm_opl_gt_mean']),
                                  (pred.mean(), metrics['ilm_opl_pred_mean']),
                                  (np.abs(pred-gt).mean(), metrics['ilm_opl_mae'])]:
            max_delta = max(max_delta, abs(float(calculated)-saved))
            assert abs(float(calculated)-saved) < 1e-12
        selected = [r for r in bins if r['case_id'] == cid]
        assert sum(int(r['pixels']) for r in selected) == gt.size
        for key, expected in [('gt_mean', gt.mean()), ('pred_mean', pred.mean()),
                               ('mae', np.abs(pred-gt).mean())]:
            weighted = sum(int(r['pixels']) * float(r[key]) for r in selected if int(r['pixels'])) / gt.size
            assert abs(weighted-expected) < 1e-12
    summary = read(ROOT / 'report/scale_summary.json')
    with (ROOT / 'report/scale_per_case.csv').open(encoding='utf-8-sig', newline='') as stream:
        scales = list(csv.DictReader(stream))
    assert len(scales) == 150
    for row in summary['scales']:
        selected = [r for r in scales if float(r['sigma_pixels']) == row['sigma_pixels']]
        assert len(selected) == 50
        for key in ['low_std_ratio', 'high_std_ratio', 'high_mae']:
            assert abs(np.mean([float(r[key]) for r in selected]) - row[key]) < 1e-12
    links = 0
    def check(link, parent):
        nonlocal links
        if re.match(r'^[A-Za-z]+:', link) or link.startswith('#'):
            return
        path = (parent / unquote(link.split('#')[0].split('?')[0])).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), (str(parent), link)
        links += 1
    for name in ['README.md', 'REVIEW_GUIDE.md', 'REPRODUCE.md', 'NOTICE.md',
                 'report/vessel_scale_analysis.md', 'report/test50/test_analysis.md']:
        path = ROOT / name
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            check(link, path.parent)
    for name in ['report/index.html', 'report/test50/index.html']:
        path = ROOT / name
        parser = Links(); parser.feed(path.read_text(encoding='utf-8'))
        for link in parser.links:
            check(link, path.parent)
    images = list((ROOT / 'report').rglob('*.png'))
    assert len(images) == 60
    for path in images:
        with Image.open(path) as im:
            im.verify()
    python_files = [p for p in ROOT.rglob('*.py') if '.git' not in p.parts]
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8-sig'))
    forbidden = {'.zip', '.tar', '.gz', '.pt', '.pth', '.key', '.pem'}
    credential_patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
                           r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts:
            continue
        assert path.suffix.lower() not in forbidden, path
        if path.suffix.lower() in {'.py', '.md', '.json', '.txt', '.sh', '.html', '.csv'}:
            text = path.read_text(encoding='utf-8-sig')
            assert not any(re.search(p, text) for p in credential_patterns), path
    record = dict(state='passed', patients=50, original_files_verified=len(manifest['files']),
                  original_figures=58, supplemental_figures=2, derived_arrays_verified=True,
                  derived_global_metric_max_delta=max_delta, intensity_partition_reconstructs_global_metrics=True,
                  scale_csv_matches_summary=True, local_links_checked=links,
                  python_files_parsed=len(python_files), no_credentials_or_delivery_archives=True,
                  training_or_model_selection=False, public_publication_authorized=True)
    (ROOT / 'verification/share_checks.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()

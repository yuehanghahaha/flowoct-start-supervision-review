"""Extract only saved 2D shallow projections; never run the model or read volumes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.source / 'results/metrics.json').read_text(encoding='utf-8'))
    assert report['split'] == 'test' and report['patients'] == 50
    records = []
    for case in report['cases']:
        cid = case['case_id']
        source = args.source / 'results/case_views' / f'{cid}.npz'
        assert digest(source) == case['views_sha256']
        with np.load(source, allow_pickle=False) as arrays:
            selected = {'gt': arrays['gt_ilm_opl'], 'pred': arrays['pred_ilm_opl']}
            selected = {k: np.array(v, dtype=np.float32) for k, v in selected.items()}
        target = ROOT / 'derived/ilm_opl' / f'{cid}.npz'
        if target.exists():
            with np.load(target, allow_pickle=False) as arrays:
                for key, value in selected.items():
                    np.testing.assert_array_equal(arrays[key], value)
        else:
            np.savez_compressed(target, **selected)
        records.append(dict(case_id=cid, source_views_sha256=case['views_sha256'],
                            derived_file=f'derived/ilm_opl/{cid}.npz',
                            derived_sha256=digest(target),
                            array_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()
                                          for k, v in selected.items()}))
    (ROOT / 'verification/derived_data.json').write_text(
        json.dumps({'patients': 50, 'volume_data_read': False,
                    'keys': ['gt', 'pred'], 'cases': records}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'saved_2d_cases': len(records), 'model_inference': False}))


if __name__ == '__main__':
    main()

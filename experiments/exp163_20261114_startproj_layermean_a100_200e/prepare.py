"""Bind A100's existing fixed cache and audit only the 140/10 training/validation cases."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import common as c


def manifest_fingerprints(deployment):
    with Path(deployment['manifest']).open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    resolved = [dict(row, path=str((Path(deployment['data_root'])/'z256/3M'/(row['case_id']+'.npz')).resolve()))
                for row in original]
    return {hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            for rows in (original, resolved)}


def prepare(args):
    cfg = c.read(c.ROOT/'experiment.json')
    deployment, rows = c.binding(args)
    inputs = c.Inputs(deployment, cfg)
    cache_manifest = inputs.cache_info['protocol']['manifest_sha256']
    if cache_manifest not in manifest_fingerprints(deployment):
        raise ValueError('Existing cache is not bound to the supplied original or resolved manifest')
    records = []
    for row in rows:
        if row['split'] == 'test':
            continue
        source, target = inputs.pair(row)
        inputs.layers(row['case_id'])
        if row['split'] == 'validation':
            inputs.regions_xy(row['case_id'])
        with np.load(row['path'], allow_pickle=False) as raw:
            if 'axis_order' in raw and raw['axis_order'].item() != 'ZYX':
                raise ValueError('Raw source axis must be ZYX')
            for key, cached in (('oct', source), ('octa', target)):
                original = raw[key]
                if original.dtype != np.float32 or original.shape != (256, 304, 304):
                    raise ValueError('Use the audited float32 Z256 raw paired NPZ')
                if not np.isfinite(original).all() or original.min() < 0 or original.max() > 255:
                    raise ValueError('Invalid raw intensity scale')
                if not np.array_equal(cached[0], original / np.float32(255)):
                    raise ValueError('Cache differs from raw /255: '+row['case_id']+'/'+key)
        records.append(dict(case_id=row['case_id'], split=row['split'], source_sha256=c.sha(row['path']),
                            normalized_cache_exact=True))
        if len(records) % 25 == 0:
            print(f'Audited {len(records)}/150 train/validation inputs', flush=True)
    c.write(c.ROOT/'verification/preparation.json', dict(passed=True, source_contract=c.source_contract(),
            deployment=deployment, cases=records, test_arrays_opened=0, test_labels_opened=0,
            new_cache_created=False, cache_manifest_binding_verified=True))
    print('Input preparation complete. Formal training has not started.', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('data-root', 'manifest', 'cache-root', 'labels-root'):
        cli.add_argument('--'+key)
    args = cli.parse_args()
    with c.training_lock():
        prepare(args)


if __name__ == '__main__':
    main()

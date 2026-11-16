"""Memory-map the parent data, preserving the released crop and intensity arithmetic."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common import DATA, DEPLOY, ROOT, read, sha


class Inputs:
    def __init__(self, verify_hashes=False):
        self.membership = read(ROOT / 'membership.json')
        self.train_ids = {row['case_id'] for row in self.membership if row['split'] == 'train'}
        self.validation = [row for row in self.membership if row['split'] == 'validation']
        val_ids = {row['case_id'] for row in self.validation}
        assert len(self.train_ids) == 140 and len(val_ids) == 10 and not self.train_ids & val_ids
        manifest = Path(DEPLOY['manifest'])
        if sha(manifest) != DEPLOY['manifest_sha256']:
            raise ValueError('Fixed patient manifest bytes changed')
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            actual = [{key: row[key] for key in ('case_id', 'patient_id', 'split')} for row in csv.DictReader(stream)]
        assert actual == self.membership
        identity = ROOT / 'source/input_identity.json'
        assert sha(identity) == DEPLOY['input_identity_sha256']
        identities = read(identity)
        assert set(identities) == self.train_ids | val_ids
        self.paths = {}
        for cid, entry in identities.items():
            assert entry['split'] == ('train' if cid in self.train_ids else 'validation')
            expected = self.case_paths(cid)
            derived = entry['derived']
            assert set(derived) == {'data/' + path.relative_to(DATA).as_posix() for path in expected}
            for path in expected:
                item = derived['data/' + path.relative_to(DATA).as_posix()]
                if path.stat().st_size != item['bytes'] or (verify_hashes and sha(path) != item['sha256']):
                    raise ValueError('Reference input changed: ' + str(path))
            self.paths[cid] = expected
        self.schedule = {}
        with (ROOT / 'source/schedule.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                epoch = int(row['epoch'])
                records = self.schedule.setdefault(epoch, [])
                assert int(row['step']) == len(records)+1
                cid, y, x = row['case_id'], int(row['y']), int(row['x'])
                assert cid in self.train_ids and 0 <= y <= 208 and 0 <= x <= 208
                records.append((cid, y, x))
        for records in self.schedule.values():
            assert len(records) == 140 and {cid for cid, _, _ in records} == self.train_ids

    @staticmethod
    def case_paths(cid):
        return [DATA / f'OCT2OCTA3M_3D/{key}/{cid}.npy' for key in ('A', 'B')] + [
            DATA / f'Label/GT_Layers_npy/{cid}.npy']

    def volumes(self, cid, include_layers=True):
        paths = self.paths[cid][:3 if include_layers else 2]
        for path in paths:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (256, 304, 304):
                raise ValueError('Expected the reference Z256 volume: ' + str(path))
            yield array

    def batch(self, cid, y, x, cds):
        if cid not in self.train_ids:
            raise ValueError('Training and preflight crops require a training patient')
        volumes = [array[:, y:y+96, x:x+96] for array in self.volumes(cid, cds)]
        data = {key: torch.from_numpy(normalize(array))[None] for key, array in zip(('A','B'), volumes[:2])}
        if cds:
            data['seg'] = torch.from_numpy(np.stack([volumes[2] == label for label in (0,1,2)]).astype(np.float32))
        return data, cid, y, x


def normalize(array):
    return (2 * (array.astype(np.float64) / 255) - 1).astype(np.float32)


class EpochCrops(Dataset):
    def __init__(self, inputs, epoch, cds):
        self.inputs = inputs
        self.records = inputs.schedule[epoch]
        self.cds = cds

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.inputs.batch(*self.records[index], self.cds)

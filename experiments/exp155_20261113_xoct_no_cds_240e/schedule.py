"""Reproduce exp059's sampler and worker crop RNG through its first 240 epochs."""
import csv
import json
import random
import shutil

import torch
from torch.utils.data import DataLoader, Dataset

from common import BASE, CFG, DATA, DEPLOY, ROOT, read, sha, write


class CropSchedule(Dataset):
    def __init__(self, patients):
        self.patients = patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        y = random.randint(0, 208)
        x = random.randint(0, 208)
        random.random()  # Draw made by the author even with no_flip enabled.
        return self.patients[index], y, x


def loader(patients):
    return DataLoader(CropSchedule(patients), batch_size=1, shuffle=True, num_workers=4)


def build():
    path = BASE / 'checkpoints/initial.pt'
    assert sha(path) == DEPLOY['parent_initial_sha256']
    initial = torch.load(path, map_location='cpu', weights_only=False)
    assert initial['epoch'] == initial['steps'] == 0
    torch.set_rng_state(initial['rng']['torch'])
    del initial
    split = DATA / 'OCT2OCTA3M_3D/datasplit/xoct'
    train = (split / 'train_pids.csv').read_text().splitlines()
    val = (split / 'val_pids.csv').read_text().splitlines()
    members = read(ROOT / 'membership.json')
    for phase, ids in [('train', train), ('validation', val)]:
        assert set(ids) == {row['case_id'] for row in members if row['split'] == phase}
    assert len(train) == 140 and len(val) == 10
    train_loader, val_loader = loader(train), loader(val)
    target = ROOT / 'source/schedule.csv'
    temporary = target.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('epoch', 'step', 'case_id', 'y', 'x'))
        for epoch in range(1, CFG['epochs'] + 1):
            historical, = (BASE / 'logs/epochs').glob(f'epoch_{epoch:03d}_*.jsonl')
            with historical.open(encoding='utf-8') as log:
                patients = [json.loads(line)['case_id'] for line in log]
            assert len(patients) == len(train)
            for step, (pid, y, x) in enumerate(train_loader, 1):
                assert pid[0] == patients[step-1], (epoch, step, pid[0], patients[step-1])
                writer.writerow((epoch, step, pid[0], y.item(), x.item()))
            # Old central validation used the same shuffle setup and advanced the sampler RNG.
            for _ in val_loader:
                pass
            if epoch % 30 == 0:
                print(f'Matched parent patient order through E{epoch}', flush=True)
    temporary.replace(target)
    # Validate real crops separately in preflight. No volume was loaded to generate the ledger.
    identity = BASE / 'verification/data_identity.json'
    assert sha(identity) == DEPLOY['input_identity_sha256']
    shutil.copyfile(identity, ROOT / 'source/input_identity.json')
    write(ROOT / 'verification/schedule.json', dict(passed=True, epochs=CFG['epochs'], steps=CFG['epochs']*140,
          same_parent_case_order_verified=CFG['epochs']*140,
          crop_origins_replayed_from_original_worker_rng=True, parent_logged_origins=False,
          parent_initial_sha256=DEPLOY['parent_initial_sha256'], schedule_sha256=sha(target),
          patient_volumes_opened=False))


if __name__ == '__main__':
    build()

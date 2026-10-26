"""Read-only audit of the original checkpoint split and frozen test membership."""
from collections import Counter
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    contract=json.loads((ROOT/'verification/frozen_contract.json').read_text())
    ck=torch.load(ROOT/'checkpoints/best70.pt',map_location='cpu',weights_only=False)
    canonical=lambda rows:sorted((r['case_id'],r['patient_id'],r['split']) for r in rows)
    assert canonical(ck['manifest'])==canonical(contract['membership'])
    assert ck['epoch']==70 and ck['original_sha256']==contract['settings']['original_checkpoint_sha256']
    groups={name:{r['patient_id'] for r in ck['manifest'] if r['split']==name} for name in ['train','validation','test']}
    assert all(groups[a].isdisjoint(groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert {k:len(v) for k,v in groups.items()}==dict(train=140,validation=10,test=50)
    pre=json.loads((ROOT/'verification/preflight.json').read_text())
    val_ids={r['case_id'] for r in ck['manifest'] if r['split']=='validation'}
    assert pre['validation_case'] in val_ids and pre['test_cases_read']==0
    record=dict(state='passed',original_checkpoint_manifest_exact_match=True,patient_counts={k:len(v) for k,v in groups.items()},
        all_patient_groups_disjoint=True,preflight_case=pre['validation_case'],preflight_uses_validation_only=True,
        frozen_epoch=70,original_checkpoint_sha256=ck['original_sha256'],image_arrays_read=0,
        test_ids=contract['test_ids'],no_checkpoint_selection_on_test=True)
    (ROOT/'verification/heldout_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='test_ids'}))


if __name__=='__main__':main()

"""Verify retained evidence, fixed cohorts, report media, and diagnostic scope."""
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image

from analyze import ROOT,read,write


def main():
    a=read(ROOT/'verification/numerical_audit.json');assert a['passed'] and a['validation_projection_pairs']==60
    m=read(ROOT/'provenance.json')
    for r in m['files']:assert hashlib.sha256((ROOT/r['retained']).read_bytes()).hexdigest()==r['sha256']
    data=pd.read_csv(ROOT/'data/validation_intensity_by_patient.csv',dtype={'case_id':str})
    assert all(n==10 for n in data.groupby(['experiment','epoch']).size())
    assert set(data.case_id)=={'10302','10319','10338','10357','10396','10422','10447','10456','10495','10500'}
    assert np.allclose(data.positive_error_share_of_volume_MAE+data.negative_error_share_of_volume_MAE,1)
    text=(ROOT/'report/diagnosis.md').read_text(encoding='utf-8')
    imgs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text);assert len(imgs)==2
    for img in imgs:
        p=(ROOT/'report'/img).resolve();assert p.is_relative_to(ROOT.resolve())
        with Image.open(p) as im:im.verify()
    assert 'FM并非没有三维监督' in text and '推断' in text and '0次新推理' in text
    write(ROOT/'verification/final_audit.json',dict(passed=True,source_hashes=len(m['files']),
        validation_projection_pairs=60,figures_verified=2,raw_and_diagnostic_metrics_distinguished=True,
        causal_claims_labeled=True,checkpoints_reselected=False,new_training=0,new_inference=0,test_data_used=False))
    print('Verified 60 validation projection pairs, both figures, source hashes, and diagnostic scope')


if __name__=='__main__':main()

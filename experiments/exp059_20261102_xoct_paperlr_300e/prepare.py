"""Convert ONLY fixed training/validation NPZ and native layer boundaries."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil
import numpy as np
from scipy.io import loadmat
from common import ROOT, CONFIG, read, write, digest, now


def rasterize(layers, depth=256):
    """Unpublished adaptation: center sampling, MATLAB 1-based native boundaries.

    Layer is [6,Y,X]. Label 1: ILM <= z < OPL. Label 2: OPL <= z <= BM.
    This makes labels disjoint; zero is the complement. No lateral transpose.
    """
    a=np.asarray(layers,dtype=np.float64)
    assert a.shape==(6,304,304) and np.isfinite(a).all()
    assert a.min()>=1 and a.max()<=640 and (np.diff(a,axis=0)>=0).all()
    z=((np.arange(depth,dtype=np.float64)+0.5)*640/depth-0.5)[:,None,None]
    out=np.zeros((depth,304,304),dtype=np.uint8)
    out[(z>=a[0]-1)&(z<a[2]-1)]=1
    out[(z>=a[2]-1)&(z<=a[5]-1)]=2
    return out


def prepare(manifest, labels_root):
    assert not (ROOT/'verification/training_launch.json').exists(), 'Preparation is frozen after launch'
    rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    frozen=read(ROOT/'membership.json')
    assert [{k:r[k] for k in ['case_id','patient_id','split']} for r in rows]==frozen
    assert Counter(r['split'] for r in rows)=={'train':140,'validation':10,'test':50}
    for split in ['train','validation','test']:
        ids=[r['patient_id'] for r in rows if r['split']==split]
        assert len(ids)==len(set(ids))
    subject_splits={}
    for r in rows:
        assert subject_splits.setdefault(r['patient_id'],r['split'])==r['split']
    selected=[r for r in rows if r['split']!='test']
    data=ROOT/'data';vol=data/'OCT2OCTA3M_3D';seg=data/'Label/GT_Layers_npy'
    for d in [vol/'A',vol/'B',vol/'datasplit/xoct',seg,data/'native_layers']:d.mkdir(parents=True,exist_ok=True)
    for phase,split in [('train','train'),('val','validation')]:
        (vol/'datasplit/xoct'/f'{phase}_pids.csv').write_text(''.join(r['case_id']+'\n' for r in selected if r['split']==split),'utf-8')
    identities=read(ROOT/'verification/data_identity.json') if (ROOT/'verification/data_identity.json').exists() else {}
    for i,r in enumerate(selected):
        cid=r['case_id'];source=(manifest.parent/r['path']).resolve();label=labels_root/'GT_Layers'/f'{cid}.mat'
        destinations=[vol/'A'/f'{cid}.npy',vol/'B'/f'{cid}.npy',seg/f'{cid}.npy']
        original=dict(source_npz_sha256=digest(source),native_layers_sha256=digest(label))
        if cid in identities and identities[cid]['original']==original:
            if all(p.exists() and digest(p)==identities[cid]['derived'][str(p.relative_to(ROOT))]['sha256'] for p in destinations):
                existing_seg=np.load(destinations[2],mmap_mode='r',allow_pickle=False)
                identities[cid]['zero_length_ilm_opl_rays']=int(((existing_seg==1).sum(0)==0).sum())
                print(f'Existing verified {i+1}/{len(selected)} {cid}',flush=True);continue
        with np.load(source,allow_pickle=False) as f:
            assert f['axis_order'].item()=='ZYX'
            for key,dest in zip(['oct','octa'],destinations):
                a=f[key]
                assert a.shape==tuple(CONFIG['shape_ZYX']) and a.dtype==np.float32
                assert np.isfinite(a).all() and a.min()>=0 and a.max()<=255
                np.save(dest,a,allow_pickle=False)
        a=rasterize(loadmat(label)['Layer']);np.save(destinations[2],a,allow_pickle=False)
        shutil.copy2(label,data/'native_layers'/label.name)
        identities[cid]=dict(split=r['split'],patient_id=r['patient_id'],source_npz=str(source),native_layers=str(label),
            original=original,derived={str(p.relative_to(ROOT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in destinations},
            label_voxels={str(k):int((a==k).sum()) for k in [0,1,2]},
            zero_length_ilm_opl_rays=int(((a==1).sum(0)==0).sum()))
        write(ROOT/'verification/data_identity.json',identities)
        print(f'Prepared {i+1}/{len(selected)} {cid} {r["split"]}',flush=True)
    expected={r['case_id'] for r in selected}
    assert set(identities)==expected
    write(ROOT/'verification/data_identity.json',identities)
    for folder in [vol/'A',vol/'B',seg]:assert {p.stem for p in folder.glob('*.npy')}==expected
    write(ROOT/'deployment.json',dict(manifest=str(manifest),manifest_sha256=digest(manifest),labels_root=str(labels_root),time=now()))
    write(ROOT/'verification/preparation.json',dict(time=now(),train=140,validation=10,test_arrays_opened=False,
        test_derived_files=0,manifest_sha256=digest(manifest),membership_sha256=digest(ROOT/'membership.json'),
        volume_shape_ZYX=CONFIG['shape_ZYX'],volume_dtype='float32',stored_intensity_units='0..255 unchanged from NPZ',
        label_protocol=rasterize.__doc__,identities_sha256=digest(ROOT/'verification/data_identity.json'),
        original_author_layer_preprocessing_available=False,status='passed'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--labels-root',type=Path,required=True)
    a=p.parse_args();prepare(a.manifest,a.labels_root)

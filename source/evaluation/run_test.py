"""Frozen epoch70 test evaluation; disjoint hardware shards, no training or tuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.io import loadmat
import torch

from fm3d.data import read_rows,load_pair
from fm3d.model import make_model
from fm3d.inference import generate_volume
from fm3d.train_start import quality,file_hash,write_json
from evaluation_metrics import additional_metrics,save_views
from inference_batched import generate_batched

ROOT=Path(__file__).resolve().parent


def frozen():
    contract=json.loads((ROOT/'verification/frozen_contract.json').read_text())
    for rel,value in contract['file_sha256'].items():assert file_hash(ROOT/rel)==value,rel
    return contract


@torch.inference_mode()
def execute(args):
    contract=frozen();settings=contract['settings']
    assert settings['split']=='test' and settings['checkpoint_epoch']==70
    plan=json.loads((ROOT/'execution_plan.json').read_text())
    ids=plan[args.role]['case_ids']
    rows=read_rows(args.data/'split_3M_z256.csv')
    membership={r['case_id']:(r['patient_id'],r['split']) for r in rows}
    assert membership=={r['case_id']:(r['patient_id'],r['split']) for r in contract['membership']}
    selected=[r for r in rows if r['case_id'] in ids]
    assert len(selected)==len(ids) and all(r['split']=='test' for r in selected)
    ck=torch.load(ROOT/'checkpoints/best70.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==70 and ck['original_sha256']==settings['original_checkpoint_sha256']
    device=torch.device(plan[args.role]['device'])
    workers=plan[args.role]['workers'];threads=plan[args.role]['threads']
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=make_model(16,True,'voxel').to(device);model.load_state_dict(ck['model']);model.eval()
    out=ROOT/'results';(out/'case_views').mkdir(parents=True,exist_ok=True)
    provenance=dict(started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),role=args.role,
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        cpu_workers=workers,cpu_threads=threads,window_batch=plan[args.role].get('window_batch',1),
        original_checkpoint_sha256=ck['original_sha256'],inference_checkpoint_sha256=file_hash(ROOT/'checkpoints/best70.pt'),
        contract_sha256=file_hash(ROOT/'verification/frozen_contract.json'),case_ids=ids,
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-image']},
        protocol='FP32, no AMP/TF32, Euler8, original 256x64x64 windows, overlap0.5, Hann velocity blend each step, final clamp only',
        full_training_started=False,test_hyperparameter_selection=False)
    write_json(ROOT/'verification'/f'run_{args.role}.json',provenance)
    records=[];tick=time.perf_counter()
    def case(row):
      with torch.inference_mode():
        cid=row['case_id'];mp=out/f'{cid}_metrics.json';vp=out/'case_views'/f'{cid}.npz'
        if mp.exists() and vp.exists():
            old=json.loads(mp.read_text())
            assert old['original_checkpoint_sha256']==ck['original_sha256'] and old['split']=='test'
            assert file_hash(vp)==old['views_sha256']
            print(cid,'verified existing result',flush=True);return old
        begin=time.perf_counter()
        # This is the first data access for the formal, frozen test-case evaluation.
        assert row['split']=='test'
        source,target=[t[None] for t in load_pair(row['path'],'ZYX',255.,256)]
        assert source.shape==target.shape==(1,1,256,304,304)
        data_sha=file_hash(row['path']);assert data_sha==contract['case_data_sha256'][cid]
        if args.layers:
            lp=args.layers/f'{cid}.mat'
            assert file_hash(lp)==contract['case_layer_mat_sha256'][cid]
            layers=np.asarray(loadmat(lp)['Layer'],dtype=np.float32)
        else:
            ap=args.assets/f'{cid}.npz'
            with np.load(ap,allow_pickle=False) as a:layers=np.array(a['layers'],dtype=np.float32)
        layer_digest=hashlib.sha256(np.ascontiguousarray(layers).tobytes()).hexdigest()
        assert layer_digest==contract['case_layer_array_sha256'][cid]
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
            prediction=generate_batched(model,source,settings['patch'],8,.5,plan[args.role]['window_batch'],
                lambda done,total,n: print(f'{cid} Euler {done}/{total}',flush=True))
        else:
            prediction=generate_volume(model,source,settings['patch'],8,.5,amp=False)
        assert torch.isfinite(prediction).all() and prediction.min()>=0 and prediction.max()<=1
        metrics,_=quality(prediction,target,layers)
        extra,pa,ga=additional_metrics(prediction,target,layers,settings['intensity_diagnostic_threshold']);metrics.update(extra)
        p,g=prediction[0,0].numpy(),target[0,0].numpy()
        high=g>settings['intensity_diagnostic_threshold'];sq=np.square(p.astype(np.float64)-g)
        metrics.update(high_intensity_mse=float(sq[high].mean()),background_mse=float(sq[~high].mean()),
            high_intensity_absolute_error_share=float(np.abs(p-g)[high].sum(dtype=np.float64)/np.abs(p-g).sum(dtype=np.float64)),
            high_intensity_squared_error_share=float(sq[high].sum()/sq.sum()),
            volume_pred_mean=float(p.mean(dtype=np.float64)),volume_gt_mean=float(g.mean(dtype=np.float64)),
            pred_zero_fraction=float((p==0).mean()),pred_one_fraction=float((p==1).mean()))
        zero,_=quality(torch.zeros_like(target),target,layers)
        save_views(vp,prediction,source,target,pa,ga)
        result=dict(case_id=cid,patient_id=row['patient_id'],disease=row.get('disease',''),split='test',
            checkpoint_epoch=70,original_checkpoint_sha256=ck['original_sha256'],role=args.role,device=str(device),
            seconds=time.perf_counter()-begin,metrics=metrics,zero_reference=zero,
            input_npz_sha256=data_sha,layer_array_sha256=layer_digest,views_sha256=file_hash(vp),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        write_json(mp,result)
        print(f'{cid} COMPLETE MAE={metrics["mae"]:.6f} SSIM={metrics["ilm_opl_ssim"]:.4f} PCC={metrics["ilm_opl_pcc"]:.4f} seconds={result["seconds"]:.1f}',flush=True)
        return result
    print(f'Frozen test: role={args.role} cases={len(selected)} workers={workers} device={device}',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(case,row) for row in selected]
        for future in as_completed(futures):
            records.append(future.result())
            write_json(out/f'status_{args.role}.json',dict(state='running',completed=len(records),total=len(selected),last_case=records[-1]['case_id']))
    write_json(out/f'status_{args.role}.json',dict(state='complete',completed=len(records),total=len(selected),seconds=time.perf_counter()-tick,split='test',checkpoint_epoch=70))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--role',choices=['gpu','cpu'],required=True)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--layers',type=Path);a.add_argument('--assets',type=Path)
    execute(p.parse_args())

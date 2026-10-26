"""Validate window batching ONLY on a previously evaluated validation patient."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
import torch
from fm3d.data import read_rows,load_pair
from fm3d.model import make_model
from fm3d.train_start import quality,file_hash,write_json
from evaluation_metrics import additional_metrics
from inference_batched import generate_batched

ROOT=Path(__file__).resolve().parent
DATA=Path(r'D:\data\code\keyan_fuxian\FlowOCT\data\OCTA500_common_v1')


@torch.inference_mode()
def main():
    settings=json.loads((ROOT/'experiment.json').read_text())
    prov=json.loads((ROOT/'provenance.json').read_text())
    ck=torch.load(ROOT/'checkpoints/best70.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==70 and ck['original_sha256']==settings['original_checkpoint_sha256']
    for name,value in ck['config']['source_sha256'].items():assert file_hash(ROOT/'fm3d'/name)==value
    rows=read_rows(DATA/'split_3M_z256.csv')
    row=next(r for r in rows if r['case_id']==settings['validation_preflight_case'])
    assert row['split']=='validation'
    source,target=[v[None] for v in load_pair(row['path'],'ZYX',255.,256)]
    layers=np.asarray(loadmat(DATA/'labels_native/GT_Layers'/f'{row["case_id"]}.mat')['Layer'],dtype=np.float32)
    old=json.loads((ROOT/'evidence/validation_10302_metrics.json').read_text())['metrics']
    with np.load(ROOT/'evidence/validation_10302_views.npz') as f:oldviews={k:f[k] for k in f.files}
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=make_model(16,True,'voxel').cuda();model.load_state_dict(ck['model']);model.eval()
    attempts=[]
    for size in settings['inference_batch_candidates_validation_only']:
        print('Validation-only preflight',row['case_id'],'window batch',size,flush=True)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        try:
            pred=generate_batched(model,source,settings['patch'],8,.5,size,
                lambda done,total,windows:print(f'validation Euler {done}/{total}, windows={windows}',flush=True))
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(dict(batch=size,result='out_of_memory',error=str(exc)))
            torch.cuda.empty_cache();continue
        seconds=time.perf_counter()-start
        metrics,_=quality(pred,target,layers)
        extra,pa,ga=additional_metrics(pred,target,layers,.1);metrics.update(extra)
        delta={k:abs(v-old[k]) for k,v in metrics.items() if v is not None}
        md=max(delta.values())
        views={name:float(np.max(np.abs(pa[key]-oldviews['pred_'+name]))) for name,key in [('full_mean','FULL'),('ilm_opl','ILM_OPL'),('opl_bm','OPL_BM')]}
        assert md<settings['preflight_metric_absolute_tolerance'],delta
        assert max(views.values())<settings['preflight_projection_max_absolute_tolerance'],views
        for name,key in [('full_mean','FULL'),('ilm_opl','ILM_OPL'),('opl_bm','OPL_BM')]:assert np.array_equal(ga[key],oldviews['gt_'+name])
        rec=dict(state='passed',validation_case=row['case_id'],test_cases_read=0,window_batch_size=size,
            seconds=seconds,metric_differences=delta,metric_max_absolute_difference=md,projection_max_differences=views,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),original_checkpoint_sha256=ck['original_sha256'],
            inference_checkpoint_sha256=file_hash(ROOT/'checkpoints/best70.pt'),gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,numpy=np.__version__,attempts=attempts,settings=settings,
            gt_projections_exact=True,source_input_sha256=file_hash(row['path']))
        write_json(ROOT/'verification/preflight.json',rec)
        print(json.dumps(rec,indent=2),flush=True);return
    raise RuntimeError('No validation-verified GPU batching mode fits this device')


if __name__=='__main__':main()

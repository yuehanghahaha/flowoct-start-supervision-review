from pathlib import Path
from time import perf_counter
import csv
import numpy as np
from PIL import Image,ImageDraw
import torch
from .runtime import read,write,atomic,digest,seed_for,paused,Pause
from .inference import generate
from .metrics import evaluate,GROUPS,METRICS,PROTOCOL


def panel(path,pred,truth):
    names=['ilm_opl_max','opl_bm_max','full_mean'];h,w=pred[names[0]].shape
    canvas=Image.new('RGB',(3*(w+8),2*(h+24)),'white');d=ImageDraw.Draw(canvas)
    for row,maps in enumerate((truth,pred)):
        for col,name in enumerate(names):
            upper=.5 if name.endswith('max') else .15
            a=np.uint8(np.clip(maps[name]/upper,0,1)*255+.5)
            x=col*(w+8);y=row*(h+24)
            d.text((x,y),('GT ' if row==0 else 'Pred ')+name,fill='black')
            canvas.paste(Image.fromarray(a),(x,y+18))
    with atomic(path) as tmp:canvas.save(tmp,format='PNG')


def evaluate_split(model,teacher,dataset,phase,cfg,stats,device,folder,weight_id,teacher_id):
    folder=Path(folder);records=[];retain=phase=='test' and cfg['inference']['retain_test_volumes']
    for row in dataset.split(phase):
        if paused(dataset.run):raise Pause('evaluation patient boundary')
        case=folder/row['case_id'];case.mkdir(parents=True,exist_ok=True)
        for key in ('npz','layers','capillary','large','faz'):dataset.identify(row,key)
        identity={'case_id':row['case_id'],'phase':phase,'weight_id':weight_id,'teacher_id':teacher_id,
                  'inputs':{k:dataset.identities[row['case_id']+'/'+k] for k in ('npz','layers','capillary','large','faz')},
                  'noise_seed':seed_for(cfg['seed'],'generate',phase,row['case_id']),
                  'inference':cfg['inference'],'retained_volume':retain}
        record=read(case/'record.json') if (case/'record.json').exists() else None
        if record is not None and record.get('identity')!=identity:
            raise ValueError('Existing evaluation belongs to another identity; use an independent output folder.')
        reusable=record is not None and record.get('identity')==identity and all((case/name).exists() and digest(case/name)==sha for name,sha in record['files'].items())
        if not reusable:
            source,target,layers=dataset.case(row);maps=dataset.maps(row)
            start=perf_counter();prediction,outside=generate(model,source.to(device),cfg,stats,identity['noise_seed'])
            prediction=prediction[0,0].cpu().numpy();seconds=perf_counter()-start
            values,diagnostics,bscans,p,g=evaluate(prediction,target[0,0].numpy(),layers[0].numpy(),maps,teacher,stats,cfg,device)
            with atomic(case/'projections.npz') as tmp,tmp.open('wb') as f:np.savez_compressed(f,**{'pred_'+k:v for k,v in p.items()},**{'gt_'+k:v for k,v in g.items()})
            panel(case/'projections.png',p,g)
            files=['projections.npz','projections.png']
            if retain:
                with atomic(case/'prediction.npy') as tmp,tmp.open('wb') as f:np.save(f,prediction,allow_pickle=False)
                files.append('prediction.npy')
            record={'identity':identity,'metrics':values,'vascular_diagnostics':diagnostics,'bscans':bscans,
                    'seconds':seconds,'clipped_fraction':outside,'files':{name:digest(case/name) for name in files}}
            write(case/'record.json',record)
            del prediction,source,target,layers,p,g
        records.append(record)
    summary={'phase':phase,'patient_count':len(records),'patient_ids':[r['identity']['case_id'] for r in records],
             'weight_id':weight_id,'teacher_id':teacher_id,'protocol':PROTOCOL,
             'mean':{group:{m:float(np.mean([r['metrics'][group][m] for r in records])) for m in METRICS} for group in GROUPS},
             'selection_score':float(np.mean([r['vascular_diagnostics']['selection_score'] for r in records]))}
    with atomic(folder/'patients.csv') as tmp,tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id','object',*METRICS])
        for r in records:
            for group in GROUPS:w.writerow([r['identity']['case_id'],group,*[r['metrics'][group][m] for m in METRICS]])
    with atomic(folder/'bscans.csv') as tmp,tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id','y',*METRICS])
        for r in records:
            for scan in r['bscans']:w.writerow([r['identity']['case_id'],scan['y'],*[scan[m] for m in METRICS]])
    with atomic(folder/'vascular_diagnostics.csv') as tmp,tmp.open('w',encoding='utf-8',newline='') as f:
        keys=list(records[0]['vascular_diagnostics']);w=csv.writer(f);w.writerow(['case_id',*keys])
        for r in records:w.writerow([r['identity']['case_id'],*[r['vascular_diagnostics'][k] for k in keys]])
    write(folder/'summary.json',summary);return summary

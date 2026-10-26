"""Render scientifically fixed-window per-patient plots and complete overviews."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'report/figures'
FONT='C:/Windows/Fonts/msyh.ttc'


def font(size):return ImageFont.truetype(FONT,size)


def label(draw,xy,value,size=20,color='#20394b'):
    draw.multiline_text(xy,value,font=font(size),fill=color,spacing=5)


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def image(values,upper,size=304,error=False):
    raw=np.clip(values/upper,0,1)
    if error:
        pixels=(colormaps['magma'](raw)[...,:3]*255).round().astype('uint8');im=Image.fromarray(pixels)
    else:im=Image.fromarray((raw*255).round().astype('uint8')).convert('RGB')
    # Keep the saved voxel-grid aspect ratio for B-scans, with letterboxing.
    ratio=min(size/im.width,size/im.height)
    im=im.resize((round(im.width*ratio),round(im.height*ratio)),Image.Resampling.NEAREST)
    bg=Image.new('RGB',(size,size),'black');bg.paste(im,((size-im.width)//2,(size-im.height)//2));return bg


def render_case(record):
    cid=record['case_id'];m=record['metrics'];a=load(ROOT/'results/case_views'/f'{cid}.npz')
    rows=[('ILM–OPL 最大投影','gt_ilm_opl','pred_ilm_opl',.5,.2),
          ('OPL–BM 最大投影','gt_opl_bm','pred_opl_bm',.5,.2),
          ('全深度均值投影','gt_full_mean','pred_full_mean',.08,.04),
          ('全深度最大投影','gt_zmax','pred_zmax',.5,.2),
          (f'中间 B-scan\nY={int(a["bscan_y_positions"][1])}','bscan_gt_1','bscan_pred_1',.35,.15),
          ('浅层中央128×128\n固定区域放大','gt_ilm_opl','pred_ilm_opl',.5,.2)]
    width=1285;header=155;rowh=355;left=245;step=337
    canvas=Image.new('RGB',(width,header+len(rows)*rowh+75),'#f7f9fb');d=ImageDraw.Draw(canvas)
    label(d,(24,18),f'测试病例 {cid}  |  {record["disease"]}  |  固定第70轮',28)
    label(d,(24,64),f'体积 MAE {m["mae"]:.6f}  PSNR {m["psnr"]:.3f}dB  |  浅层 SSIM {m["ilm_opl_ssim"]:.4f}  PCC {m["ilm_opl_pcc"]:.4f}',20)
    label(d,(24,101),'原始浮点值计算指标；所有病例共享固定显示窗，无逐图拉伸、锐化或去噪。',19)
    for j,name in enumerate(['真实OCTA','预测OCTA','绝对误差 / magma']):label(d,(left+j*step,header-25),name,21)
    for i,(title,gkey,pkey,upper,eupper) in enumerate(rows):
        y=header+i*rowh+15;gt,pred=a[gkey],a[pkey]
        if i==5:
            center=np.array(gt.shape)//2;sl=(slice(center[0]-64,center[0]+64),slice(center[1]-64,center[1]+64));gt,pred=gt[sl],pred[sl]
        label(d,(24,y+80),title,20)
        label(d,(24,y+155),f'显示窗 [0,{upper:g}]\n误差窗 [0,{eupper:g}]',17)
        for j,v in enumerate([gt,pred,np.abs(gt-pred)]):canvas.paste(image(v,eupper if j==2 else upper,error=j==2),(left+j*step,y))
        gradient=np.repeat(np.linspace(1,0,256)[:,None],14,axis=1)
        bar=(colormaps['magma'](gradient)[...,:3]*255).round().astype('uint8')
        canvas.paste(Image.fromarray(bar),(width-29,y+24));label(d,(width-56,y+4),f'{eupper:g}',12);label(d,(width-29,y+286),'0',12)
    label(d,(24,canvas.height-55),'投影：行=X，列=Y；B-scan保留保存网格比例。层标注仅用于评价，生成输入只有OCT。',17)
    canvas.save(FIG/f'{cid}_comparison.png')


def overview(records):
    for first in range(0,len(records),10):
        batch=records[first:first+10];w=1024;h=1560
        canvas=Image.new('RGB',(w,h),'#f7f9fb');d=ImageDraw.Draw(canvas)
        label(d,(24,16),f'全部50例测试 · 第{first//10+1}/5页 · 按病例编号排序',26)
        label(d,(24,61),'每例：左GT、右预测；ILM–OPL最大投影固定显示窗 [0,0.5]',18)
        for i,r in enumerate(batch):
            x=24+i%2*500;y=112+i//2*281;a=load(ROOT/'results/case_views'/f'{r["case_id"]}.npz')
            label(d,(x,y),f'{r["case_id"]}  SSIM {r["metrics"]["ilm_opl_ssim"]:.3f}  PCC {r["metrics"]["ilm_opl_pcc"]:.3f}',17)
            for j,key in enumerate(['gt_ilm_opl','pred_ilm_opl']):canvas.paste(image(a[key],.5,230),(x+j*238,y+29))
        canvas.save(FIG/f'test_overview_{first//10+1}.png')


def main(args):
    FIG.mkdir(parents=True,exist_ok=True)
    ids=json.loads((ROOT/'verification/frozen_contract.json').read_text())['test_ids']
    records=[]
    for cid in ids:
        p=ROOT/'results'/f'{cid}_metrics.json'
        if not p.exists():continue
        r=json.loads(p.read_text());records.append(r)
        if args.overwrite or not (FIG/f'{cid}_comparison.png').exists():render_case(r)
    if len(records)==50:overview(records)
    print(f'Rendered {len(records)}/50 test cases; complete overview={len(records)==50}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overwrite',action='store_true');main(p.parse_args())

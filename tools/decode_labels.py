"""Explicitly decode one version of CAVF labels; never infer a palette.

Example: python tools/decode_labels.py --input label.png --mapping palette.json --output masks/case
palette.json maps background/capillary/artery/vein/faz to integer codes OR RGB triples.
Actual release codes must be inspected; this project does not guess them.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image


def decode(array,mapping):
    names={'background','capillary','artery','vein','faz'}
    if set(mapping)!=names:raise ValueError('Provide explicit codes for all five CAVF categories.')
    assigned=np.zeros(array.shape[:2],bool);result={}
    for name,code in mapping.items():
        if isinstance(code,list):
            if array.ndim!=3 or array.shape[-1]!=len(code):raise ValueError('RGB mapping requires an RGB input.')
            mask=(array==np.array(code)).all(-1)
        else:
            if array.ndim!=2:raise ValueError('Integer mapping requires an index image.')
            mask=array==code
        if (assigned&mask).any():raise ValueError('Class codes overlap.')
        assigned|=mask;result[name]=mask
    if not assigned.all():raise ValueError('Unmapped values exist; check the dataset version and palette.')
    return {'capillary':result['capillary'],'large':result['artery']|result['vein'],'faz':result['faz']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--mapping',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();mapping=json.loads(args.mapping.read_text())
    image=Image.open(args.input)
    array=np.asarray(image.convert('RGB') if any(isinstance(v,list) for v in mapping.values()) else image)
    masks=decode(array,mapping);args.output.mkdir(parents=True,exist_ok=False)
    for name,mask in masks.items():np.save(args.output/(name+'.npy'),mask.astype(np.uint8),allow_pickle=False)
    report={'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'mapping':mapping,
            'shape':list(array.shape),'orientation':'unchanged; declare XY/YX in manifest'}
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

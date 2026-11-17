"""Explicit existing split import and canonical OCTA-500 label adapters."""
from pathlib import Path
from collections import Counter,OrderedDict
import csv
import os
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt,binary_dilation
from skimage.morphology import skeletonize
from .runtime import digest,read,write,seed_for
from .geometry import slab,uniform_tile


def owned(path):
    p=Path(path).resolve()
    if os.name!='nt' and p!=Path.home().resolve() and Path.home().resolve() not in p.parents:
        raise ValueError('Inputs must remain inside the authorized persistent home: '+str(p))
    return p


def binary_mask(path,shape,orientation):
    p=owned(path)
    a=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.asarray(Image.open(p))
    if a.ndim!=2 or not set(np.unique(a)).issubset({0,1,255}):
        raise ValueError('Supply an explicitly decoded binary label, not an inferred CAVF palette: '+str(p))
    if orientation=='XY':a=a.T
    elif orientation!='YX':raise ValueError('Declare annotation_orientation=YX or XY.')
    if a.shape!=tuple(shape):raise ValueError('Annotation alignment/size mismatch.')
    return a>0


def structure_maps(cap,large,faz,bins):
    # Some releases include large vessels in capillary masks. Subtract explicitly.
    overlap=int((cap&large).sum());cap=cap&~large&~faz;large=large&~faz
    vessel=cap|large
    skeleton=skeletonize(cap)
    radius=distance_transform_edt(vessel)
    if skeleton.any():
        _,nearest=distance_transform_edt(~skeleton,return_indices=True)
        diameter=2*radius[tuple(nearest)]
    else:diameter=np.zeros_like(radius)
    small=skeleton&(diameter<=bins[0]);medium=skeleton&(diameter>bins[0])&(diameter<=bins[1])
    return {'capillary':cap,'large':large,'faz':faz,'vessel':vessel,'skeleton':skeleton,
            'small_centerline':small,'medium_centerline':medium,
            'faz_ring':binary_dilation(faz,iterations=3)&~faz,
            'background':~vessel,'removed_cap_large_overlap':overlap}


class Dataset:
    def __init__(self,manifest,cfg,run,allowed):
        self.cfg=cfg;self.run=Path(run);self.allowed=set(allowed);self.cache=OrderedDict();self.labels=OrderedDict()
        self.manifest=owned(manifest)
        with self.manifest.open(encoding='utf-8-sig',newline='') as f:self.rows=list(csv.DictReader(f))
        required={'case_id','patient_id','split','npz','layers','capillary','large','faz','annotation_orientation','axis_order'}
        if not self.rows or not required.issubset(self.rows[0]):raise ValueError('See docs/manifest_schema.md.')
        if Counter(r['split'] for r in self.rows)!=Counter(cfg['data']['expected_counts']):raise ValueError('Existing fixed 140/10/50 split required.')
        if len({r['case_id'] for r in self.rows})!=len(self.rows):raise ValueError('Repeated case ID.')
        if len({r['patient_id'] for r in self.rows})!=len(self.rows):raise ValueError('The declared 3M protocol requires one distinct patient per case.')
        groups={};image_paths=set()
        for row in self.rows:
            if not row['case_id'].isalnum():raise ValueError('Unsafe case ID.')
            if groups.setdefault(row['patient_id'],row['split'])!=row['split']:raise ValueError('Patient leakage between splits.')
            if row['axis_order']!='ZYX':raise ValueError('Explicit ZYX input required.')
            for key in ('npz','layers','capillary','large','faz'):
                row[key]=str(owned(self.manifest.parent/row[key]))
            if row['npz'] in image_paths:raise ValueError('Multiple cases point to the same OCT/OCTA file.')
            image_paths.add(row['npz'])
        lock={'manifest_sha256':digest(self.manifest),'members':[{k:r[k] for k in ('case_id','patient_id','split')} for r in self.rows]}
        path=self.run/'split_lock.json'
        if path.exists() and read(path)!=lock:raise ValueError('Fixed split/manifest changed. Do not silently rebind.')
        if not path.exists():write(path,lock)
        self.identities=read(self.run/'input_identities.json') if (self.run/'input_identities.json').exists() else {}
        self.checked=set()

    def split(self,name):
        if name not in self.allowed:raise PermissionError('This phase cannot read '+name)
        return [r for r in self.rows if r['split']==name]

    def identify(self,row,key):
        if row['split'] not in self.allowed:raise PermissionError('Forbidden data split.')
        token=row['case_id']+'/'+key
        if token not in self.checked:
            p=owned(row[key]);item={'sha256':digest(p),'bytes':p.stat().st_size}
            if token in self.identities and self.identities[token]!=item:raise ValueError('Previously used input changed: '+token)
            self.identities[token]=item;self.checked.add(token);write(self.run/'input_identities.json',self.identities)

    def case(self,row):
        self.identify(row,'npz');self.identify(row,'layers')
        key=row['case_id']
        if key not in self.cache:
            with np.load(row['npz'],allow_pickle=False) as f:
                arrays=[]
                for name in ('oct','octa'):
                    raw=f[name]
                    if raw.dtype.kind=='f' and ('intensity_scale' not in f or float(f['intensity_scale'])!=self.cfg['data']['stored_scale']):
                        raise ValueError('Float arrays require explicit NPZ intensity_scale=255; do not guess units.')
                    a=np.array(f[name],dtype=np.float32,copy=True)
                    if list(a.shape)!=self.cfg['data']['shape'] or not np.isfinite(a).all() or a.min()<0 or a.max()>self.cfg['data']['stored_scale']:
                        raise ValueError('Invalid canonical volume: '+key+'/'+name)
                    arrays.append(torch.from_numpy(a/self.cfg['data']['stored_scale'])[None,None])
            p=Path(row['layers'])
            layer=loadmat(p)['Layer'] if p.suffix=='.mat' else np.load(p,allow_pickle=False)
            layer=np.asarray(layer,np.float32)
            if layer.shape!=(6,*self.cfg['data']['shape'][1:]) or not np.isfinite(layer).all() or layer.min()<1 or layer.max()>self.cfg['data']['native_layer_depth'] or (np.diff(layer,axis=0)<0).any():
                raise ValueError('Expected native640 MATLAB 1-based layer boundaries, shape6YX.')
            arrays.append(torch.from_numpy(layer)[None]);self.cache[key]=arrays
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def maps(self,row):
        for key in ('capillary','large','faz'):self.identify(row,key)
        key=row['case_id']
        if key not in self.labels:
            masks=[binary_mask(row[k],self.cfg['data']['shape'][1:],row['annotation_orientation']) for k in ('capillary','large','faz')]
            self.labels[key]=structure_maps(*masks,self.cfg['patch']['apparent_diameter_px'])
            if len(self.labels)>16:self.labels.popitem(last=False)
        self.labels.move_to_end(key);return self.labels[key]


def patient_schedule(cfg,epoch,rows):
    rng=np.random.default_rng(seed_for(cfg['seed'],'patients',epoch))
    perm=rng.permutation(len(rows));times=rng.random(len(rows)).astype(np.float32)
    times=np.clip(times,np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(1),np.float32(0)))
    times[rng.permutation(len(rows))[:round(len(rows)*cfg['train']['exact_zero_fraction'])]]=0
    for i in range(0,len(rows),2):yield [(rows[int(k)],float(t)) for k,t in zip(perm[i:i+2],times[i:i+2])]


def patch_pair(cfg,maps,step,case_id):
    rng=np.random.default_rng(seed_for(cfg['seed'],'patches',step,case_id))
    shape=cfg['data']['shape'];core=cfg['patch']['core']
    uniform=uniform_tile(shape,core,rng)
    available={k:w for k,w in cfg['patch']['target_mixture'].items() if maps[k].any()}
    if not available:return [uniform,uniform],'uniform_fallback'
    keys=list(available);prob=np.array(list(available.values()),float);prob/=prob.sum()
    label=keys[int(rng.choice(len(keys),p=prob))]
    anchor=np.argwhere(maps[label]);y,x=anchor[int(rng.integers(len(anchor)))]
    target=(0,int(np.clip(y-core[1]//2,0,max(0,shape[1]-core[1]))),int(np.clip(x-core[2]//2,0,max(0,shape[2]-core[2]))))
    return [uniform,target],label


def statistics(dataset,cfg):
    path=dataset.run/'normalization.json'
    identity={'split_sha256':digest(dataset.run/'split_lock.json'),'shape':cfg['data']['shape']}
    if path.exists():
        result=read(path)
        if result['identity']!=identity:raise ValueError('Frozen statistics do not match input binding.')
        return result
    sq=[0.,0.];counts=[0,0];label_counts={k:0 for k in ('capillary','large','faz')}
    for row in dataset.split('train'):
        _,target,layers=dataset.case(row)
        mip,valid=slab(target,layers,native_depth=cfg['data']['native_layer_depth'])
        for i,a in enumerate((target.numpy().ravel(),mip[valid].numpy().ravel())):
            sq[i]+=float(np.einsum('i,i->',a,a,dtype=np.float64));counts[i]+=len(a)
        masks=dataset.maps(row)
        for k in label_counts:label_counts[k]+=int(masks[k].sum())
    if not label_counts['capillary'] or not label_counts['large']:
        raise ValueError('Decoded training labels contain no capillary or large-vessel foreground; review palette/version.')
    result={'identity':identity,'volume_rms':max((sq[0]/counts[0])**.5,1/255),
            'mip_rms':max((sq[1]/counts[1])**.5,1/255),'counts':counts,'annotation_counts':label_counts,'training_only':True}
    write(path,result);return result

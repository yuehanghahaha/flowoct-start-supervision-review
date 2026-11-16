"""Experiment paths, identities, author options, and full resume state."""
import contextlib
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT/'source/author'
CONFIG = json.loads((ROOT/'experiment.json').read_text('utf-8'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'weights'))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.path.insert(0, str(AUTHOR))


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf-8')
    tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_identity():
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+[ROOT/'experiment.json',ROOT/'membership.json']
    files += [p for p in AUTHOR.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}


def strict_precision():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    torch.set_default_dtype(torch.float32)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


@contextlib.contextmanager
def preserve_rng():
    state=rng_state()
    try:yield
    finally:restore_rng(state)


def options():
    from options.train_options import TrainOptions
    args=['--dataroot',str(ROOT/'data/OCT2OCTA3M_3D'),'--name','author',
          '--checkpoints_dir',str(ROOT/'checkpoints'),'--model','XOCT','--add_scale',
          '--direction','AtoB','--base_ch_g',str(CONFIG['base_ch_g']),'--max_ch_g',str(CONFIG['max_ch_g']),
          '--lambda_L1_3D','10','--lambda_L1_2D','10','--lambda_Perceptual_2D','1',
          '--dataset_mode','alignedoct2octaall','--preprocess','crop','--norm','group','--pool_size','0',
          '--crop_size','96','--batch_size','1','--input_nc','1','--output_nc','1',
          '--gpu_ids','0','--no_flip','--save_epoch_freq','50','--lr',str(CONFIG['learning_rate']),
          '--n_epochs','200','--n_epochs_decay','100','--num_threads',str(CONFIG['num_threads']),
          '--seed',str(CONFIG['seed']),'--no_html','--display_id','-1']
    previous=sys.argv;sys.argv=[str(AUTHOR/'train3d.py')]+args
    try:opt=TrainOptions().parse()
    finally:sys.argv=previous
    return opt


NETS=('G','D_3d','D_ilm_opl','D_opl_bm','D_other')


def build_model(opt):
    from models import create_model
    model=create_model(opt);model.setup(opt);strict_precision()
    return model


def checkpoint(model, epoch, steps, best, path, identity):
    import torch
    value=dict(epoch=epoch,steps=steps,best=best,time=now(),config=CONFIG,source_identity=identity,
               data_identity_sha256=digest(ROOT/'verification/data_identity.json'),
               nets={n:getattr(model,'net'+n).state_dict() for n in NETS},
               optimizers=[o.state_dict() for o in model.optimizers],
               schedulers=[s.state_dict() for s in model.schedulers],rng=rng_state())
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');torch.save(value,temp);temp.replace(path)


def restore(model,path,identity):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=False)
    assert value['config']==CONFIG and value['source_identity']==identity, 'Configuration/source changed'
    assert value['data_identity_sha256']==digest(ROOT/'verification/data_identity.json'), 'Input identity record changed'
    for n in NETS:getattr(model,'net'+n).load_state_dict(value['nets'][n])
    for o,state in zip(model.optimizers,value['optimizers']):o.load_state_dict(state)
    for s,state in zip(model.schedulers,value['schedulers']):s.load_state_dict(state)
    restore_rng(value['rng'])
    return {k:value[k] for k in ['epoch','steps','best']}


def environment():
    import importlib.metadata
    import platform
    import subprocess
    import torch
    return dict(time=now(),python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','scipy','scikit-image','torchio','Pillow']},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        tensor_default_dtype=str(torch.get_default_dtype()),amp=False,grad_scaler=False,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        gpu_state=subprocess.check_output(['nvidia-smi'],text=True))

"""Frozen XOCT ablation protocol, server inputs and resume state."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT / 'source/author'
RUNS = ROOT / 'runs'
VERIFY = ROOT / 'verification'
CKPT = ROOT / 'checkpoints'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


CFG = read(ROOT / 'experiment.json')
DEPLOY = read(ROOT / 'deployment.json')
BASE = Path(DEPLOY['reference_root'])
DATA = Path(DEPLOY['data_root'])
CDS = CFG['cds']
NETS = ('G', 'D_3d', 'D_ilm_opl', 'D_opl_bm', 'D_other') if CDS else ('G', 'D')
sys.path.insert(0, str(AUTHOR))


def source_contract():
    author_hashes = read(ROOT / 'source/author_files.json')
    for name, digest in author_hashes.items():
        if sha(AUTHOR / name) != digest:
            raise ValueError('Published XOCT file differs from the retained commit: ' + name)
    local = sorted(ROOT.glob('*.py')) + sorted(ROOT.glob('*.sh')) + [
        ROOT / name for name in ('experiment.json', 'deployment.json', 'membership.json', 'requirements.txt',
                                  'source/author_files.json', 'source/schedule.csv', 'source/input_identity.json')]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in local}


def configure():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    os.environ['TORCH_HOME'] = DEPLOY['torch_home']
    os.environ['MPLCONFIGDIR'] = str(RUNS / 'matplotlib')


def versions():
    return {p: importlib.metadata.version(p) for p in
            ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-image', 'Pillow', 'torchio', 'natsort')}


def environment():
    return dict(python=sys.version, executable=sys.executable, versions=versions(),
                gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                tensor_dtype=str(torch.get_default_dtype()), amp=False, scaler=False,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def seed_all():
    random.seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    torch.cuda.manual_seed_all(CFG['seed'])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def operation_lock():
    import fcntl
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / 'operation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This experiment already has an active operation') from exc
        yield


def options():
    from options.train_options import TrainOptions

    args = ['--dataroot', str(DATA / 'OCT2OCTA3M_3D'), '--name', 'author',
            '--checkpoints_dir', str(CKPT), '--model', CFG['author_model'],
            '--direction', 'AtoB', '--base_ch_g', '64', '--max_ch_g', '512',
            '--lambda_L1_3D', '10', '--dataset_mode', 'alignedoct2octaall',
            '--preprocess', 'crop', '--norm', 'group', '--pool_size', '0',
            '--crop_size', '96', '--batch_size', '1', '--input_nc', '1', '--output_nc', '1',
            '--gpu_ids', '0', '--no_flip', '--lr', '0.0001', '--n_epochs', '200',
            '--n_epochs_decay', '100', '--num_threads', '4', '--seed', str(CFG['seed']),
            '--no_html', '--display_id', '-1']
    if CDS:
        args.extend(['--lambda_L1_2D', '10', '--lambda_Perceptual_2D', '1'])
    else:
        args.append('--add_scale')
    previous = sys.argv
    sys.argv = [str(AUTHOR / 'train3d.py'), *args]
    try:
        return TrainOptions().parse()
    finally:
        sys.argv = previous


def create_model():
    from models import create_model as author_model

    seed_all()
    opt = options()
    model = author_model(opt)
    model.setup(opt)
    return model


def states(model):
    return {name: getattr(model, 'net' + name).state_dict() for name in NETS}


def model_digest(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize(model):
    # Carry over exactly the random tensors in the intact epoch-zero checkpoint.
    path = BASE / 'checkpoints/initial.pt'
    if sha(path) != DEPLOY['parent_initial_sha256']:
        raise ValueError('The reference initial checkpoint fingerprint changed')
    parent = torch.load(path, map_location='cpu', weights_only=False)
    assert parent['epoch'] == parent['steps'] == 0
    assert parent['config']['seed'] == CFG['seed']
    reference = parent['nets']
    reused = []
    if CDS:
        g = model.netG.state_dict()
        for name, tensor in reference['G'].items():
            if not name.startswith('generator_3d.'):
                if g[name].shape != tensor.shape:
                    raise ValueError('2D feature head layout differs: ' + name)
                g[name] = tensor
        model.netG.load_state_dict(g)
        for name in NETS[1:]:
            getattr(model, 'net' + name).load_state_dict(reference[name])
            assert model_digest(getattr(model, 'net' + name).state_dict()) == model_digest(reference[name])
        reused = ['2D feature heads', *NETS[1:]]
    else:
        prefix = 'generator_3d.'
        backbone = {name.removeprefix(prefix): tensor for name, tensor in reference['G'].items()
                    if name.startswith(prefix)}
        model.netG.load_state_dict(backbone)
        model.netD.load_state_dict(reference['D_3d'])
        assert model_digest(model.netG.state_dict()) == model_digest(backbone)
        assert model_digest(model.netD.state_dict()) == model_digest(reference['D_3d'])
        reused = ['3D generator', '3D discriminator']
    assert all(not optimizer.state for optimizer in model.optimizers)
    return dict(parent_initial_sha256=DEPLOY['parent_initial_sha256'], parent_epoch=0,
                reused_random_initialization=reused, trained_weights_loaded=False, adam_states_empty=True,
                network_tensor_sha256={name: model_digest(state) for name, state in states(model).items()})


def load_checkpoint(path, model, contract):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['contract'] != contract:
        raise ValueError('Checkpoint configuration, inputs or environment differs')
    for name in NETS:
        getattr(model, 'net' + name).load_state_dict(value['nets'][name])
    for optimizer, state in zip(model.optimizers, value['optimizers'], strict=True):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(model.schedulers, value['schedulers'], strict=True):
        scheduler.load_state_dict(state)
    restore_rng(value['rng'])
    return value['epoch'], value['best'], value['record']

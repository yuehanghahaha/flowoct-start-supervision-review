"""Bind A800 inputs to the fixed validation set and freeze trusted checkpoint tensors."""
import json
import hashlib
import io
import os
import subprocess

import numpy as np
import torch

from fm3d.data import load_pair
from shared import (ROOT, BASE, DATA, OWNED, XOCT, HPC_PARENT, GPU_UUID,
                    bounded, config, input_paths, model_weights, now, read, sha, validation_members, write)


def main():
    cfg = config()
    if ROOT.parent != BASE or ROOT.name != cfg['id'] or not ROOT.is_relative_to(OWNED):
        raise ValueError('Use the independent A800 experiment directory')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip()
    if gpu != GPU_UUID + ', NVIDIA A800 80GB PCIe':
        raise ValueError(f'A800 identity mismatch: {gpu}')
    frozen_path = ROOT / 'verification/frozen.json'
    if frozen_path.exists():
        raise FileExistsError('Frozen input record already exists')
    members = validation_members(cfg)
    parent = json.loads(read(ROOT / 'source/historical/parent_validation_inputs.json'))
    parent_inputs = parent['parent_validation_inputs']
    historic = json.loads(read(ROOT / 'source/historical/exp163_selected.json'))
    faz = json.loads(read(ROOT / 'source/historical/faz_conversion.json'))
    frozen = dict(time=now(), host=os.uname().nodename, cwd=str(ROOT), device=gpu,
                  manifest_sha256=sha(bounded(DATA / 'split_3M_z256.csv')),
                  membership_sha256=sha(ROOT / 'membership.json'), inputs={}, models={})
    for cid, member in members.items():
        paths = input_paths(cid)
        digests = {key: sha(path) for key, path in paths.items()}
        parent_paths = dict(pair=f'/workspace/flowoct_work/data/OCTA500_v1/z256/3M/{cid}.npz',
                            layers=f'/workspace/experience/{cfg["parent"]}/annotations/GT_Layers/{cid}.mat',
                            capillary=f'/workspace/experience/{cfg["parent"]}/annotations/GT_Capillary/{cid}.bmp',
                            large=f'/workspace/experience/{cfg["parent"]}/annotations/GT_LargeVessel/{cid}.bmp')
        parity = {key: digests[key] == parent_inputs[parent_path]['sha256'] for key, parent_path in parent_paths.items()}
        if not parity['layers'] or not parity['capillary'] or not parity['large']:
            raise ValueError(f'Parent label bytes differ for {cid}: {parity}')
        if not parity['pair']:
            # Check the exact decoded float32 arrays against the parent's immutable cache.
            pair = load_pair(paths['pair'], 'ZYX', 255., cfg['shape'][0])
            for key, array in zip(('oct', 'octa'), pair):
                candidates = [value['sha256'] for name, value in parent_inputs.items()
                              if name.endswith(f'/{cid}/{key}.npy')]
                stream = io.BytesIO()
                np.save(stream, array.numpy(), allow_pickle=False)
                if candidates != [hashlib.sha256(stream.getbuffer()).hexdigest()]:
                    raise ValueError(f'Decoded canonical image differs from parent: {cid}/{key}')
        if digests['faz2d'] != parent['labels'][cid]['GT_FAZ']['sha256'] or digests['faz3d'] != faz[cid]['output_sha256']:
            raise ValueError(f'FAZ annotation changed in transit for {cid}')
        frozen['inputs'][cid] = dict(**member, paths={k: str(v) for k, v in paths.items()},
                                     sha256=digests, matches_hpc_file=parity, decoded_pair_matches_hpc=True)
    reference = bounded(XOCT / 'checkpoints/best.pt')
    value = torch.load(reference, map_location='cpu', weights_only=False, mmap=True)
    if value['epoch'] != 213 or value['best']['epoch'] != 213:
        raise ValueError('Unexpected original XOCT best checkpoint')
    xoct_file = ROOT / 'checkpoints' / cfg['models']['XOCT_E213']['file']
    tmp = xoct_file.with_suffix('.tmp')
    torch.save(dict(model=value['nets']['G'], epoch=213), tmp)
    tmp.replace(xoct_file)
    xoct_selection = value['best']
    del value
    for tag, spec in cfg['models'].items():
        path = ROOT / 'checkpoints' / spec['file']
        digest = sha(path)
        if tag == 'XOCT_E213':
            origin = str(reference)
        else:
            origin = f'{HPC_PARENT}/checkpoints/{spec["file"]}'
            if digest != spec['sha256']:
                raise ValueError(f'Parent checkpoint bytes differ: {tag}')
        model, tensor_sha = model_weights(tag, cfg)
        if tag != 'XOCT_E213' and tensor_sha != historic[str(spec['epoch'])]['model_tensor_sha256']:
            raise ValueError(f'Checkpoint tensors differ from original validation: {tag}')
        frozen['models'][tag] = dict(path=path.relative_to(ROOT).as_posix(), origin=origin, epoch=spec['epoch'],
                                     sha256=digest, tensor_sha256=tensor_sha,
                                     parameters=sum(p.numel() for p in model.parameters()))
        if tag == 'XOCT_E213':
            frozen['models'][tag].update(origin_sha256=sha(reference), selection=xoct_selection)
        del model
    frozen['code_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in
                             sorted([*ROOT.glob('*.py'), *ROOT.glob('*.sh'), ROOT / 'experiment.json',
                                     *ROOT.glob('fm3d/*.py'), *ROOT.glob('source/xoct/*.py')])}
    write(frozen_path, frozen)
    print('Frozen five models and ten validation inputs', flush=True)


if __name__ == '__main__':
    main()

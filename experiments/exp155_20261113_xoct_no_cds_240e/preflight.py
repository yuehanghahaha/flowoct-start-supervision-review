"""Check real training updates and whole-grid output before opening validation."""
import argparse
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from common import (CDS, CFG, DATA, DEPLOY, VERIFY, configure, create_model, environment, initialize,
                    model_digest, now, operation_lock, sha, source_contract, versions, write)
from evaluate import validate
from inputs import Inputs
from run import update


def same_crop(inputs, cid, y, x):
    from data.alignedoct2octaall_dataset import AlignedOCT2OCTAALLDataset

    opt = SimpleNamespace(dataroot=str(DATA/'OCT2OCTA3M_3D'), direction='AtoB', input_nc=1, output_nc=1,
                          preprocess='crop', crop_size=96, no_flip=True)
    dataset = AlignedOCT2OCTAALLDataset(opt, 'train')
    crop = dict(crop_pos=(0,0,y,208-y,x,208-x), flip=False)
    with patch('data.alignedoct2octaall_dataset.get_params_3D', return_value=crop):
        original = dataset[dataset.pids.index(cid)]
    faster, _, _, _ = inputs.batch(cid, y, x, cds=True)
    assert all(torch.equal(faster[key],original[key]) for key in ('A','B','seg'))


def main(check_hashes):
    configure()
    contract = dict(source=source_contract(), versions=versions(), config=CFG)
    inputs = Inputs(verify_hashes=check_hashes)
    assert set(inputs.schedule) == set(range(1,CFG['epochs']+1))
    weights = DEPLOY['torch_home'] + '/hub/checkpoints/vgg19-dcbb9e9d.pth'
    assert sha(weights) == DEPLOY['vgg19_sha256']
    first = inputs.schedule[1][0]
    same_crop(inputs, *first)
    work = VERIFY / f'whole_grid_{time.time_ns()}'
    model = create_model()
    initial = initialize(model)
    names = ['netD_3d','netD_ilm_opl','netD_opl_bm','netD_other'] if CDS else ['netD']
    assert all(hasattr(model,name) for name in names)
    assert CDS or not hasattr(model,'criterionPerceptual')
    generator = model.netG.generator_3d if CDS else model.netG
    parameters = sum(parameter.numel() for parameter in generator.parameters())
    assert parameters == (64_638_145 if CDS else 52_615_169)
    fingerprints = {name:model_digest(getattr(model,name).state_dict()) for name in ['netG',*names]}
    times = []
    torch.cuda.reset_peak_memory_stats()
    for step,(cid,y,x) in enumerate(inputs.schedule[1][:7]):
        data, _, _, _ = inputs.batch(cid,y,x,CDS)
        data = {key:t[None] for key,t in data.items()}
        torch.cuda.synchronize()
        start = time.monotonic()
        losses = update(model,data)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        if step:
            times.append(elapsed)
        if step == 0:
            for net in [generator, *(getattr(model,name) for name in names)]:
                gradients = [p.grad for p in net.parameters() if p.grad is not None]
                assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
            if CDS:
                assert sum(losses[name] for name in losses if name.startswith('G_Perceptual')) > 0
                assert sum(losses[name] for name in losses if name.startswith('G_L1_3d2d')) > 0
            else:
                assert all('3d2d' not in name and 'Perceptual' not in name for name in losses)
        print(f'Preflight training case {cid}: {elapsed:.2f}s G_L1={losses["G_L1"]:.6f}',flush=True)
    assert all(model_digest(getattr(model,name).state_dict()) != digest for name,digest in fingerprints.items())
    allocated = torch.cuda.max_memory_allocated()/2**30
    reserved = torch.cuda.max_memory_reserved()/2**30
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = validate(model,inputs,0,output=work,training_case=first[0])
    assert len(result['cases']) == 1
    assert len(result['patient_macro']['standard']) == 9
    assert all(result['patient_macro']['standard'][group]['ssim'] is not None for group in result['patient_macro']['standard'])
    report = dict(passed=True, time=now(), contract=contract, environment=environment(),
                  initial=initial, generator_3d_parameters=parameters, exact_author_crop_checked=True,
                  training_cases=[row[0] for row in inputs.schedule[1][:7]], only_training_cases_in_preflight=True,
                  train_peak_allocated_gib=allocated, train_peak_reserved_gib=reserved,
                  validation_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  median_training_update_seconds=statistics.median(times), finite_gradients_and_updates=True,
                  whole_grid_report=f'verification/{work.name}/metrics.json',
                  all_input_hashes_verified_here=check_hashes, test_used=False)
    write(VERIFY/'preflight.json',report)
    print(f'Preflight passed. Training reserved {reserved:.2f} GiB; '
          f'whole-grid reserved {report["validation_peak_reserved_gib"]:.2f} GiB.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-input-hashes', action='store_true')
    with operation_lock():
        main(parser.parse_args().verify_input_hashes)

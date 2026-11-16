"""Train the published XOCT ablation with paired crops and resumable validation."""
import argparse
import json
import math
import os
import signal
import statistics
import time
import traceback

import torch
from torch.utils.data import DataLoader

from common import (CDS, CFG, CKPT, ROOT, RUNS, VERIFY, configure, create_model, environment, initialize,
                    load_checkpoint, model_digest, now, operation_lock, read, rng_state, save,
                    source_contract, states, versions, write)
from evaluate import validate
from inputs import EpochCrops, Inputs

LATEST = CKPT / 'latest.pt'
BEST = CKPT / 'best.pt'
STAGED_BEST = CKPT / 'best_candidate.pt'
PENDING = CKPT / 'trained_pending_validation.pt'


def status(state, epoch, step, completed, **extra):
    write(RUNS / 'status.json', dict(state=state, epoch=epoch, step=step, completed_epoch=completed,
                                    steps=(epoch-1)*140+step if epoch else 0,
                                    epochs=CFG['epochs'], pid=os.getpid(), time=now(), **extra))


def update(model, data):
    if CDS:
        model.set_input(data)
    else:
        # The published MSFF-only set_input also computes two unused validation projections.
        model.real_A = data['A'].to(model.device, dtype=torch.float32)
        model.real_B = data['B'].to(model.device, dtype=torch.float32)
    model.optimize_parameters()
    values = model.get_current_errors()
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError('Nonfinite optimization loss')
    return values


def train_epoch(model, inputs, epoch, best):
    model.update_learning_rate()
    lr = model.optimizers[0].param_groups[0]['lr']
    expected = 0.0001 * (1 - max(0, epoch + 1 - 200)/101)
    if not math.isclose(lr, expected, rel_tol=1e-12) or model.optimizers[1].param_groups[0]['lr'] != lr:
        raise ValueError(f'E{epoch} author learning rate differs: {lr} vs {expected}')
    tick = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(EpochCrops(inputs, epoch, CDS), batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True,
                        generator=torch.Generator().manual_seed(CFG['seed'] + epoch))
    status('training', epoch, 0, epoch-1, best=best)
    log = ROOT / 'logs/epochs' / f'epoch_{epoch:03d}_{time.time_ns()}.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    losses = []
    with log.open('x', encoding='utf-8', buffering=1) as stream:
        for step, (data, cid, y, x) in enumerate(loader, 1):
            tick_step = time.monotonic()
            if (cid[0],y.item(),x.item()) != inputs.schedule[epoch][step-1]:
                raise ValueError('Training crop order differs')
            values = update(model, data)
            losses.append(values)
            stream.write(json.dumps(dict(epoch=epoch, step=step, case_id=cid[0],
                                         origin_ZYX=[0,y.item(),x.item()], learning_rate=lr,
                                         seconds=time.monotonic()-tick_step, losses=values),
                                    allow_nan=False)+'\n')
            if step == 1 or step % 10 == 0:
                status('training', epoch, step, epoch-1, best=best)
                print(f'E{epoch}/{CFG["epochs"]} step {step}/140 '
                      f'G_L1={values["G_L1"]:.6f} G_GAN={values["G_GAN"]:.6f}',flush=True)
    if len(losses) != 140:
        raise RuntimeError('Incomplete patient ledger')
    return dict(epoch=epoch, steps=epoch*140, learning_rate=lr,
                mean_losses={key:statistics.fmean(row[key] for row in losses) for key in losses[0]},
                seconds=time.monotonic()-tick, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                ledger=log.relative_to(ROOT).as_posix())


def promote_best(best):
    if best is None:
        return
    for path in (BEST, STAGED_BEST):
        if path.exists():
            value = torch.load(path, map_location='cpu', weights_only=False)
            if value['selection'] == best:
                if path == STAGED_BEST:
                    STAGED_BEST.replace(BEST)
                return
    raise RuntimeError('Committed best model checkpoint is missing')


def checkpoint(model, contract, epoch, best, record):
    return dict(contract=contract, nets=states(model), epoch=epoch, steps=epoch*140, best=best,
                record=record, rng=rng_state(), optimizers=[o.state_dict() for o in model.optimizers],
                schedulers=[s.state_dict() for s in model.schedulers])


def train():
    configure()
    contract = dict(source=source_contract(), versions=versions(), config=CFG)
    preflight = read(VERIFY / 'preflight.json')
    if not preflight['passed'] or preflight['contract'] != contract:
        raise RuntimeError('The preflight belongs to different code, data or runtime')
    inputs = Inputs()
    model = create_model()
    completed, best = 0, None
    if LATEST.is_file():
        completed, best, record = load_checkpoint(LATEST, model, contract)
        if record is not None:
            write(RUNS / 'epochs' / f'epoch_{completed:03d}.json', record)
        promote_best(best)
        STAGED_BEST.unlink(missing_ok=True)
    else:
        initialization = initialize(model)
        write(VERIFY / 'initialization.json', initialization)
        save(LATEST, checkpoint(model,contract,0,None,None))
    pending = None
    if PENDING.is_file():
        # A stale pending file can exist if latest committed just before interruption.
        candidate = torch.load(PENDING, map_location='cpu', weights_only=False)
        pending_epoch = candidate['epoch']
        del candidate
        if pending_epoch <= completed:
            PENDING.unlink()
        elif pending_epoch == completed+1 and pending_epoch % CFG['validation_every'] == 0:
            _, saved_best, pending = load_checkpoint(PENDING, model, contract)
            if saved_best != best:
                raise ValueError('Pending validation refers to a different selection state')
        else:
            raise ValueError('Unexpected pending epoch')
    write(VERIFY / 'latest_launch.json', dict(time=now(), pid=os.getpid(), completed_epoch=completed,
                                            resumed=completed > 0, environment=environment()))
    for epoch in range(completed+1, CFG['epochs']+1):
        if pending is not None:
            record, pending = pending, None
        else:
            record = train_epoch(model, inputs, epoch, best)
            if epoch % CFG['validation_every'] == 0:
                save(PENDING, checkpoint(model,contract,epoch,best,record))
        improved = False
        if epoch % CFG['validation_every'] == 0:
            status('validating', epoch, 140, epoch-1, best=best)
            validation = validate(model, inputs, epoch)
            score = validation['patient_macro']['standard']['ilm_opl_max']['ssim']
            if score is None or not math.isfinite(score):
                raise FloatingPointError('Invalid validation selection metric')
            record['validation'] = validation['patient_macro']
            record['report'] = f'reports/validation/epoch_{epoch:03d}/metrics.json'
            if best is None or score > best['value']:
                best = dict(epoch=epoch, value=score, metric='standard_ilm_opl_max_ssim', report=record['report'],
                            generator_tensor_sha256=validation['generator_tensor_sha256'])
                save(STAGED_BEST, dict(model=model.netG.state_dict(), contract=contract, selection=best))
                improved = True
        record.update(best=best, time=now())
        save(LATEST, checkpoint(model,contract,epoch,best,record))
        if improved:
            STAGED_BEST.replace(BEST)
        write(RUNS / 'epochs' / f'epoch_{epoch:03d}.json', record)
        PENDING.unlink(missing_ok=True)
        status('epoch_committed', epoch, 140, epoch, best=best)
        print(f'E{epoch} committed in {record["seconds"]:.1f}s training time',flush=True)
    validation_epochs = list(range(CFG['validation_every'], CFG['epochs']+1, CFG['validation_every']))
    for epoch in validation_epochs:
        record = read(RUNS / 'epochs' / f'epoch_{epoch:03d}.json')
        if not (ROOT / record['report']).is_file():
            raise FileNotFoundError(record['report'])
    write(RUNS / 'training_complete.json', dict(epochs=CFG['epochs'], updates=CFG['epochs']*140,
          validation_epochs=validation_epochs, best=best, final_checkpoint='checkpoints/latest.pt', test_started=False, time=now()))
    status('complete', CFG['epochs'], 140, CFG['epochs'], best=best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train','status'])
    args = parser.parse_args()
    if args.action == 'status':
        path = RUNS / 'status.json'
        print(json.dumps(read(path) if path.exists() else {'state':'not_started'}, indent=2))
    else:
        def on_term(signum, frame):
            raise KeyboardInterrupt(f'Interrupted by signal {signum}')

        signal.signal(signal.SIGTERM, on_term)
        with operation_lock():
            try:
                train()
            except BaseException as exc:
                path = RUNS / 'status.json'
                current = read(path) if path.exists() else {}
                current.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', time=now(),
                               error=repr(exc))
                write(path,current)
                write(RUNS / f'failure_{time.time_ns()}.json', dict(time=now(), traceback=traceback.format_exc()))
                raise

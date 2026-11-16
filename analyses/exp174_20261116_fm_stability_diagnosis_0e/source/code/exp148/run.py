"""Bind fixed patients, train, resume, and report status for exp148."""
import argparse
from datetime import datetime
import json
import os
import signal
import sys
import time

import numpy as np
import torch

import common as c
import method
from evaluate import validate

LATEST = c.CKPT / 'latest.pt'
BEST = c.CKPT / 'best.pt'
CANDIDATE = c.CKPT / 'best_candidate.pt'
PENDING = c.CKPT / 'trained_pending_validation.pt'
HISTORY = c.RUN / 'history.json'
STATUS = c.RUN / 'status.json'


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def status(cfg, state, epoch, completed, updates, best, **details):
    c.write(STATUS, dict(state=state, epoch=epoch, completed_epoch=completed, updates=updates,
                         epochs=cfg['epochs'], best=best, pid=os.getpid(), time=timestamp(), **details))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def reconcile_best(selected):
    if selected is None:
        return
    for path in (BEST, CANDIDATE):
        if path.is_file() and load(path)['selection'] == selected:
            if path == CANDIDATE:
                CANDIDATE.replace(BEST)
            return
    raise RuntimeError('The committed validation-best checkpoint is missing')


def train_epoch(model, optimizer, inputs, rows, cfg, epoch, best):
    c.seed_all(cfg['seed'] + epoch)
    model.train()
    indices = np.random.default_rng(cfg['seed'] + epoch).integers(len(rows), size=cfg['steps_per_epoch'])
    times = c.time_schedule(cfg, epoch)
    steps = cfg['steps_per_epoch']
    updates = (epoch-1)*steps
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    log = c.RUN / f'steps_epoch_{epoch:04d}_{time.time_ns()}.jsonl'
    losses = []
    status(cfg, 'training', epoch, epoch-1, updates, best, step=0)
    with log.open('x', encoding='utf8', buffering=1) as stream:
        for step, (index, t) in enumerate(zip(indices, times), 1):
            row = rows[int(index)]
            source, target, origin = inputs.training_patch(row)
            values = method.update(model, optimizer, source, target, float(t), cfg)
            losses.append(values['fm_mse'])
            stream.write(json.dumps(dict(epoch=epoch, step=step, update=updates+step, case_id=row['case_id'],
                                         patient_id=row['patient_id'], origin_ZYX=origin, time=float(t), **values),
                                    allow_nan=False)+'\n')
            if step == 1 or step % 20 == 0 or step == steps:
                status(cfg, 'training', epoch, epoch-1, updates+step, best, step=step)
                print(f'E{epoch}/{cfg["epochs"]} step {step}/{steps} update={updates+step} '
                      f'FM MSE={values["fm_mse"]:.7f} velocity={values["velocity_dtype"]}', flush=True)
    return dict(epoch=epoch, updates=updates+steps, mean_fm_mse=float(np.mean(losses)),
                train_seconds=time.perf_counter()-tick,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                step_evidence=log.relative_to(c.ROOT).as_posix())


def train(cfg, deployment, rows, resume):
    c.configure(cfg)
    c.seed_all(cfg['seed'])
    model = method.make_model(cfg).to(cfg['device'])
    optimizer = method.make_optimizer(model, cfg)
    contract = dict(config=cfg, source=c.source_contract(), deployment=deployment,
                    runtime_versions=c.versions(), manifest_sha256=c.sha(deployment['manifest']),
                    precision=c.precision_flags(model))
    c.CKPT.mkdir(parents=True, exist_ok=True)
    c.RUN.mkdir(parents=True, exist_ok=True)
    history = c.read(HISTORY) if HISTORY.exists() else []
    best = None
    completed = 0
    pending = None
    verified = {}
    if resume:
        latest = load(LATEST)
        if latest['contract'] != contract:
            raise ValueError('Resume source, configuration, software or data binding changed')
        completed = latest['epoch']
        best = latest['best']
        reconcile_best(best)
        if PENDING.exists():
            trained = load(PENDING)
            if trained['epoch'] > completed:
                if trained['epoch'] != completed+1 or trained['best'] != best or trained['contract'] != contract:
                    raise ValueError('Pending validation does not match the committed epoch')
                pending = trained
            else:
                PENDING.unlink()
            del trained
        CANDIDATE.unlink(missing_ok=True)
        saved = pending if pending is not None else latest
        for path, entry in saved['input_identity'].items():
            if c.sha(path) != entry['sha256']:
                raise ValueError('Previously consumed input bytes changed: ' + path)
        verified = saved['input_identity']
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        c.restore_rng(saved['rng_state'])
        history = [row for row in history if row['epoch'] <= completed]
        if completed and (not history or history[-1]['epoch'] != completed):
            history.append(latest['record'])
            c.write(HISTORY, history)
        del saved, latest
    elif LATEST.exists() or PENDING.exists() or BEST.exists():
        raise FileExistsError('Started run exists. Use resume.')

    inputs = c.Inputs(deployment, cfg, verified)
    training = [row for row in rows if row['split'] == 'train']

    def checkpoint(epoch, record, selection):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng_state=c.rng_state(), contract=contract, epoch=epoch,
                    updates=epoch*cfg['steps_per_epoch'], record=record,
                    best=selection, input_identity=inputs.identities)

    if not resume:
        c.save_checkpoint(LATEST, checkpoint(0, None, None))
        c.write(c.VERIFY / 'initialization.json', dict(seed=cfg['seed'], external_checkpoint=None,
                                                      input_channels=model.stem.in_channels,
                                                      model_tensor_sha256=c.model_digest(model)))
    c.write(c.VERIFY / 'run_environment.json', dict(time=timestamp(), executable=sys.executable,
                python=sys.version, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(cfg['device']), contract=contract,
                parameters=sum(p.numel() for p in model.parameters())))
    for epoch in range(completed+1, cfg['epochs']+1):
        validate_epoch = epoch % cfg['val_every'] == 0 or epoch == cfg['epochs']
        if pending is None:
            record = train_epoch(model, optimizer, inputs, training, cfg, epoch, best)
            if validate_epoch:
                c.save_checkpoint(PENDING, checkpoint(epoch, record, best))
        else:
            record = pending['record']
            pending = None
        record.update(validation=None, report=None)
        improved = False
        if validate_epoch:
            status(cfg, 'validating', epoch, epoch-1, record['updates'], best)
            digest = c.model_digest(model)
            rng = c.rng_state()
            try:
                result, report = validate(model, inputs, rows, cfg, epoch, digest)
            finally:
                c.restore_rng(rng)
            record.update(validation=result['metrics'], report=report)
            metric = cfg['selection_metric']
            score = result['metrics'][metric]
            if score is None or not np.isfinite(score):
                raise FloatingPointError('Validation selection metric is invalid')
            improved = best is None or score > best['value']
            if improved:
                best = dict(epoch=epoch, metric=metric, value=score, report=report, model_tensor_sha256=digest)
                c.save_checkpoint(CANDIDATE, dict(model=model.state_dict(), epoch=epoch, selection=best,
                                                  contract=contract, input_identity=inputs.identities))

        record.update(best=best, time=timestamp())
        # The candidate can be promoted after an interruption once latest commits its selection.
        c.save_checkpoint(LATEST, checkpoint(epoch, record, best))
        if improved:
            CANDIDATE.replace(BEST)
        history.append(record)
        c.write(HISTORY, history)
        inputs.flush()
        PENDING.unlink(missing_ok=True)
        CANDIDATE.unlink(missing_ok=True)
        status(cfg, 'epoch_committed', epoch, epoch, record['updates'], best)
        print(f'Epoch {epoch} committed in {record["train_seconds"]:.1f}s training time; best={best}', flush=True)

    updates = cfg['epochs']*cfg['steps_per_epoch']
    c.write(c.RUN / 'training_complete.json',
            dict(state='complete', epochs=cfg['epochs'], updates=updates, best=best,
                 validation_epochs=[row['epoch'] for row in history if row['validation'] is not None],
                 time=timestamp()))
    status(cfg, 'complete', cfg['epochs'], cfg['epochs'], updates, best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['bind', 'train', 'resume', 'auto', 'status'])
    args = parser.parse_args()
    cfg = c.read(c.ROOT / 'experiment.json')
    if args.action == 'status':
        print(json.dumps(c.read(STATUS) if STATUS.exists() else {'state': 'not_started'}, indent=2))
        return
    deployment, rows = c.binding()
    if args.action == 'bind':
        print(json.dumps(dict(deployment=deployment, split_counts={'train': 140, 'validation': 10, 'test': 50}), indent=2))
        return
    with c.training_lock():
        resume = args.action == 'resume' or (args.action == 'auto' and LATEST.exists())
        train(cfg, deployment, rows, resume)


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Run interrupted; restart from the last saved checkpoint')

    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except KeyboardInterrupt as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(130)

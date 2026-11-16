"""Single-model optimization and cooperative recovery at completed updates."""
from copy import deepcopy
import hashlib
import math
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from dataset import schedule
from fm3d.flow import objective
from report import evaluate_split
from runtime import (load_checkpoint, now, pause_at_boundary, pause_requested, read_json,
                     save_checkpoint, sha256, stable_seed, write_json)


def make_optimizer(model, config):
    settings = config["optimizer"]
    groups = {"decay": [], "zero": []}
    for param in model.parameters():
        use_decay = param.ndim > 1 and not getattr(param, "_no_weight_decay", False)
        groups["decay" if use_decay else "zero"].append(param)
    return torch.optim.AdamW(
        [{"params": groups["decay"], "weight_decay": settings["weight_decay"]},
         {"params": groups["zero"], "weight_decay": 0.0}],
        lr=settings["peak_lr"], betas=tuple(settings["betas"]), eps=settings["eps"],
    )


def learning_rate(config, step):
    opt = config["optimizer"]
    warm = opt["warmup_updates"]
    if warm and step <= warm:
        return opt["peak_lr"] * step / warm
    total = config["epochs"] * config["steps_per_epoch"]
    progress = (step - warm) / max(1, total - warm)
    return opt["min_lr"] + 0.5 * (opt["peak_lr"] - opt["min_lr"]) * (1 + math.cos(math.pi * progress))


def auxiliary_weight(config, step):
    loss = config["loss"]
    return min(1.0, step / loss["auxiliary_ramp_updates"])


@torch.no_grad()
def update_ema(ema, model, decay):
    for running, live in zip(ema.parameters(), model.parameters()):
        running.lerp_(live, 1 - decay)


def optimization_step(model, optimizer, config, inputs, rows, times, step, device, aux=None,
                      microbatch_size=2, pairs=None):
    started = perf_counter()
    pairs = pairs if pairs is not None else [inputs.pair(row) for row in rows]
    factor = auxiliary_weight(config, step) if aux is None else aux
    lr = learning_rate(config, step)
    for group in optimizer.param_groups: group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    scalars, seeds = {}, []
    for begin in range(0, 2, microbatch_size):
        subset = pairs[begin:begin + microbatch_size]
        source, target, layers = [torch.cat([pair[k].to(device, non_blocking=True) for pair in subset])
                                  for k in range(3)]
        noise_parts = []
        for row in rows[begin:begin + microbatch_size]:
            seed = stable_seed(config['seed'], 'training_noise', step, row['case_id'])
            seeds.append(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            noise_parts.append(torch.randn((1, 8, *(n // 2 for n in source.shape[2:])),
                               device=device, dtype=torch.float32, generator=generator))
        noise = torch.cat(noise_parts)
        time = source.new_tensor(times[begin:begin + microbatch_size])
        loss, components = objective(model, source, target, layers, noise, time, None,
                                     config['loss'], config['native_layer_depth'], factor)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Nonfinite loss at update {step}.')
        (loss * (len(subset) / 2)).backward()
        for key, value in components.items():
            scalars[key] = scalars.get(key, 0.) + float(value.detach()) * len(subset) / 2
        del source, target, layers, noise, noise_parts, loss, components
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_norm'],
                                               error_if_nonfinite=True)
    optimizer.step()
    return dict(scalars, update=step, case_ids=[r['case_id'] for r in rows], times=times, noise_seeds=seeds,
                lr=lr, auxiliary_factor=factor, grad_norm=float(grad_norm),
                microbatch_size=microbatch_size, gradient_accumulation=2 // microbatch_size,
                seconds=perf_counter() - started)


def prefetched(inputs, batches):
    # One reader avoids concurrent LRU/identity writes. At most two batches are prefetched.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='paired-volume') as pool:
        pending = []
        iterator = iter(batches)
        def enqueue():
            try: rows, times = next(iterator)
            except StopIteration: return
            future = pool.submit(lambda rs=rows: [inputs.pair(r) for r in rs])
            pending.append((rows, times, future))
        enqueue(); enqueue()
        while pending:
            rows, times, future = pending.pop(0)
            pairs = future.result()
            enqueue()
            yield rows, times, pairs


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_status(root, state, **fields):
    write_json(root / "runs/status.json", dict(state=state, time=now(), **fields))


def train_epoch(root, model, ema, optimizer, config, inputs, epoch, device, signature, best, partial=None):
    model.train()
    start = perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    profile = read_json(root / 'runs/execution.json')
    model.checkpoint_mode = profile['checkpoint_mode']
    records = list(partial['steps']) if partial else []
    previous_updates = len(records)
    def result():
        record = dict(epoch=epoch, updates=len(records), last_update=records[-1]['update'],
                      mean_loss=float(np.mean([r['loss'] for r in records])),
                      seconds=perf_counter() - start + (partial['seconds'] if partial else 0), steps=records)
        for key, value in (('cuda_peak_allocated_bytes', torch.cuda.max_memory_allocated(device)),
                           ('cuda_peak_reserved_bytes', torch.cuda.max_memory_reserved(device))):
            record[key] = max(value, partial.get(key, 0) if partial else 0)
        return record
    batches = list(schedule(config, epoch, inputs.split('train')))[previous_updates:]
    for iteration, (rows, times, pairs) in enumerate(prefetched(inputs, batches), start=previous_updates + 1):
        if pause_requested(root):
            if records: break
            pause_at_boundary(root, 'training', committed_epoch=epoch - 1,
                              update=(epoch - 1) * config['steps_per_epoch'])
        step = (epoch - 1) * config['steps_per_epoch'] + iteration
        record = optimization_step(model, optimizer, config, inputs, rows, times, step, device,
                                   microbatch_size=profile['microbatch_size'], pairs=pairs)
        update_ema(ema, model, config['ema_decay'])
        records.append(record)
        if iteration == 1 or iteration % 20 == 0:
            set_status(root, 'training', epoch=epoch, update=step, committed_epoch=epoch - 1,
                       loss=record['loss'], last_update_seconds=record['seconds'], precision=config['precision'],
                       execution=profile, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            print(f"epoch {epoch:03d} step {iteration:03d} | loss={record['loss']:.6f} | {record['seconds']:.2f}s", flush=True)
        if iteration % 20 == 0 and iteration < config['steps_per_epoch']:
            save_checkpoint(root / 'checkpoints/latest.pt',
                snapshot(model, ema, optimizer, epoch - 1, config, device, signature, best, False, result()))
    record = result()
    if len(records) == config['steps_per_epoch']:
        write_json(root / f'runs/train/epoch_{epoch:03d}.json', record)
    return record


def snapshot(model, ema, optimizer, epoch, config, device, signature, best, awaiting_validation, partial=None):
    update = epoch * config["steps_per_epoch"] + (partial["updates"] if partial else 0)
    return dict(signature=signature, epoch=epoch, update=update, partial_epoch=partial,
                awaiting_validation=awaiting_validation, best=best, model=model.state_dict(),
                ema=ema.state_dict(), optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def commit_validation(root, config, inputs, ema, payload, device):
    epoch = payload["epoch"]
    set_status(root, "validating", epoch=epoch, update=payload["update"])

    def check_pause(patients):
        pause_at_boundary(root, "validation", epoch=epoch, update=payload["update"], patients=patients)

    summary = evaluate_split(ema, inputs, "validation", root / f"runs/validation/e{epoch:03d}",
                             config, device, model_digest(ema), check_pause=check_pause)
    score = summary["mean"]["ilm_opl_max"]["ssim"]
    best = payload["best"]
    if best is None or score > best["score"]:
        best = dict(epoch=epoch, update=payload["update"], score=score)
        save_checkpoint(root / "checkpoints/best.pt",
                        dict(signature=payload["signature"], selection=best, ema=ema.state_dict()))
        payload["best"] = best
    payload["awaiting_validation"] = False
    save_checkpoint(root / "checkpoints/latest.pt", payload)
    print(f"validation epoch {epoch:03d} | ILM-OPL SSIM {score:.6f} | best epoch {best['epoch']:03d}", flush=True)
    return best


def train(root, config, inputs, model, device, signature):
    optimizer = make_optimizer(model, config)
    ema = deepcopy(model).eval().requires_grad_(False)
    latest = root / "checkpoints/latest.pt"
    best, completed, partial = None, 0, None
    if latest.is_file():
        payload = load_checkpoint(latest)
        if payload["signature"] != signature:
            raise ValueError("Source, scientific settings, or input binding differ from the saved run.")
        model.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["rng_cuda"], device)
        completed, best = payload["epoch"], payload["best"]
        partial = payload.get("partial_epoch")
        if partial and (partial["epoch"] != completed + 1 or partial["updates"] != len(partial["steps"])
                        or not 0 < partial["updates"] < config["steps_per_epoch"]
                        or payload["update"] != completed * config["steps_per_epoch"] + partial["updates"]
                        or payload["update"] != partial["last_update"]
                        or payload["awaiting_validation"]):
            raise ValueError("Partial epoch metadata does not match the committed update.")
        if payload["awaiting_validation"]:
            best = commit_validation(root, config, inputs, ema, payload, device)
        del payload
    update = completed * config["steps_per_epoch"] + (partial["updates"] if partial else 0)
    pause_at_boundary(root, "training", committed_epoch=completed, update=update)
    for epoch in range(completed + 1, config["epochs"] + 1):
        record = train_epoch(root, model, ema, optimizer, config, inputs, epoch, device, signature, best, partial)
        optimizer.zero_grad(set_to_none=True)
        if record["updates"] < config["steps_per_epoch"]:
            payload = snapshot(model, ema, optimizer, epoch - 1, config, device, signature, best, False, record)
            save_checkpoint(latest, payload)
            pause_at_boundary(root, "training", epoch=epoch, committed_epoch=epoch - 1, update=payload["update"])
            raise RuntimeError("Partial epoch committed without a pause marker.")
        partial = None
        evaluate_now = epoch % config["validation_every"] == 0
        payload = snapshot(model, ema, optimizer, epoch, config, device, signature, best, evaluate_now)
        save_checkpoint(latest, payload)
        pause_at_boundary(root, "training", committed_epoch=epoch, update=payload["update"],
                          awaiting_validation=evaluate_now)
        if evaluate_now:
            best = commit_validation(root, config, inputs, ema, payload, device)
        del payload
    path = root / "checkpoints/best.pt"
    selected = load_checkpoint(path)
    if selected["selection"] != best or selected["signature"] != signature:
        raise ValueError("Best checkpoint does not match the committed validation history.")
    complete = dict(signature=signature, completed_epochs=config["epochs"],
                    updates=config["epochs"] * config["steps_per_epoch"], best=best,
                    best_sha256=sha256(path), validation_epochs=list(range(config["validation_every"],
                                                                           config["epochs"] + 1,
                                                                           config["validation_every"])))
    write_json(root / "runs/training_complete.json", complete)
    set_status(root, "training_complete", **{k: complete[k] for k in ("completed_epochs", "updates", "best")})
    return complete


def frozen_best(root, config, signature):
    path = root / "runs/training_complete.json"
    if not path.is_file():
        raise RuntimeError("The configured training budget and final validation must be committed first.")
    complete = read_json(path)
    best_path = root / "checkpoints/best.pt"
    if (complete["signature"] != signature or complete["completed_epochs"] != config["epochs"]
            or complete["updates"] != config["epochs"] * config["steps_per_epoch"]
            or sha256(best_path) != complete["best_sha256"]):
        raise ValueError("The training receipt or frozen checkpoint differs from this experiment.")
    selected = load_checkpoint(best_path)
    if selected["selection"] != complete["best"] or selected["signature"] != signature:
        raise ValueError("Selected weights do not match the completed training receipt.")
    return selected, complete

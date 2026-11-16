"""Bind data, inspect the model, check, train, test, or generate from OCT."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dataset import Inputs, bind, patient_noise, read_volume
from fm3d.model import WaveletVelocity
from metrics import evaluate
from report import evaluate_split, generate_case
from runtime import (PAUSE_EXIT, ROOT, PauseRequested, atomic_path, configure, load_checkpoint,
                     load_config, now, operation_lock, pause_at_boundary, read_json, sha256,
                     source_contract, write_json)
from train import frozen_best, make_optimizer, optimization_step, set_status, train, update_ema


def model_info(config, scan_factory):
    with torch.device('meta'):
        model = WaveletVelocity(config['model'], scan_factory)
    return dict(parameters=model.parameter_counts(), coefficient_grid_zyx=[128, 152, 152],
                global_token_grid_zyx=[32, 38, 38], global_tokens=46208,
                local_encoder_depths=[2, 2, 1], decoder_depths=[1, 2, 3],
                oct_channels=[32, 64, 96], state_channels=[48, 96, 128])


def signature_for(root=ROOT):
    return dict(sources=source_contract(root), deployment_sha256=sha256(root / 'runs/deployment.json'),
                normalization_sha256=sha256(root / 'runs/normalization.json'))


def full_check(config, binding, model, device, runtime, signature):
    import gc
    from verification.reference_scan import official_parity
    kernel = official_parity(device)
    from mamba_ssm import Mamba
    probe = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).to(device)
    x = torch.randn(2, 11, 32, device=device, requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = probe(x)
        loss = output.float().square().mean()
    loss.backward()
    if output.dtype != torch.bfloat16 or not torch.isfinite(x.grad).all():
        raise RuntimeError('Official Mamba BF16 forward/backward acceptance failed.')
    kernel['bf16_output_dtype'] = str(output.dtype)
    kernel['bf16_input_gradient_finite'] = True
    del probe, x, output, loss
    inputs = Inputs(binding, config, {'train'})
    rows = inputs.split('train')[:2]
    pairs = [inputs.pair(row) for row in rows]
    trials = []
    selected = None
    for microbatch, mode in ((2, 'none'), (2, 'selective'), (2, 'all'), (1, 'all')):
        pause_at_boundary(ROOT, 'gpu_check', diagnostic_updates=0, update=0)
        model.checkpoint_mode = mode
        optimizer = make_optimizer(model, config)
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        updates, durations = [], []
        started = perf_counter()
        try:
            for step in range(1, 26):
                pause_at_boundary(ROOT, 'gpu_check', diagnostic_updates=step-1, update=0)
                times = ([0., .5], [.125, .875], [.25, .75], [.9375, .99999])[((step - 1) % 4)]
                record = optimization_step(model, optimizer, config, inputs, rows, times, step, device,
                                           aux=1., microbatch_size=microbatch, pairs=pairs)
                torch.cuda.synchronize(device)
                updates.append(record)
                if step > 5: durations.append(record['seconds'])
                write_json(ROOT / 'verification/check_progress.json',
                    dict(time=now(), checkpoint_mode=mode, microbatch_size=microbatch, step=step,
                         loss=record['loss'], seconds=record['seconds'], formal_updates=0))
                print(f"full-size check B{microbatch} {mode} {step}/25 | {record['seconds']:.2f}s", flush=True)
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            limit = min(72 * 1024**3, runtime['memory_bytes'] - 8 * 1024**3)
            trial = dict(state='passed' if reserved <= limit else 'memory_headroom_failed',
                         checkpoint_mode=mode, microbatch_size=microbatch,
                         gradient_accumulation=2 // microbatch, effective_batch=2,
                         cuda_peak_allocated_bytes=allocated, cuda_peak_reserved_bytes=reserved,
                         warmup_updates=5, measured_updates=20, seconds=perf_counter() - started,
                         steady_median_seconds=float(np.median(durations)), updates=updates)
            trials.append(trial)
            if trial['state'] == 'passed':
                selected = {k: trial[k] for k in ('checkpoint_mode', 'microbatch_size', 'gradient_accumulation',
                                                  'effective_batch', 'steady_median_seconds')}
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(state='out_of_memory', checkpoint_mode=mode, microbatch_size=microbatch,
                               cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
        finally:
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
        write_json(ROOT / 'verification/profile_trials.json', trials)
        if selected is not None: break
    if selected is None:
        raise RuntimeError('Full science setup did not fit even B1 x2; do not shrink the protocol.')
    write_json(ROOT / 'runs/execution.json', selected)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    source = pairs[0][0]
    noise, seed = patient_noise(config, rows[0], (1, 8, 128, 152, 152))
    prediction, elapsed, clipped = generate_case(model, source, noise, config, device)
    quality, _, _, _, _ = evaluate(prediction, pairs[0][1][0, 0].numpy(), pairs[0][2][0].numpy())
    inference = dict(seconds=elapsed, seed=seed, clipped_fraction=clipped,
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    record = dict(state='passed', time=now(), signature=signature, runtime=runtime,
                  case_ids=[r['case_id'] for r in rows], split='train',
                  parameters=model.parameter_counts(), mamba_kernel_parity=kernel,
                  training_trials=trials, execution=selected, inference=inference,
                  entrypoint_metric_checks=quality, formal_updates=0)
    write_json(ROOT / 'verification/full_gpu.json', record)
    return record


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("bind")
    setup.add_argument("--manifest", required=True)
    setup.add_argument("--labels-root", required=True)
    setup.add_argument("--data-root")
    for name in ("model-info", "check", "train", "test", "status", "stats", "diagnostics"):
        commands.add_parser(name)
    inference = commands.add_parser("generate")
    inference.add_argument("--oct", required=True, type=Path)
    inference.add_argument("--seed", required=True, type=int)
    inference.add_argument("--name", required=True, help="A unique local output name, letters/numbers/_/-")
    inference.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best.pt")
    return parser.parse_args()


def new_model(config, device):
    environment = configure(config, device)
    torch.manual_seed(config["seed"])
    model = WaveletVelocity(config["model"]).to(device)
    model.load_statistics(read_json(ROOT / 'runs/normalization.json'))
    return model, environment


def generate_from_oct(args, config, device):
    if not args.name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.name):
        raise ValueError("Output name can contain letters, digits, underscores, or dashes.")
    output = ROOT / "runs/generate" / args.name
    if (output / "generation.json").exists():
        raise FileExistsError(output)
    selected = load_checkpoint(args.checkpoint)
    if selected["signature"]["sources"] != source_contract():
        raise ValueError("Checkpoint sources or scientific settings differ from this experiment.")
    model, environment = new_model(config, device)
    model.load_state_dict(selected["ema"])
    source = read_volume(args.oct, config, "oct")
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn((1, 8, *(n // 2 for n in source.shape[2:])), generator=rng)
    prediction, seconds, clipped = generate_case(model.eval(), source, noise, config, device)
    with atomic_path(output / "prediction.npy") as temp, temp.open("wb") as stream:
        np.save(stream, prediction, allow_pickle=False)
    write_json(output / "generation.json", dict(input=str(args.oct.resolve()), oct_sha256=sha256(args.oct),
               seed=args.seed, axis_order=config["axis_order"], shape=config["full_shape"],
               best=selected["selection"], checkpoint_sha256=sha256(args.checkpoint),
               prediction_sha256=sha256(output / "prediction.npy"), solver=config["solver"],
               seconds=seconds, clipped_fraction=clipped, runtime=environment))
    print(output / "prediction.npy")


def main():
    args = cli()
    config = load_config()
    if args.command == "model-info":
        from verification.reference_scan import ReferenceMamba

        print(json.dumps(model_info(config, ReferenceMamba), indent=2))
        return
    if args.command == "status":
        path = ROOT / "runs/status.json"
        print(json.dumps(read_json(path) if path.is_file() else {"state": "not_started"}, indent=2))
        return
    binding_path = ROOT / "runs/deployment.json"
    with operation_lock(ROOT / "runs/operation.lock"):
        if args.command == "bind":
            bound = bind(args.manifest, args.labels_root, args.data_root)
            print(f"Bound {len(bound['rows'])} fixed patient entries.")
            return
        if args.command == 'stats':
            from normalization import prepare
            stats = prepare(config, read_json(binding_path))
            print(json.dumps({k: v for k, v in stats.items() if k != 'inputs'}, indent=2))
            return
        device = torch.device(args.device)
        if args.command == "generate":
            generate_from_oct(args, config, device)
            return
        binding = read_json(binding_path)
        if (sha256(binding["manifest"]) != binding["manifest_sha256"]
                or sha256(ROOT / "membership.json") != binding["membership_sha256"]):
            raise ValueError("The input binding must match the fixed patient membership and manifest.")
        signature = signature_for()
        selected = completion = None
        if args.command in ("test", "diagnostics"):
            selected, completion = frozen_best(ROOT, config, signature)
        model, environment = new_model(config, device)
        write_json(ROOT / f"runs/environment/{args.command}.json", dict(time=now(), runtime=environment,
                   signature=signature, parameters=model.parameter_counts()))
        if args.command == "check":
            try:
                record = full_check(config, binding, model, device, environment, signature)
            except PauseRequested:
                raise
            except Exception as error:
                write_json(ROOT / "verification/full_gpu.json",
                           dict(state="failed", time=now(), signature=signature, runtime=environment,
                                error_type=type(error).__name__, error=str(error),
                                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
                raise
            print(json.dumps(record, indent=2))
        elif args.command == "train":
            inputs = Inputs(binding, config, {"train", "validation"})
            try:
                train(ROOT, config, inputs, model, device, signature)
            except PauseRequested:
                raise
            except KeyboardInterrupt:
                set_status(ROOT, "interrupted")
                raise
            except Exception as error:
                set_status(ROOT, "training_failed", error_type=type(error).__name__, error=str(error))
                raise
        elif args.command == 'diagnostics':
            from diagnostics import run_diagnostics
            model.load_state_dict(selected['ema'])
            inputs = Inputs(binding, config, {'validation'})
            run_diagnostics(ROOT, config, inputs, model.eval(), device, completion)
        elif args.command == "test":
            model.load_state_dict(selected["ema"])
            epoch = selected["selection"]["epoch"]
            inputs = Inputs(binding, config, {"test"})
            set_status(ROOT, "testing", checkpoint_epoch=epoch)

            def check_pause(patients):
                set_status(ROOT, "testing", checkpoint_epoch=epoch, patients=patients)
                pause_at_boundary(ROOT, "test", checkpoint_epoch=epoch, patients=patients,
                                  update=completion["updates"])

            try:
                summary = evaluate_split(model, inputs, "test", ROOT / f"runs/test/best_e{epoch:03d}",
                                         config, device, completion["best_sha256"], check_pause=check_pause)
            except PauseRequested:
                raise
            except KeyboardInterrupt:
                set_status(ROOT, "test_interrupted", checkpoint_epoch=epoch)
                raise
            except Exception as error:
                set_status(ROOT, "test_failed", checkpoint_epoch=epoch,
                           error_type=type(error).__name__, error=str(error))
                raise
            set_status(ROOT, "test_complete", checkpoint_epoch=epoch, patients=summary["patient_count"])
            print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except PauseRequested as error:
        set_status(ROOT, "paused", **error.progress)
        print(str(error), flush=True)
        raise SystemExit(PAUSE_EXIT) from None

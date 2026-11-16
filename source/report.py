"""Patient-level full-resolution evaluation, figures, and resumable evidence."""
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch

from dataset import patient_noise, patient_seed
from fm3d.flow import generate
from metrics import GROUPS, METRICS, PROTOCOL, evaluate, macro
from runtime import atomic_path, read_json, sha256, write_json


def generate_case(model, source, noise, config, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    raw = generate(model, source.to(device), noise.to(device), config["solver"]["steps"], config["infer_core_yx"])
    raw = raw[0, 0].cpu().numpy()
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite volume after inverse Haar.")
    elapsed = perf_counter() - started
    outside = float(np.count_nonzero((raw < 0) | (raw > 1)) / raw.size)
    np.clip(raw, 0.0, 1.0, out=raw)
    return raw, elapsed, outside


def adjacent_bscan_mae(prediction, reference, layers):
    """Y-neighbor intensity-difference error inside the shared ILM–BM support."""
    depth = reference.shape[0]
    centers = (np.arange(depth, dtype=np.float64) + 0.5) * (640 / depth) - 0.5
    error, count = 0.0, 0
    for start in range(0, depth, 16):
        slab = slice(start, start + 16)
        z = centers[slab, None, None]
        mask = (z >= layers[0] - 1) & (z <= layers[5] - 1)
        valid = mask[:, 1:] & mask[:, :-1]
        pred_delta = np.diff(prediction[slab], axis=1)
        true_delta = np.diff(reference[slab], axis=1)
        error += float(np.abs(pred_delta[valid] - true_delta[valid]).sum(dtype=np.float64))
        count += int(valid.sum())
    return error / count if count else None


def panel(path, case, prediction, reference, values):
    names = ("ilm_opl_max", "opl_bm_max", "ilm_bm_mean", "full_mean")
    labels = ("ILM-OPL max", "OPL-BM max", "ILM-BM mean", "Full Z mean")
    h, w = prediction[names[0]].shape
    cell_w, cell_h = max(w + 12, 154), h + 40
    canvas = Image.new("RGB", (4 * cell_w + 12, 2 * cell_h + 38), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{case}   |   matched windows   |   top reference, bottom prediction", fill="#142633")
    for col, (name, label) in enumerate(zip(names, labels)):
        upper = 0.5 if name.endswith("_max") else 0.15
        x = 12 + col * cell_w
        for row, maps in enumerate((reference, prediction)):
            y = 34 + row * cell_h
            array = np.uint8(np.clip(maps[name] / upper, 0, 1) * 255 + 0.5)
            canvas.paste(Image.fromarray(array), (x, y + 17))
            draw.text((x, y), f"{label}   [0,{upper:.2f}]", fill="#142633")
            if row:
                s = values[name]
                draw.text((x, y + 19 + h),
                          f"MAE {s['mae']:.4f}   PSNR {s['psnr']:.2f}   SSIM {s['ssim']:.4f}", fill="#142633")
    with atomic_path(path) as temp:
        canvas.save(temp, format="PNG")


def save_bscans(folder, prediction, truth, source):
    height = truth.shape[1]
    indices = [height // 4, height // 2, 3 * height // 4]
    arrays = {"y_indices": np.asarray(indices, dtype=np.int32),
              "oct_zx": np.stack([source[:, y, :] for y in indices]),
              "gt_zx": np.stack([truth[:, y, :] for y in indices]),
              "pred_zx": np.stack([prediction[:, y, :] for y in indices])}
    with atomic_path(folder / "bscans.npz") as temp, temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    depth, width = truth.shape[0], truth.shape[2]
    cell_w, cell_h = width + 12, depth + 35
    canvas = Image.new("RGB", (3 * cell_w + 12, 3 * cell_h + 12), "white")
    draw = ImageDraw.Draw(canvas)
    for row, y in enumerate(indices):
        for col, (key, label, upper) in enumerate((("oct_zx", "OCT", 1.0), ("gt_zx", "OCTA reference", 0.3),
                                                  ("pred_zx", "OCTA prediction", 0.3))):
            x0, y0 = 12 + col * cell_w, 12 + row * cell_h
            draw.text((x0, y0), f"Y={y} {label} [0,{upper:.1f}]", fill="#142633")
            image = np.uint8(np.clip(arrays[key][row] / upper, 0, 1) * 255 + 0.5)
            canvas.paste(Image.fromarray(image), (x0, y0 + 18))
    with atomic_path(folder / "bscans.png") as temp:
        canvas.save(temp, format="PNG")


def output_record(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_case(folder, case, prediction, truth, layers, receipt, retain_volume, source):
    folder.mkdir(parents=True, exist_ok=True)
    values, slices, prediction_maps, reference_maps, empty = evaluate(prediction, truth, layers)
    with atomic_path(folder / "projections.npz") as temp, temp.open("wb") as stream:
        np.savez_compressed(stream, **{f"pred_{k}": v for k, v in prediction_maps.items()},
                            **{f"gt_{k}": v for k, v in reference_maps.items()})
    panel(folder / "projections.png", case, prediction_maps, reference_maps, values)
    save_bscans(folder, prediction, truth, source)
    files = ["projections.npz", "projections.png", "bscans.npz", "bscans.png"]
    if retain_volume:
        with atomic_path(folder / "prediction.npy") as temp, temp.open("wb") as stream:
            np.save(stream, prediction, allow_pickle=False)
        files.append("prediction.npy")
    record = dict(receipt, case_id=case, metrics=values, bscans=slices, empty_projection_rays=empty,
                  adjacent_bscan_mae=adjacent_bscan_mae(prediction, truth, layers),
                  files={name: output_record(folder / name) for name in files})
    write_json(folder / "record.json", record)
    return record


def read_completed(folder, receipt, retain_volume):
    path = folder / "record.json"
    if not path.is_file():
        return None
    record = read_json(path)
    if any(record.get(key) != value for key, value in receipt.items()):
        raise ValueError(f"Saved evaluation belongs to a different weight, input, or solver: {path}.")
    expected = {"projections.npz", "projections.png", "bscans.npz", "bscans.png"}
    if retain_volume:
        expected.add("prediction.npy")
    if set(record["files"]) != expected:
        return None
    if any(not (folder / name).is_file() or output_record(folder / name) != info
           for name, info in record["files"].items()):
        return None
    return record


def write_tables(folder, records, summary):
    with atomic_path(folder / "patients.csv") as temp, temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case_id", "object", *METRICS))
        for record in records:
            for group in GROUPS:
                writer.writerow((record["case_id"], group, *(record["metrics"][group][name] for name in METRICS)))
    with atomic_path(folder / "summary.csv") as temp, temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("object", "metric", "mean", "patient_sd", "n"))
        for group in GROUPS:
            for metric in METRICS:
                writer.writerow((group, metric, summary["mean"][group][metric],
                                 summary["patient_sd"][group][metric], len(records)))


def evaluate_split(model, inputs, phase, folder, config, device, weight_identity, check_pause=None):
    model.eval()
    retain_volume = phase == "test" and config["retain_test_volumes"]
    records = []
    for row in inputs.split(phase):
        if check_pause is not None:
            check_pause(len(records))
        case = row["case_id"]
        folder_case = Path(folder) / case
        receipt = dict(phase=phase, weight_identity=weight_identity, solver=config["solver"],
                       inference_core=config["infer_core_yx"], image=inputs.identify(row, "image"),
                       noise_seed=patient_seed(config, row))
        if (folder_case / "record.json").is_file():
            receipt["layers"] = inputs.identify(row, "layers")
        record = read_completed(folder_case, receipt, retain_volume)
        if record is None:
            source = inputs.volume(row, "oct")
            noise, seed = patient_noise(config, row, (1, 8, *(n // 2 for n in source.shape[2:])))
            if seed != receipt["noise_seed"]:
                raise RuntimeError("Patient noise identity changed.")
            prediction, seconds, outside = generate_case(model, source, noise, config, device)
            truth = inputs.volume(row, "octa")[0, 0].numpy()
            layers = inputs.layers(row)
            receipt.update(layers=inputs.identify(row, "layers"), seconds=seconds, clipped_fraction=outside)
            record = write_case(folder_case, case, prediction, truth, layers, receipt, retain_volume,
                                source[0, 0].numpy())
            del source, noise, prediction, truth, layers
        records.append(record)
    if check_pause is not None:
        check_pause(len(records))
    values = [row["metrics"] for row in records]
    summary = dict(phase=phase, weight_identity=weight_identity, patient_count=len(records),
                   patient_ids=[row["case_id"] for row in records], protocol=PROTOCOL, solver=config["solver"],
                   mean=macro(values), retain_volumes=retain_volume,
                   mean_seconds=float(np.mean([row["seconds"] for row in records])),
                   patient_sd={group: {metric: float(np.std([row[group][metric] for row in values],
                                                             ddof=1 if len(values) > 1 else 0))
                                       for metric in METRICS} for group in GROUPS})
    write_tables(Path(folder), records, summary)
    write_json(Path(folder) / "summary.json", summary)
    return summary

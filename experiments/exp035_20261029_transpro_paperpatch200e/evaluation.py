"""Full-field prediction, all eight metrics, paper comparison and projection figures."""
import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import crop, unit
from experiment import read, sha, write
from metrics import NAMES, eight_metrics, aggregate


@torch.no_grad()
def predict_full(generator, source, device):
    generator.eval()
    prediction = np.zeros((256, 304, 304), dtype=np.float32)
    count = np.zeros((304, 304), dtype=np.float32)
    for y in (0, 48):
        for x in (0, 48):
            tensor = torch.from_numpy(crop(source, y, x)).to(device=device, dtype=torch.float32)
            output = generator(tensor)
            if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
                raise FloatingPointError("Inference output must be finite FP32")
            values = ((output[0, 0] + 1) / 2).cpu().numpy().transpose(1, 0, 2)
            prediction[:, y:y + 256, x:x + 256] += values
            count[y:y + 256, x:x + 256] += 1
            del tensor, output, values
    if np.any(count == 0):
        raise RuntimeError("Uncovered full-view voxels")
    prediction /= count[None]
    return prediction


def projection_figure(source_yx, views, cid, path, synthetic=False):
    arrays = [source_yx, views["target_yx"], views["prediction_yx"], views["absolute_error_yx"]]
    titles = ["OCT mean projection", "OCTA ground truth", "TransPro prediction", "Absolute error (0-0.2)"]
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    for index, (axis, array, title) in enumerate(zip(axes, arrays, titles)):
        axis.imshow(array.T, cmap="magma" if index == 3 else "gray", vmin=0, vmax=.2 if index == 3 else 1)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(("SYNTHETIC PIPELINE CHECK - " if synthetic else "") + str(cid) + " | anatomical-depth mean | native BMP orientation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def build_report(output, patients, summary, config, split, epoch):
    paper = config["reference"]["paper_table3M"]
    comparison = [{"metric": name, "paper_3M": paper[name], "current": summary["patient_mean"][name],
                   "difference": None if summary["patient_mean"][name] is None else summary["patient_mean"][name] - paper[name],
                   "valid_patients": summary["valid_patients"][name],
                   "comparison_status": "descriptive; author exact preprocessing/SSIM/mask identity not verified"} for name in NAMES]
    write_csv(output / "paper_comparison.csv", comparison, ("metric", "paper_3M", "current", "difference", "valid_patients", "comparison_status"))
    write_csv(output / "per_patient.csv", patients, ("case_id", "patient_id", *NAMES))
    table = ["| Metric | Paper 3M | Current | Current - paper | Valid patients |", "| --- | ---: | ---: | ---: | ---: |"]
    def display(value):
        return "undefined" if value is None else f"{float(value):.6g}"
    for row in comparison:
        table.append(f"| {row['metric']} | {display(row['paper_3M'])} | {display(row['current'])} | {display(row['difference'])} | {row['valid_patients']} |")
    chart, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for axis, row in zip(axes.flat, comparison):
        if row["current"] is not None and np.isfinite(row["current"]):
            axis.bar(["Paper 3M", "This protocol"], [row["paper_3M"], row["current"]], color=["#9aa4b2", "#176b75"])
        axis.set_title(row["metric"])
    chart.suptitle("Descriptive comparison: unpublished author details and mask identity remain unverified")
    chart.savefig(output / "paper_comparison.png", dpi=180)
    plt.close(chart)
    text = "\n".join([
        f"# TransPro {split}: epoch {epoch}", "", *table, "",
        "All eight names follow the paper; implementation details are fully recorded in experiment.json and PROTOCOL.md.",
        "Volume metrics are averages of 2D B-scans, then equal-patient averages. SSIM/VDC are displayed in percent.",
        "This comparison is descriptive: data_range/SSIM settings, public GT_Capillary identity, full-view stitching and original-depth preprocessing are not verified identical to the authors' experiment.",
        "Validation data select checkpoints; validation scores are not an independent replication of the paper's test table.",
        "Test results are generated only after freezing a completed training run. Do not use these scores to retune this experiment.",
        "Complete float32 predictions, per-B-scan scores, float projection maps and every patient's figure are preserved alongside this report.", ""])
    (output / "report.md").write_text(text, encoding="utf-8")
    images = "".join(f"<figure><img src='figures/{html.escape(row['case_id'])}.png'><figcaption>{html.escape(row['case_id'])}</figcaption></figure>" for row in patients)
    body = "".join(f"<tr><td>{row['metric']}</td><td>{display(row['paper_3M'])}</td><td>{display(row['current'])}</td><td>{display(row['difference'])}</td></tr>" for row in comparison)
    page = f"<!doctype html><meta charset='utf-8'><title>TransPro {split}</title><style>body{{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:9px}}img{{width:100%}}figure{{margin:35px 0}}p{{max-width:1000px;line-height:1.6}}</style><h1>TransPro {split} — epoch {epoch}</h1><p>八项指标与论文3M表1数值对照。浮点归一化、SSIM细节、公开GT_Capillary及全视野拼接尚未证明与作者完全一致；此表为描述性对照。验证集已用于选模。</p><table><tr><th>Metric</th><th>Paper 3M</th><th>Current</th><th>Difference</th></tr>{body}</table><img src='paper_comparison.png'>{images}"
    (output / "index.html").write_text(page, encoding="utf-8")


def evaluate(generator, inputs, split, config, device, epoch, output, save_predictions, protocol_id, checkpoint_hash=None, frozen=False):
    if split == "test" and not frozen:
        raise ValueError("Test evaluation requires a verified freeze record")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    patients, slice_rows = [], []
    if save_predictions:
        for name in ("predictions", "projections", "figures", "cases"):
            (output / name).mkdir(exist_ok=True)
    for row in inputs.split(split):
        cid = row["case_id"]
        record = output / "cases" / (cid + ".json")
        reuse = False
        if save_predictions and record.exists():
            saved = read(record)
            if saved["protocol_id"] != protocol_id or saved["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Existing case result belongs to different weights/protocol")
            for relative, digest in saved["files"].items():
                if sha(output / relative) != digest:
                    raise ValueError("Saved result changed: " + relative)
            patient, slices = saved["metrics"], saved["bscans"]
            reuse = True
        if not reuse:
            source, target = inputs.pair(row, allow_test=frozen)
            prediction = predict_full(generator, source, device)
            target_unit = unit(target[0])
            annotation = inputs.mask(row, allow_test=frozen)
            values, slices, views = eight_metrics(prediction, target_unit, annotation, config)
            patient = {"case_id": cid, "patient_id": row["patient_id"], **values}
            if save_predictions:
                pred_path = output / "predictions" / (cid + ".npz")
                np.savez_compressed(pred_path, prediction=prediction, axis_order=np.array("ZYX"),
                                    target_cache_minmax=np.array([target.min(), target.max()], dtype=np.float32))
                source_projection = unit(source[0].mean(axis=0))
                proj_path = output / "projections" / (cid + ".npz")
                np.savez_compressed(proj_path, oct_yx=source_projection, **views)
                figure_path = output / "figures" / (cid + ".png")
                projection_figure(source_projection, views, cid, figure_path)
                files = {str(path.relative_to(output)): sha(path) for path in (pred_path, proj_path, figure_path)}
                write(record, {"protocol_id": protocol_id, "checkpoint_sha256": checkpoint_hash,
                               "metrics": patient, "bscans": slices, "files": files,
                               "mask_source": config["vessel_mask_source"], "prediction_grid_zyx": [256, 304, 304]})
            del source, target, prediction, target_unit, annotation, views
        patients.append(patient)
        slice_rows.extend({"case_id": cid, "bscan_y": index, **values} for index, values in enumerate(slices))
        print(f"{split} {cid} MAE={patient['MAE']:.6f} PSNR={patient['PSNR']:.4f}", flush=True)
    if not patients:
        raise ValueError("Empty evaluation split")
    summary = aggregate(patients)
    summary.update(split=split, epoch=epoch, protocol_id=protocol_id, checkpoint_sha256=checkpoint_hash,
                   complete_float_predictions_saved=save_predictions, frozen_test=frozen,
                   strict_author_equivalence_verified=False, operational_settings=config)
    write(output / "summary.json", summary)
    write_csv(output / "per_bscan.csv", slice_rows, ("case_id", "bscan_y", "MAE", "PSNR", "SSIM"))
    if save_predictions:
        build_report(output, patients, summary, config, split, epoch)
    else:
        write_csv(output / "per_patient.csv", patients, ("case_id", "patient_id", *NAMES))
    return summary

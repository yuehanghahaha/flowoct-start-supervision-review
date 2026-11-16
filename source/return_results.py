"""Collect frozen results on the A100 and verify a downloaded analysis_return directory."""
import argparse
from contextlib import contextmanager
import csv
from pathlib import Path
import shutil

from a100 import ROOT, atomic_json, digest, read_json, stamp


@contextmanager
def atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def copy_file(source, target):
    with atomic_file(target) as temp:
        shutil.copyfile(source, temp)


def write_csv(path, headers, rows):
    with atomic_file(path) as temp, temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def identity(path):
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def verify_files(root, files):
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid evidence path: {name}")
        if identity(root / relative) != expected:
            raise ValueError(f"Evidence checksum differs: {root / relative}")


def verify(folder):
    report = read_json(folder / "test_complete.json")
    if report["state"] != "complete" or report["patient_count"] != report["planned_test_patients"]:
        raise ValueError("The frozen test is incomplete.")
    verify_files(folder, report["files"])
    return report


def histories(root, folder, config, validation_epochs, validation_patients):
    rows, patient_rows = [], []
    for epoch in range(1, config["epochs"] + 1):
        record = read_json(root / f"runs/train/epoch_{epoch:03d}.json")
        if record["updates"] != config["steps_per_epoch"] or record["last_update"] != epoch * config["steps_per_epoch"]:
            raise ValueError(f"Incomplete training log at epoch {epoch}.")
        rows.append((epoch, record["last_update"], record["mean_loss"], record["seconds"],
                     record.get("cuda_peak_allocated_bytes"), record.get("cuda_peak_reserved_bytes")))
        copy_file(root / f"runs/train/epoch_{epoch:03d}.json", folder / f"training_logs/epoch_{epoch:03d}.json")
    write_csv(folder / "training_history.csv", ("epoch", "update", "mean_loss", "seconds", "gpu_peak_allocated",
                                               "gpu_peak_reserved"), rows)
    rows = []
    for epoch in validation_epochs:
        source = root / f"runs/validation/e{epoch:03d}/summary.json"
        record = read_json(source)
        if record["patient_ids"] != validation_patients or record["solver"] != config["solver"]:
            raise ValueError(f"Validation membership or solver differs at epoch {epoch}.")
        copy_file(source, folder / f"validation_summaries/e{epoch:03d}.json")
        for name, values in record["mean"].items():
            rows.append((epoch, name, values["mae"], values["psnr"], values["ssim"]))
        for patient in validation_patients:
            case = read_json(source.parent / patient / "record.json")
            if case["case_id"] != patient or case["weight_identity"] != record["weight_identity"]:
                raise ValueError(f"Validation case identity differs at epoch {epoch}: {patient}")
            for name, metrics in case["metrics"].items():
                patient_rows.append((epoch, patient, name, metrics["mae"], metrics["psnr"], metrics["ssim"]))
    write_csv(folder / "validation_history.csv", ("epoch", "object", "mae", "psnr", "ssim"), rows)
    write_csv(folder / "validation_patient_metrics.csv", ("epoch", "case_id", "object", "mae", "psnr", "ssim"),
              patient_rows)


def collect(root=ROOT):
    folder = root / "analysis_return"
    if (folder / "test_complete.json").is_file():
        return verify(folder)
    config = read_json(root / "experiment.json")
    complete = read_json(root / "runs/training_complete.json")
    best = complete["best"]
    best_hash = complete["best_sha256"]
    binding = root / "runs/deployment.json"
    validation_epochs = list(range(config["validation_every"], config["epochs"] + 1, config["validation_every"]))
    if (complete["completed_epochs"] != config["epochs"]
            or complete["updates"] != config["epochs"] * config["steps_per_epoch"]
            or complete["validation_epochs"] != validation_epochs
            or best["epoch"] not in validation_epochs
            or digest(root / "checkpoints/best.pt") != best_hash
            or digest(binding) != complete["signature"]["deployment_sha256"]):
        raise ValueError("The final validation and frozen checkpoint do not match the current experiment.")
    for relative, expected in complete["signature"]["sources"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Formal source file changed: {relative}")
    members = read_json(root / "membership.json")
    patients = [row["case_id"] for row in members if row["split"] == "test"]
    validation_patients = [row["case_id"] for row in members if row["split"] == "validation"]
    test_dir = root / f"runs/test/best_e{best['epoch']:03d}"
    summary = read_json(test_dir / "summary.json")
    if (summary["patient_ids"] != patients or summary["patient_count"] != len(patients)
            or summary["weight_identity"] != best_hash or summary["solver"] != config["solver"]
            or sum(len(metrics) for metrics in summary["mean"].values()) != 27):
        raise ValueError("The test summary does not match the frozen best and fixed test patients.")
    if digest(root / 'runs/normalization.json') != complete['signature']['normalization_sha256']:
        raise ValueError('Frozen normalization differs from the trained model.')
    diagnostic_summary = read_json(root / 'runs/diagnostics/summary.json')
    if diagnostic_summary['weight_identity'] != best_hash or diagnostic_summary['patient_count'] != 10:
        raise ValueError('Post-selection validation diagnostics are incomplete.')
    for path in sorted((root / 'runs/diagnostics').rglob('*')):
        if path.is_file() and path.suffix != '.tmp':
            copy_file(path, folder / 'diagnostics' / path.relative_to(root / 'runs/diagnostics'))
    volumes, slice_rows, diagnostics = {}, [], []
    for patient in patients:
        case = test_dir / patient
        record = read_json(case / "record.json")
        if (record["case_id"] != patient or record["weight_identity"] != best_hash
                or record["solver"] != config["solver"] or record["phase"] != "test"
                or [row["y"] for row in record["bscans"]] != list(range(config["full_shape"][1]))):
            raise ValueError(f"Incomplete or mismatched frozen test result for {patient}.")
        expected = {"projections.npz", "projections.png", "bscans.npz", "bscans.png", "prediction.npy"}
        if set(record["files"]) != expected:
            raise ValueError(f"Missing full-resolution test evidence for {patient}.")
        verify_files(case, record["files"])
        for name in ("record.json", "projections.npz", "projections.png", "bscans.npz", "bscans.png"):
            copy_file(case / name, folder / "test_cases" / patient / name)
        relative = (case / "prediction.npy").relative_to(root).as_posix()
        volumes[relative] = record["files"]["prediction.npy"]
        slice_rows.extend((patient, scan["y"], scan["mae"], scan["psnr"], scan["ssim"])
                          for scan in record["bscans"])
        diagnostics.append((patient, record["noise_seed"], record["seconds"], record["clipped_fraction"],
                            record["adjacent_bscan_mae"]))
    write_csv(folder / "bscan_metrics.csv", ("case_id", "y", "mae", "psnr", "ssim"), slice_rows)
    write_csv(folder / "test_diagnostics.csv", ("case_id", "noise_seed", "seconds", "clipped_fraction",
                                               "adjacent_bscan_mae"), diagnostics)
    histories(root, folder, config, validation_epochs, validation_patients)
    validation_best = root / f"runs/validation/e{best['epoch']:03d}"
    best_summary = read_json(validation_best / "summary.json")
    if best_summary["mean"]["ilm_opl_max"]["ssim"] != best["score"]:
        raise ValueError("The selected score differs from the committed validation best.")
    for patient in validation_patients:
        selected = validation_best / patient
        for name in ("record.json", "projections.npz", "projections.png", "bscans.npz", "bscans.png"):
            copy_file(selected / name, folder / "validation_best" / patient / name)
    for source, target in (("summary.csv", "summary.csv"), ("summary.json", "summary.json"),
                           ("patients.csv", "patient_metrics.csv")):
        copy_file(test_dir / source, folder / target)
    for name in ("experiment.json", "membership.json", "sources.sha256", "provenance.json",
                 "verification/input_identities.json", "verification/full_gpu.json", "verification/correctness.json",
                 "runs/deployment.json", "runs/normalization.json", "runs/execution.json", "runs/training_complete.json",
                 "runs/environment/train.json", "runs/environment/test.json"):
        copy_file(root / name, folder / "provenance" / name)
    for relative in complete["signature"]["sources"]:
        copy_file(root / relative, folder / "source" / relative)
    copy_file(root / "analysis_return_README.md", folder / "README.md")
    atomic_json(folder / "server_volumes.json",
                dict(server_experiment=str(root), count=len(volumes), files=volumes, checkpoint_sha256=best_hash))
    files = {path.relative_to(folder).as_posix(): identity(path) for path in sorted(folder.rglob("*"))
             if path.is_file() and path.suffix != ".tmp"}
    receipt = dict(state="complete", time=stamp(), experiment=config["experiment_id"], patient_count=len(patients),
                   planned_test_patients=len(patients), bscan_rows=len(slice_rows), best=best,
                   best_sha256=best_hash, solver=config["solver"], validation_epochs=validation_epochs, files=files)
    atomic_json(folder / "test_complete.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "verify"))
    parser.add_argument("--folder", type=Path, default=ROOT / "analysis_return", help="Local folder for verify.")
    args = parser.parse_args()
    record = collect() if args.command == "collect" else verify(args.folder)
    print(f"{record['patient_count']} frozen test patients, {record['bscan_rows']} B-scans, "
          f"best E{record['best']['epoch']}, {len(record['files'])} files verified.")


if __name__ == "__main__":
    main()

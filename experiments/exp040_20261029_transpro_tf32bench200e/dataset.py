"""Fixed patient membership, existing read-only NPY caches, native mask orientation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from experiment import ROOT, read, sha, write


def unit(array):
    array = np.asarray(array, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    if not np.isfinite(array).all() or high <= low:
        raise ValueError("Non-finite or constant input; no silent normalization change")
    return (array - low) / (high - low)


def crop(array, y, x):
    if array.shape != (1, 256, 304, 304) or array.dtype != np.float32:
        raise ValueError("Expected a full float32 [1,256,304,304] cache")
    if not 0 <= y <= 48 or not 0 <= x <= 48:
        raise ValueError("Crop outside full view")
    value = array[0, :, y:y + 256, x:x + 256].transpose(1, 0, 2)
    return np.ascontiguousarray((2 * unit(value) - 1)[None, None])


def mask_yx(labels, cid):
    path = Path(labels) / "GT_Capillary" / (cid + ".bmp")
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.shape != (304, 304) or not set(np.unique(value)).issubset({0, 1, 255}):
        raise ValueError("Expected binary native 304x304 GT_Capillary: " + str(path))
    return np.ascontiguousarray((value > 0).T)  # Native BMP is XY; model projection is YX.


def prepare(parent, labels, cache_override=None, manifest_override=None, data_root_override=None):
    deployed = read(Path(parent) / "deployment.json")
    exp004 = ROOT.parent / "exp004_20261025_a100_prior_direct_residualfm_100e100e"
    def locate(override, recorded, fallback, directory=False):
        candidates = [Path(override)] if override is not None else [Path(recorded), Path(fallback)]
        for candidate in candidates:
            if (candidate.is_dir() if directory else candidate.is_file()):
                return candidate.resolve()
        raise FileNotFoundError("No existing path among: " + ", ".join(map(str, candidates)))
    cache = locate(cache_override, deployed["cache"], exp004 / "cache/train_validation", directory=True)
    manifest = locate(manifest_override, deployed["manifest"], exp004 / "data/manifest.csv")
    data_root = locate(data_root_override, deployed["data_dir"], Path.home() / "flowoct_work/data/OCTA500_v1", directory=True)
    historical = read(ROOT / "provenance/parent_protocol.json")["cache_source"]
    if sha(cache / "cache.json") != historical["cache_metadata_sha256_historical"] or sha(manifest) != historical["manifest_file_sha256_historical"]:
        raise ValueError("Current cache/manifest differ from the preserved exp011 server evidence")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fixed = read(ROOT / "data/membership.json")
    signature = lambda values: sorted((r["case_id"], r["patient_id"], r["split"]) for r in values)
    if signature(rows) != signature(fixed):
        raise ValueError("Fixed patient membership changed")
    by_patient = {}
    recorded_rows = [row.copy() for row in rows]
    for row in rows:
        if row["patient_id"] in by_patient and by_patient[row["patient_id"]] != row["split"]:
            raise ValueError("Patient crosses splits")
        by_patient[row["patient_id"]] = row["split"]
        # Remap stale absolute paths into the explicitly bound current data tree.
        # Only a training sample is opened to verify its original source hash below.
        row["path"] = str(data_root / "z256/3M" / (row["case_id"] + ".npz"))
    if labels is None:
        labels = data_root / "labels_native"
    labels = Path(labels).resolve()
    info = read(cache / "cache.json")
    fingerprint = hashlib.sha256(json.dumps(recorded_rows, sort_keys=True).encode()).hexdigest()
    if fingerprint != info["protocol"]["manifest_sha256"]:
        raise ValueError("Original manifest row fingerprint differs from the anchored cache metadata")
    expected = {row["case_id"]: row["split"] for row in fixed if row["split"] in ("train", "validation")}
    if set(info["cases"]) != set(expected):
        raise ValueError("Cache must contain precisely 140 training and 10 validation cases")
    for key, value in {"fov": "3M", "axis_order": "ZYX", "depth": 256, "dtype": "float32", "intensity_scale": 255.0}.items():
        if info["protocol"][key] != value:
            raise ValueError("Unsupported cache preprocessing: " + key)
    mask_hashes = {}
    for cid, split in expected.items():
        if info["cases"][cid]["split"] != split:
            raise ValueError("Cache split differs: " + cid)
        for key in ("oct", "octa"):
            array = np.load(cache / cid / (key + ".npy"), mmap_mode="r", allow_pickle=False)
            if array.shape != (1, 256, 304, 304) or array.dtype != np.float32:
                raise ValueError("Wrong cache geometry: " + cid)
            del array
        mask_yx(labels, cid)
        mask_hashes[cid] = sha(labels / "GT_Capillary" / (cid + ".bmp"))
    sample = next(row for row in rows if row["split"] == "train")
    if sha(sample["path"]) != info["cases"][sample["case_id"]]["source_npz_sha256"]:
        raise ValueError("Fixed first training source NPZ changed")
    parity = {}
    with np.load(sample["path"], allow_pickle=False) as raw:
        for key in ("oct", "octa"):
            original = np.asarray(raw[key], dtype=np.float32)
            if original.shape != (256, 304, 304):
                raise ValueError("This z256 cache must be compared to its z256 source; native resampling needs a new audit")
            cached = np.load(cache / sample["case_id"] / (key + ".npy"), mmap_mode="r", allow_pickle=False)
            error = float(np.max(np.abs(cached[0] - original / np.float32(255))))
            if error != 0:
                raise ValueError("Cached training sample differs from its recorded source/255")
            parity[key] = error
            del cached
    binding = {"parent": str(Path(parent).resolve()), "cache": str(cache.resolve()), "manifest": str(manifest.resolve()),
               "manifest_sha256": sha(manifest), "cache_metadata_sha256": sha(cache / "cache.json"),
               "labels_root": str(labels), "mask_hashes_train_validation": mask_hashes,
               "mask_source": "public_OCTA500_GT_Capillary_not_verified_author_private_masks", "rows": rows,
               "source_parity_sample": {"case_id": sample["case_id"], "max_abs_error": parity},
               "data_root": str(data_root), "historical_manifest_paths_preserved_in_source_file": True,
               "recorded_cache_path": deployed["cache"], "recorded_manifest_path": deployed["manifest"],
               "recorded_rows_fingerprint": fingerprint}
    previous = ROOT / "binding.json"
    if previous.exists() and read(previous) != binding:
        raise ValueError("Existing binding differs; preserve this experiment and use a new one")
    write(previous, binding)
    return binding


class Inputs:
    def __init__(self, binding):
        self.binding = binding
        self.rows = binding["rows"]
        if sha(binding["manifest"]) != binding["manifest_sha256"] or sha(Path(binding["cache"]) / "cache.json") != binding["cache_metadata_sha256"]:
            raise ValueError("Bound manifest/cache metadata changed")

    def split(self, name):
        return [row for row in self.rows if row["split"] == name]

    def pair(self, row, allow_test=False):
        if row["split"] == "test":
            if not allow_test:
                raise ValueError("Test arrays are inaccessible before frozen evaluation")
            with np.load(row["path"], allow_pickle=False) as data:
                if "axis_order" in data and str(data["axis_order"].item()) != "ZYX":
                    raise ValueError("Expected verified ZYX source")
                arrays = [np.asarray(data[key], dtype=np.float32) for key in ("oct", "octa")]
            if any(not np.isfinite(a).all() or a.min() < 0 or a.max() > 255 for a in arrays):
                raise ValueError("Invalid source intensity")
            if arrays[0].shape != arrays[1].shape or arrays[0].shape[1:] != (304, 304):
                raise ValueError("Unaligned source dimensions")
            if arrays[0].shape[0] != 256:
                arrays = [F.interpolate(torch.from_numpy(a)[None, None], size=(256, 304, 304), mode="trilinear", align_corners=False)[0, 0].numpy() for a in arrays]
            return [np.ascontiguousarray(a[None] / np.float32(255)) for a in arrays]
        result = [np.load(Path(self.binding["cache"]) / row["case_id"] / (key + ".npy"), mmap_mode="r", allow_pickle=False) for key in ("oct", "octa")]
        if any(a.shape != (1, 256, 304, 304) or a.dtype != np.float32 for a in result):
            raise ValueError("Cache shape/dtype changed")
        return result

    def mask(self, row, allow_test=False):
        if row["split"] == "test" and not allow_test:
            raise ValueError("Test labels are inaccessible before frozen evaluation")
        cid = row["case_id"]
        path = Path(self.binding["labels_root"]) / "GT_Capillary" / (cid + ".bmp")
        if row["split"] != "test" and sha(path) != self.binding["mask_hashes_train_validation"][cid]:
            raise ValueError("Bound mask changed: " + cid)
        return mask_yx(self.binding["labels_root"], cid)

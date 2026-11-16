"""Fixed patient membership, canonical ZYX pairs, and reproducible crop schedules."""
from collections import Counter, OrderedDict
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from runtime import ROOT, read_json, sha256, stable_seed, write_json, guard_path


COUNTS = {"train": 140, "validation": 10, "test": 50}
MEMBER_FIELDS = ("case_id", "patient_id", "split")


def bind(manifest, labels_root, data_root=None):
    manifest = guard_path(manifest)
    labels_root = guard_path(labels_root)
    data_root = Path(data_root).resolve() if data_root else None
    members = read_json(ROOT / "membership.json")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    observed = [{key: row[key].strip() for key in MEMBER_FIELDS} for row in rows]
    by_case = {row["case_id"]: row for row in observed}
    if (len(rows) != len(members) or by_case != {row["case_id"]: row for row in members}
            or Counter(row["split"] for row in observed) != COUNTS):
        raise ValueError("Manifest must match the fixed exp018 140/10/50 patient membership.")
    source_rows = {row["case_id"].strip(): row for row in rows}
    result = []
    paths = set()
    leakage_groups = {}
    for member in members:
        case_id, split = member["case_id"], member["split"]
        row = source_rows[case_id]
        if data_root:
            image = data_root / (case_id + ".npz")
        else:
            image = manifest.parent / row["path"]
        image = guard_path(image)
        if image.name != case_id + ".npz" or image in paths:
            raise ValueError(f"Manifest image filename or uniqueness does not match {case_id}.")
        paths.add(image)
        group = row.get("leakage_group", "").strip()
        if group and leakage_groups.setdefault(group, split) != split:
            raise ValueError("A leakage group crosses fixed splits.")
        label = guard_path(labels_root / "GT_Layers" / (case_id + ".mat"))
        if split != "test" and not image.is_file():
            raise FileNotFoundError(image)
        if split != "test" and not label.is_file():
            raise FileNotFoundError(label)
        result.append(dict(member, image=str(image), layers=str(label)))
    binding = dict(manifest=str(manifest), manifest_sha256=sha256(manifest),
                   membership_sha256=sha256(ROOT / "membership.json"), rows=result)
    target = ROOT / "runs/deployment.json"
    if target.exists() and read_json(target) != binding and (ROOT / "checkpoints/latest.pt").exists():
        raise ValueError("The data binding of a started experiment is immutable.")
    if not target.exists() or read_json(target) != binding:
        write_json(target, binding)
    return binding


class Inputs:
    def __init__(self, binding, config, allowed_splits, root=ROOT):
        if (sha256(guard_path(binding["manifest"])) != binding["manifest_sha256"]
                or sha256(root / "membership.json") != binding["membership_sha256"]):
            raise ValueError("The fixed split manifest changed after binding.")
        self.rows = [row for row in binding["rows"] if row["split"] in allowed_splits]
        self.config = config
        self.allowed_splits = set(allowed_splits)
        self.identity_path = root / "verification/input_identities.json"
        self.identities = read_json(self.identity_path) if self.identity_path.exists() else {}
        self.checked = set()
        self.cache = OrderedDict()

    def split(self, phase):
        if phase not in self.allowed_splits:
            raise ValueError(f"This operation cannot access {phase}.")
        return [row for row in self.rows if row["split"] == phase]

    def identify(self, row, kind):
        if row["split"] not in self.allowed_splits:
            raise ValueError(f"This operation cannot access {row['split']}.")
        key = f"{row['case_id']}/{kind}"
        if key not in self.checked:
            path = guard_path(row[kind])
            receipt = dict(split=row["split"], bytes=path.stat().st_size, sha256=sha256(path))
            previous = self.identities.get(key)
            if previous is not None and previous != receipt:
                raise ValueError(f"Previously used input changed: {key}.")
            if previous is None:
                self.identities[key] = receipt
                write_json(self.identity_path, self.identities)
            self.checked.add(key)
        return self.identities[key]

    def volume(self, row, key):
        self.identify(row, "image")
        return read_volume(row["image"], self.config, key)

    def layers(self, row):
        self.identify(row, "layers")
        layers = np.asarray(loadmat(row["layers"])["Layer"], dtype=np.float32)
        expected = (6, *self.config["full_shape"][1:])
        if (layers.shape != expected or not np.isfinite(layers).all()
                or layers.min() < 1 or layers.max() > self.config["native_layer_depth"]
                or (np.diff(layers, axis=0) < 0).any()):
            raise ValueError(f"{row['case_id']}: invalid native-640 layer geometry.")
        return layers



    def pair(self, row):
        # A single archive open reads both OCT and OCTA; bounded four-patient CPU LRU.
        key = row['case_id']
        if key not in self.cache:
            self.identify(row, 'image')
            with np.load(guard_path(row['image']), allow_pickle=False) as archive:
                if 'axis_order' in archive:
                    axis = archive['axis_order'].item()
                    if isinstance(axis, bytes): axis = axis.decode('ascii')
                    if axis != self.config['axis_order']: raise ValueError('Expected ZYX.')
                arrays = []
                for name in ('oct', 'octa'):
                    array = np.array(archive[name], dtype=np.float32, order='C', copy=True)
                    if (list(array.shape) != self.config['full_shape'] or not np.isfinite(array).all()
                            or array.min() < 0 or array.max() > self.config['intensity_scale']):
                        raise ValueError(f'Invalid paired volume: {row["case_id"]}/{name}')
                    array /= np.float32(self.config['intensity_scale'])
                    tensor = torch.from_numpy(array)[None, None]
                    arrays.append(tensor.pin_memory() if torch.cuda.is_available() else tensor)
            arrays.append(torch.from_numpy(self.layers(row))[None])
            self.cache[key] = arrays
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def read_volume(path, config, key="oct"):
    shape = tuple(config["full_shape"])
    with np.load(guard_path(path), allow_pickle=False) as data:
        if "axis_order" in data:
            axis = data["axis_order"].item()
            if isinstance(axis, bytes):
                axis = axis.decode("ascii")
            if axis != config["axis_order"]:
                raise ValueError(f"{path}: expected ZYX axes.")
        array = data[key]
        if array.shape != shape or array.dtype.kind not in "uif":
            raise ValueError(f"{path}/{key}: expected canonical shape {shape}.")
        values = np.array(array, dtype=np.float32, order="C", copy=True)
    scale = config["intensity_scale"]
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > scale:
        raise ValueError(f"{path}/{key}: expected stored values in [0,{scale}].")
    values /= np.float32(scale)
    return torch.from_numpy(values)[None, None]


def schedule(config, epoch, rows):
    if len(rows) != 140 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Training schedule requires the fixed 140 training patients.')
    rng = np.random.default_rng(stable_seed(config['seed'], 'patients', epoch))
    times_rng = np.random.default_rng(stable_seed(config['seed'], 'times', epoch))
    indices = np.concatenate([rng.permutation(140) for _ in range(3)])[:400]
    # Permutation length 140 is even, so pair boundaries cannot cross permutations.
    assert all(indices[i] != indices[i + 1] for i in range(0, 400, 2))
    zero_batches = set(times_rng.permutation(200)[:100].tolist())
    for i in range(200):
        times = times_rng.random(2).astype(np.float32)
        times = np.clip(times, np.nextafter(np.float32(0), np.float32(1)),
                        np.nextafter(np.float32(1), np.float32(0)))
        if i in zero_batches:
            times[int(times_rng.integers(2))] = 0.
        yield [rows[int(k)] for k in indices[2*i:2*i+2]], times.tolist()


def patient_seed(config, row):
    return stable_seed(config["seed"], "inference_noise", row["split"], row["case_id"])


def patient_noise(config, row, shape):
    seed = patient_seed(config, row)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator), seed

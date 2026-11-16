"""Cache/path, log and checkpoint integration around the released author training code."""
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import Inputs
from experiment import ROOT, read, write, sha, checkpoint, rng_state, restore_rng, append_event
from data.base_dataset3d import get_params


def cache_crop(array, params):
    """Author axis order, crop and min-max on an already resized, read-only cache."""
    d0, d1, z0, z1, x0, x1 = params["crop_pos"]
    if z0 or z1:
        raise ValueError("The released 3M transform does not crop anatomical depth")
    view = array[0, :, d0:304-d1, x0:304-x1].transpose(1, 0, 2)
    value = (view - view.min()) / (view.max() - view.min())
    return torch.from_numpy(np.ascontiguousarray((2 * value - 1)[None]))


class CachedAuthorDataset(Dataset):
    def __init__(self, opt, phase):
        if phase not in ("train", "val"):
            raise ValueError("Author training datasets never expose test patients")
        self.opt = opt
        self.inputs = Inputs(read(ROOT / "binding.json"))
        self.rows = self.inputs.split("validation" if phase == "val" else "train")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        a, b = self.inputs.pair(row)
        params = get_params(self.opt, (1, 304, 256, 304))
        base = Path(self.inputs.binding["cache"]) / row["case_id"]
        return {"A": cache_crop(a, params), "B": cache_crop(b, params),
                "A_paths": str(base / "oct.npy"), "B_paths": str(base / "octa.npy")}


def create_dataset(opt, phase):
    # Same DataLoader arguments as source/data/__init__.py; batch1 means len is patients.
    dataset = CachedAuthorDataset(opt, phase)
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                      num_workers=int(opt.num_threads))


MODEL_NAMES = ("netG", "netD", "netD_proj", "netG_t", "net")


def model_states(model, names=MODEL_NAMES):
    return {name: {key: value.detach().cpu() for key, value in getattr(model, name).state_dict().items()}
            for name in names}


class AuthorSession:
    def __init__(self):
        self.start_epoch = 1
        self.total_steps = 0
        self.best_mae = 100000000000000
        self.history = []
        self.best = None
        self.epoch = 0

    def attach(self, model, opt):
        from run import protocol_id
        self.protocol = protocol_id()
        self.opt = opt
        self.author_dir = Path(model.save_dir).resolve()
        resume = ROOT / "checkpoints/resume.pt"
        if resume.exists():
            state = torch.load(resume, map_location="cpu", weights_only=False)
            if state["protocol_id"] != self.protocol:
                raise ValueError("Resume source/data configuration changed")
            for name, weights in state["models"].items():
                getattr(model, name).load_state_dict(weights)
            for optimizer, weights in zip(model.optimizers, state["optimizers"]):
                optimizer.load_state_dict(weights)
            for scheduler, weights in zip(model.schedulers, state["schedulers"]):
                scheduler.load_state_dict(weights)
            self.start_epoch = state["epoch"] + 1
            self.total_steps, self.best_mae = state["total_steps"], state["best_mae"]
            self.history = state["history"]
            # A fresh save directory preserves numbered checkpoints from replayed epochs.
            self.author_dir = self.author_dir / f"resume_after_{state['epoch']:03d}_{time.time_ns()}"
            self.author_dir.mkdir(parents=True)
            model.save_dir = str(self.author_dir)
            for name, weights in state["best_payload"]["models"].items():
                checkpoint(self.author_dir / f"best_net_{self.labels[name]}.pth", weights)
            path = self.author_dir / "best_net_G.pth"
            self.best = {**state["best"], "path":path.relative_to(ROOT).as_posix(), "sha256":sha(path)}
            write(ROOT / "checkpoints/best.json", self.best)
            restore_rng(state["rng"])
            del state
        elif (ROOT / "checkpoints/best.json").exists():
            raise RuntimeError("Best exists but no complete five-epoch resume exists; preserve this interrupted run and start in a new folder")
        self.original_save = model.save
        model.save = lambda label: self.save(model, label)
        write(ROOT / "runs/runtime.json", {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "float_tensor_dtype": str(next(model.netG.parameters()).dtype),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "amp": False, "gradient_scaling": False, "precision_policy": "author_and_framework_defaults"})

    def begin_epoch(self, epoch):
        self.epoch, self.started = epoch, time.perf_counter()
        write(ROOT / "runs/status.json", {"state": "training", "epoch": epoch, "completed_epoch": epoch - 1,
                                           "epochs": 200, "best": self.best, "protocol_id": self.protocol})

    def training_finished(self):
        self.train_seconds = time.perf_counter() - self.started

    def begin_validation(self):
        self.validation_started = time.perf_counter()

    def observe_validation(self, epoch, mae):
        self.mae = float(mae)
        self.validation_finished = time.perf_counter()
        if not np.isfinite(mae):
            raise FloatingPointError("Author validation MAE is non-finite")

    labels = {"netG":"G", "netD":"D", "netD_proj":"D_proj"}

    def save(self, model, label):
        self.original_save(label)  # Original filenames, numbered snapshots, ordering and save implementation.
        if label == "best":
            path = self.author_dir / "best_net_G.pth"
            self.best = {"path":path.relative_to(ROOT).as_posix(), "epoch":self.epoch,
                         "MAE_author_minus1_plus1":self.mae, "sha256":sha(path),
                         "protocol_id": self.protocol}
            write(ROOT / "checkpoints/best.json", self.best)

    def epoch_complete(self, epoch, model, best_mae, total_steps):
        self.best_mae, self.total_steps = float(best_mae), int(total_steps)
        entry = {"epoch": epoch, "total_steps": total_steps, "MAE_author_minus1_plus1": self.mae,
                 "best": self.best, "training_seconds": self.train_seconds,
                 "validation_seconds": self.validation_finished - self.validation_started,
                 "learning_rate_for_next_epoch": model.optimizers[0].param_groups[0]["lr"]}
        self.history.append(entry)
        if epoch == 1 or epoch % self.opt.save_epoch_freq == 0:
            checkpoint(ROOT / "checkpoints/resume.pt", {
                "epoch": epoch, "models": model_states(model),
                "optimizers": [item.state_dict() for item in model.optimizers],
                "schedulers": [item.state_dict() for item in model.schedulers],
                "rng": rng_state(), "best_mae": self.best_mae, "best": self.best,
                "best_payload": {"models":{name:torch.load(self.author_dir / f"best_net_{label}.pth", map_location="cpu", weights_only=True)
                                              for name, label in self.labels.items()}},
                "total_steps": total_steps, "history": self.history, "protocol_id": self.protocol})
        entry["epoch_seconds_including_saves"] = time.perf_counter() - self.started
        entry["saving_and_other_seconds"] = entry["epoch_seconds_including_saves"] - entry["training_seconds"] - entry["validation_seconds"]
        write(ROOT / "runs/history.json", self.history)
        status = {"state": "complete" if epoch == 200 else "training", "epoch": epoch,
                  "completed_epoch": epoch, "epochs": 200, "best": self.best, "protocol_id": self.protocol}
        write(ROOT / "runs/status.json", status)
        append_event({"event": "epoch_complete", **entry})
        if epoch == 200:
            write(ROOT / "runs/training_complete.json", status)
        print(f"[timing] epoch {epoch}/200 train={self.train_seconds:.1f}s validation={entry['validation_seconds']:.1f}s total={entry['epoch_seconds_including_saves']:.1f}s", flush=True)


session = AuthorSession()

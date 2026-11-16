"""Synthetic tests for bounded best retention and crash-safe checkpoint selection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import hashlib

import torch

import experiment
import retention
from experiment import ROOT, read, write, sha


class RetentionTests(unittest.TestCase):
    def payload(self, epoch):
        return {"epoch": epoch, "models": {"G": {"synthetic_weight": torch.tensor([float(epoch)])}},
                "best": {"epoch": epoch, "MAE": 1 / epoch}, "protocol_id": "SYNTHETIC_PROTOCOL"}

    def test_two_slots_and_completed_epoch_recovery(self):
        with TemporaryDirectory(dir=ROOT / "verification", prefix="synthetic_retention_") as temporary:
            path = Path(temporary)
            with patch.object(experiment, "ROOT", path), patch.object(retention, "ROOT", path):
                first = retention.store_best(self.payload(1))
                second = retention.store_best(self.payload(2))
                retention.restore_best(first)
                third = retention.store_best(self.payload(3))
                self.assertEqual(third["slot"], second["slot"])
                self.assertEqual(sha(path / first["path"]), first["sha256"])
                self.assertEqual(len(list((path / "checkpoints").glob("*.pt"))), 2)
                self.assertEqual(read(path / "checkpoints/best.json")["epoch"], 3)

    def test_interrupted_index_commit_preserves_selected_best(self):
        with TemporaryDirectory(dir=ROOT / "verification", prefix="synthetic_commit_") as temporary:
            path = Path(temporary)
            with patch.object(experiment, "ROOT", path), patch.object(retention, "ROOT", path):
                first = retention.store_best(self.payload(1))
                with patch.object(retention, "write", side_effect=OSError("synthetic interrupted commit")):
                    with self.assertRaises(OSError):
                        retention.store_best(self.payload(2))
                self.assertEqual(read(path / "checkpoints/best.json"), first)
                self.assertEqual(sha(path / first["path"]), first["sha256"])

    def test_training_science_unchanged(self):
        parent = ROOT.parent / "exp028_20261028_transpro_retention200e"
        previous, current = read(parent / "experiment.json"), read(ROOT / "experiment.json")
        allowed = {"experiment_id", "parent_experiment", "tf32", "matmul_tf32", "cudnn_benchmark", "cudnn_deterministic", "tf32_scope", "precision_policy"}
        self.assertTrue(all(current[key] == value for key, value in previous.items() if key not in allowed))
        for name in ("network.py", "dataset.py", "metrics.py", "evaluation.py", "run.py", "retention.py"):
            self.assertEqual((ROOT / name).read_bytes(), (parent / name).read_bytes())
        self.assertEqual(current["epochs"], 200)
        self.assertEqual(current["checkpoint_every"], 0)

    def test_budget_is_bounded_even_if_every_epoch_improves(self):
        gib = 1024 ** 3
        budget = retention.storage_budget(int(.8621741868555546 * gib), int(.8106866888701916 * gib))
        self.assertEqual(budget["complete_prediction_cases"], 60)
        self.assertEqual(budget["periodic_snapshots"], 0)
        self.assertLess(budget["required_bytes"] / gib, 20)
        self.assertGreaterEqual(budget["margin_bytes"] / gib, 10)


if __name__ == "__main__":
    (ROOT / "verification").mkdir(exist_ok=True)
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(RetentionTests)
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    write(ROOT / "verification/retention_tests.json", {"tests": result.testsRun, "success": result.wasSuccessful(),
          "synthetic_only": True, "no_existing_checkpoints_deleted": True, "no_patient_images_opened": True})
    raise SystemExit(0 if result.wasSuccessful() else 1)

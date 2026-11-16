"""CPU-only checks for scoped backend changes and honest speedup summaries."""
import ast
import copy
import unittest

import torch

from experiment import ROOT, read, write, precision
from benchmark_speed import PROFILES, apply_profile, summarize


class BackendComparisonTests(unittest.TestCase):
    def test_only_precision_and_runtime_helpers_changed(self):
        parent = ROOT.parent / "exp028_20261028_transpro_retention200e"
        def functions(path):
            return {node.name:ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef)}
        before, after = functions(parent / "experiment.py"), functions(ROOT / "experiment.py")
        self.assertEqual(before.keys(), after.keys())
        self.assertEqual({name for name in before if before[name] != after[name]}, {"precision", "runtime"})

    def test_tf32_scope_and_profile_settings(self):
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        try:
            precision(read(ROOT / "experiment.json"))
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.is_autocast_enabled())
            self.assertEqual(torch.get_default_dtype(), torch.float32)
            for name, profile in PROFILES.items():
                apply_profile(name)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertEqual(torch.backends.cudnn.allow_tf32, profile["cudnn_tf32"])
                self.assertEqual(torch.backends.cudnn.benchmark, profile["benchmark"])
                self.assertEqual(torch.backends.cudnn.deterministic, profile["deterministic"])
            invalid = copy.deepcopy(read(ROOT / "experiment.json")); invalid["amp"] = True
            with self.assertRaises(ValueError):
                precision(invalid)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous

    def test_no_speedup_claim_without_successful_baseline(self):
        combined = {"profile":"combined","status":"passed","mean_seconds":2.,"median_seconds":2.}
        self.assertIsNone(summarize([combined])[0]["speedup_vs_baseline"])
        baseline = {"profile":"baseline","status":"failed"}
        self.assertIsNone(summarize([baseline,combined])[1]["speedup_vs_baseline"])
        baseline.update(status="passed",mean_seconds=8.,median_seconds=8.)
        self.assertEqual(summarize([baseline,combined])[1]["speedup_vs_baseline"], 4.)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendComparisonTests))
    write(ROOT / "verification/local_backend_tests.json", {"tests":result.testsRun,"success":result.wasSuccessful(),
          "CPU_only":True,"GPU_benchmark_executed":False,"patient_arrays_opened":False})
    raise SystemExit(0 if result.wasSuccessful() else 1)

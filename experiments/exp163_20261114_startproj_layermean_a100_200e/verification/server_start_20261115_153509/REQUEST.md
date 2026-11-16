Scope: start exp163 only, without exp162 or formal test.
Authorization: user's 2026-11-15 instruction to start exp163.
Allocation: existing slot1, physical GPU0, NVIDIA A100 80GB PCIe; no request/release.
Scientific source/configuration/200 epochs/200 updates/precision/selection remain unchanged.
Use delivered per-experiment prepare.py, checks.py, run.py, and suite.start([common.ROOT], 'lyh_tmux'); do not call the paired suite CLI.
Reuse existing lyh_flowoct and lyh_tmux without installation or environment changes.
Runtime differences from requirements.txt: scipy 1.15.3 instead of 1.14.1; Pillow 12.3.0 instead of 11.0.0; matplotlib 3.10.1 instead of 3.9.4.
Torch 2.5.1+cu121, numpy 1.26.4 and scikit-image 0.25.2 match. Local retained GPU acceptance already used scipy 1.15.3 and Pillow 12.3.0; its matplotlib was 3.10.3.
Thus computation-library versions match retained local acceptance; plotting version differs. Repeat the delivered mathematical/resume and real-data full-grid acceptance in the actual runtime. Record measured results; do not change losses, metrics, tensors, patient splits, or frozen requirement file.
Only 140 training and 10 validation arrays may be used during preparation; hardware check uses a training case. Delivery byte-integrity verification does not evaluate test cases.
Temporary fixtures stay under exp163/runtime/tmp and are removed by existing TemporaryDirectory contexts. Preserve original local synthetic receipt here before the server check replaces its mutable pointer.
Host /home/lvyuehang and container /workspace were confirmed by matching AGENTS.md SHA256 and lvyuehang identity.

Verification supplement: original synthetic assertion at checks.py:117 failed for different float32 sum grouping (5.960464477539063e-08, one ULP). The diagnostic proves both sums within the float32 forward-error bound and bitwise-identical parameter gradients. Keep delivered checks.py unchanged; an isolated AST harness replaces this single runtime assertion with those two numerical conditions. All other assertions remain unchanged. Preserve the original failed log and diagnostic. Scientific source, configuration, training precision and update arithmetic are unchanged.

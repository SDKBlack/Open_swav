"""Smoke test for Stage2 clustering feature space.

This script is intentionally tiny and dependency-light.
It validates that `compute_stage2_up` runs end-to-end and returns expected keys.

Run:
    python scripts/stage2_space_smoketest.py
"""

import os
import sys

import numpy as np
import torch

# Allow running from repo root without installing as a package.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SWAV_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SWAV_ROOT not in sys.path:
    sys.path.insert(0, SWAV_ROOT)

from src.eval_openset import compute_stage2_up


def main():
    np.random.seed(0)

    N = 200
    D = 128
    num_known = 18

    # Fake features and labels
    X = np.random.randn(N, D).astype("float32")
    Y = np.random.randint(0, num_known + 6, size=(N,)).astype("int64")

    # Pretend 60% predicted unknown
    label_hat = np.random.choice([-1, 0], size=(N,), p=[0.6, 0.4]).astype("int64")

    theta = torch.rand(num_known)

    res = compute_stage2_up(X, torch.from_numpy(Y), label_hat, theta, num_known)
    assert isinstance(res, dict)

    # Baseline keys
    for k in ["up_db", "up_sil", "up_kmeans"]:
        assert k in res, f"missing key: {k}"

    # Optional keys (should be present when sklearn mixture/agg is available)
    # We don't hard-depend on them here; just ensure no crash.
    print("OK. keys=", sorted(res.keys()))
    print("values=", res)


if __name__ == "__main__":
    main()

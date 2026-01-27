"""Stage2 purification + spherical KMeans smoke test.

Runs compute_stage2_up on synthetic data where predicted-unknown contains a mix
of true unknowns and ambiguous near-known samples.

This is NOT a benchmark; it just ensures the new code paths execute and return
expected keys.
"""

import os
import sys
import numpy as np
import torch

# Make "src" importable when running from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval_openset import compute_stage2_up  # noqa: E402


def main():
    rng = np.random.RandomState(0)

    num_known = 5
    num_unknown_classes = 3
    d = 128

    n_known = 200
    n_unknown = 120

    # Known clusters
    known_means = rng.randn(num_known, d) * 3.0
    X_known = np.vstack([
        known_means[c] + rng.randn(n_known // num_known, d) * 0.8
        for c in range(num_known)
    ])
    y_known = np.hstack([
        np.full(n_known // num_known, c, dtype=np.int64)
        for c in range(num_known)
    ])

    # True unknown clusters
    unk_means = rng.randn(num_unknown_classes, d) * 3.0 + 10.0
    X_unk = np.vstack([
        unk_means[c] + rng.randn(n_unknown // num_unknown_classes, d) * 0.8
        for c in range(num_unknown_classes)
    ])
    y_unk = np.hstack([
        np.full(n_unknown // num_unknown_classes, num_known + c, dtype=np.int64)
        for c in range(num_unknown_classes)
    ])

    # Combine
    X = np.vstack([X_known, X_unk]).astype(np.float32)
    y = np.hstack([y_known, y_unk]).astype(np.int64)

    # Predicted labels: mark all unknown as unknown (-1)
    # and also contaminate with some ambiguous known samples near the boundary.
    label_hat = y.copy().astype(np.int64)
    label_hat[label_hat >= num_known] = -1

    # contaminate: flip 10 known points to unknown
    contam_idx = rng.choice(np.where(y < num_known)[0], size=10, replace=False)
    label_hat[contam_idx] = -1

    # theta doesn't matter much for this smoke test; just provide shape.
    theta = torch.ones(num_known) * 1.0

    res = compute_stage2_up(torch.from_numpy(X), torch.from_numpy(y), label_hat, theta, num_known)

    # Basic contract checks
    for k in ["up_db", "up_sil", "up_kmeans"]:
        assert k in res, f"missing key {k}"

    # New key should exist (may be None depending on sklearn behavior)
    assert "up_kmeans_spherical" in res

    print("OK", {k: res[k] for k in sorted(res.keys())})


if __name__ == "__main__":
    main()

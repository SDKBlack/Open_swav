#!/usr/bin/env python3
"""Benchmark model params + inference speed.

Run from repo root:

  python swav/tools/benchmark_models.py --models s3r,wtnet --device cuda \
    --batch-sizes 1,8 --runs 50 --warmup 10 --fp16

This tool benchmarks *forward only* (no dataloading/preprocessing).

Metrics:
- total params / trainable params / approx param size (MB, fp32)
- latency (mean, p50, p90, p99) in ms
- throughput (images/s)
- peak CUDA memory during forward (MB) if CUDA is used

Supported models:
- s3r: S3R/train.py::NET (forward(x, y, z))
- wtnet: swav/src/wtnet.py::WTNet (forward(x))

Examples
--------
1) Quick compare (CUDA if available):
    python swav/tools/benchmark_models.py --models s3r,wtnet --batch-sizes 1,8

2) CPU benchmark (slower but portable):
    python swav/tools/benchmark_models.py --models s3r,wtnet --device cpu --batch-sizes 1

3) FP16 inference (CUDA only):
    python swav/tools/benchmark_models.py --models s3r,wtnet --device cuda --fp16 --batch-sizes 1,8

Troubleshooting
---------------
- Please run from the repo root so imports work.
- If you see ImportError for S3R modules, make sure the directory structure is intact.
- For stable numbers on GPU, we call cuda.synchronize() and do warmup iterations.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class BenchResult:
    mean_s: float
    p50_s: float
    p90_s: float
    p99_s: float
    std_s: float
    throughput: float
    peak_mem_mb: Optional[float]


def _percentile(xs: np.ndarray, p: float) -> float:
    if xs.size == 0:
        return float('nan')
    return float(np.percentile(xs, p))


def sizeof_params(model: torch.nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4.0 / (1024.0**2)
    return int(total), int(trainable), float(size_mb)


def benchmark_forward(
    model: torch.nn.Module,
    inputs,
    *,
    runs: int,
    warmup: int,
    use_cuda: bool,
    fp16: bool,
) -> BenchResult:
    model.eval()

    def _call():
        if isinstance(inputs, (tuple, list)):
            return model(*inputs)
        return model(inputs)

    times = np.zeros((runs,), dtype=np.float64)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # warmup
        for _ in range(warmup):
            if fp16 and use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _call()
            else:
                _ = _call()
            if use_cuda:
                torch.cuda.synchronize()

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        for i in range(runs):
            t0 = time.perf_counter()
            if fp16 and use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _call()
            else:
                _ = _call()
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times[i] = t1 - t0

    mean_s = float(times.mean())
    std_s = float(times.std())
    p50_s = _percentile(times, 50)
    p90_s = _percentile(times, 90)
    p99_s = _percentile(times, 99)

    if isinstance(inputs, (tuple, list)):
        batch = int(inputs[0].shape[0])
    else:
        batch = int(inputs.shape[0])
    throughput = float(batch / mean_s) if mean_s > 0 else 0.0

    peak_mem_mb = None
    if use_cuda:
        peak_mem_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

    return BenchResult(
        mean_s=mean_s,
        p50_s=p50_s,
        p90_s=p90_s,
        p99_s=p99_s,
        std_s=std_s,
        throughput=throughput,
        peak_mem_mb=peak_mem_mb,
    )


def _parse_list_int(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_models(s: str) -> List[str]:
    return [m.strip().lower() for m in s.split(',') if m.strip()]


def build_s3r_net(device: torch.device, *, t: int, w: int, semantic_dim: int, num_known: int):
    repo_root = os.path.abspath('.')
    s3r_dir = os.path.join(repo_root, 'S3R')
    if s3r_dir not in sys.path:
        sys.path.insert(0, s3r_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from train import NET as S3R_NET  # type: ignore

    return S3R_NET(
        in_channels=1,
        input_size=[t, w],
        semantic_dim=semantic_dim,
        num_class=num_known,
        device=device,
    ).to(device)


def build_wtnet(device: torch.device, *, t: int, w: int, semantic_dim: int):
    from swav.src.wtnet import WTNet

    return WTNet(
        in_channels=1,
        input_size=[t, w],
        semantic_dim=semantic_dim,
        num_classes=0,
        use_aux_heads=False,
    ).to(device)


def make_inputs(model_name: str, device: torch.device, *, b: int, t: int, w: int, fp16: bool):
    dtype = torch.float16 if fp16 and device.type == 'cuda' else torch.float32

    if model_name == 's3r':
        x = torch.randn(b, 1, t, w, device=device, dtype=dtype)
        y = torch.randn(b, t, w, device=device, dtype=dtype)
        z = torch.randn(b, w, t, device=device, dtype=dtype)
        return (x, y, z)

    if model_name == 'wtnet':
        x = torch.randn(b, 1, t, w, device=device, dtype=dtype)
        return (x,)

    raise ValueError(f"Unknown model name for inputs: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='s3r,wtnet', help='Comma-separated: s3r,wtnet')
    parser.add_argument('--batch-sizes', type=str, default='1,8', help='Comma-separated batch sizes')
    parser.add_argument('--t', type=int, default=512, help='Input time/freq dimension T (height)')
    parser.add_argument('--w', type=int, default=512, help='Input width dimension W')
    parser.add_argument('--semantic-dim', type=int, default=128)
    parser.add_argument('--num-known', type=int, default=18, help='S3R known-class count (only affects S3R head)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--fp16', action='store_true', help='Use fp16 autocast (CUDA only)')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = (args.device == 'cuda') and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    models = _parse_models(args.models)
    batch_sizes = _parse_list_int(args.batch_sizes)
    if len(models) == 0:
        raise SystemExit('No models selected')
    if len(batch_sizes) == 0:
        raise SystemExit('No batch sizes selected')

    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")
    print(f"Models: {models}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Input: T={args.t}, W={args.w}, semantic_dim={args.semantic_dim}")
    print(f"runs={args.runs}, warmup={args.warmup}, fp16={args.fp16}")

    built: Dict[str, torch.nn.Module] = {}
    for m in models:
        if m == 's3r':
            built[m] = build_s3r_net(device, t=args.t, w=args.w, semantic_dim=args.semantic_dim, num_known=args.num_known)
        elif m == 'wtnet':
            built[m] = build_wtnet(device, t=args.t, w=args.w, semantic_dim=args.semantic_dim)
        else:
            raise SystemExit(f"Unsupported model: {m}")

    print('\n== Params ==')
    for m, model in built.items():
        total, trainable, size_mb = sizeof_params(model)
        print(f"{m:>6s} | total={total:,} | trainable={trainable:,} | approx(fp32)={size_mb:.2f} MB")

    print('\n== Forward benchmark ==')
    for b in batch_sizes:
        print(f"\n-- batch={b} --")
        for m, model in built.items():
            inputs = make_inputs(m, device, b=b, t=args.t, w=args.w, fp16=(args.fp16 and use_cuda))
            res = benchmark_forward(
                model,
                inputs,
                runs=args.runs,
                warmup=args.warmup,
                use_cuda=use_cuda,
                fp16=(args.fp16 and use_cuda),
            )
            print(
                f"{m:>6s} | mean={res.mean_s*1000:.2f} ms "
                f"p50={res.p50_s*1000:.2f} ms p90={res.p90_s*1000:.2f} ms p99={res.p99_s*1000:.2f} ms "
                f"std={res.std_s*1000:.2f} ms | thr={res.throughput:.1f} img/s"
                + (f" | peak_mem={res.peak_mem_mb:.1f} MB" if res.peak_mem_mb is not None else "")
            )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Quick DataLoader benchmark for this project.
Usage:
  python dataloader_bench.py --data_path ... --split_path ... --batch_size 64 --workers 4 --iters 100

The script is defensive: it prints memory using psutil if available, otherwise falls back to /proc/meminfo.
"""
import time
import argparse
import sys
import os

try:
    import psutil
except Exception:
    psutil = None

import torch
from torch.utils.data import DataLoader
from src.s3r_dataset import S3RDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--split_path', required=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nmb_crops', type=int, default=6)
parser.add_argument('--size_crops', type=int, default=224)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('--min_scale', type=float, default=0.8)
parser.add_argument('--max_scale', type=float, default=1.0)
args = parser.parse_args()

print(f"Starting dataloader benchmark: workers={args.workers} batch_size={args.batch_size} iters={args.iters}")

# attempt to limit python threadpool contention (helpful on big machines)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# build dataset
try:
    # S3RDataset expects size_crops and nmb_crops to be sequences of the same length.
    size_crops_list = [args.size_crops] if not hasattr(args.size_crops, '__len__') else args.size_crops
    if isinstance(args.nmb_crops, int):
        nmb_crops_list = [args.nmb_crops]
    else:
        nmb_crops_list = args.nmb_crops

    ds = S3RDataset(args.data_path, args.split_path, size_crops_list, nmb_crops_list, [args.min_scale], [args.max_scale], is_train=True)
except Exception as e:
    print('Failed to construct S3RDataset:', e)
    raise

loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

# quick warmup
it = iter(loader)
warm = min(10, args.iters//5)
try:
    for _ in range(warm):
        next(it)
except StopIteration:
    pass

start = time.time()
count = 0
try:
    for i, batch in enumerate(loader):
        count += 1
        if count >= args.iters:
            break
except Exception as e:
    print('Error during DataLoader iteration:', e)
    raise
end = time.time()

total_samples = args.batch_size * count
elapsed = end - start if end - start > 0 else 1e-6
print(f"workers={args.workers}  samples={total_samples}  time={elapsed:.2f}s  throughput={total_samples/elapsed:.2f} samples/s")

if psutil:
    mem = psutil.virtual_memory()
    print(f"Memory total={mem.total/1e9:.2f}GB  used={mem.used/1e9:.2f}GB  avail={mem.available/1e9:.2f}GB")
else:
    # fallback
    try:
        with open('/proc/meminfo') as f:
            info = f.read()
        # rough parsing
        import re
        m = re.search(r'MemTotal:\s+(\d+) kB', info)
        n = re.search(r'MemAvailable:\s+(\d+) kB', info)
        if m:
            total = int(m.group(1)) / 1024.0 / 1024.0
        else:
            total = 0.0
        if n:
            avail = int(n.group(1)) / 1024.0 / 1024.0
        else:
            avail = 0.0
        print(f"Memory total={total:.2f}GB  available={avail:.2f}GB (from /proc/meminfo)")
    except Exception:
        print('Cannot determine memory usage (no psutil and /proc/meminfo unavailable)')

print('Done')

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
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def try_profile_flops(model: torch.nn.Module, inputs) -> Optional[Tuple[float, float]]:
    """Try to profile FLOPs via external libraries.

    Returns a tuple (macs, params) for one forward pass, or None if profiling isn't available.
    We prefer ptflops for single-tensor models; otherwise fall back to thop (supports multi-input).
    """
    # ptflops: only works for single-tensor inputs
    try:
        from ptflops import get_model_complexity_info  # type: ignore

        if not isinstance(inputs, (tuple, list)):
            input_shape = tuple(int(x) for x in inputs.shape[1:])
            macs, params = get_model_complexity_info(
                model,
                input_shape,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
            return float(macs), float(params)
    except Exception:
        pass

    # thop: supports multi-input
    try:
        from thop import profile  # type: ignore

        model.eval()
        with torch.inference_mode():
            macs, params = profile(model, inputs=inputs, verbose=False)
        return float(macs), float(params)
    except Exception:
        return None


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


class WTNetBackboneOnly(torch.nn.Module):
    """Adapter to benchmark WTNet backbone only.

    This disables projection head + prototypes usage by skipping WTNet.forward_head
    entirely and only calling forward_backbone.

    NOTE: return value is a single tensor (semantic embedding before projection/prototypes).
    """

    def __init__(self, wtnet: torch.nn.Module, *, return_intermediate_features: bool):
        super().__init__()
        self.wtnet = wtnet
        self.return_intermediate_features = return_intermediate_features

    def forward(self, x):
        out, _aux, _inter = self.wtnet.forward_backbone(
            x,
            return_intermediate_features=self.return_intermediate_features,
        )
        return out


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
    parser.add_argument('--wtnet-no-intermediate', action='store_true', help='WTNet inference mode: disable intermediate features (stage-2 clustering features)')
    parser.add_argument('--wtnet-backbone-only', action='store_true', help='WTNet benchmark mode: disable projection head + prototypes by timing backbone only')
    parser.add_argument('--compile', action='store_true', help='Try torch.compile(model) to measure compiled inference speed (PyTorch 2.x)')
    parser.add_argument('--verify', action='store_true', help='Print a small output summary per (model,batch) to verify the benchmark is executing the expected path')
    parser.add_argument('--isolate-per-model', action='store_true', help='Run each model in a separate process to isolate peak CUDA memory stats')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    # If requested, re-launch this script once per model in a fresh process.
    # This isolates CUDA memory peaks (avoids one model inflating the peak of another).
    if args.isolate_per_model:
        models = _parse_models(args.models)
        if len(models) == 0:
            raise SystemExit('No models selected')

        print('== isolate-per-model: launching separate processes ==')
        for m in models:
            cmd = [sys.executable, '-m', 'swav.tools.benchmark_models']
            cmd += ['--models', m]
            cmd += ['--batch-sizes', args.batch_sizes]
            cmd += ['--t', str(args.t), '--w', str(args.w)]
            cmd += ['--semantic-dim', str(args.semantic_dim)]
            cmd += ['--num-known', str(args.num_known)]
            cmd += ['--device', args.device]
            cmd += ['--runs', str(args.runs), '--warmup', str(args.warmup)]
            if args.fp16:
                cmd.append('--fp16')
            if args.wtnet_no_intermediate:
                cmd.append('--wtnet-no-intermediate')
            if args.wtnet_backbone_only:
                cmd.append('--wtnet-backbone-only')
            if args.compile:
                cmd.append('--compile')
            if args.verify:
                cmd.append('--verify')
            # Prevent recursion
            # (do NOT pass --isolate-per-model to child)
            cmd += ['--seed', str(args.seed)]

            print('\n' + '=' * 80)
            print('Running:', ' '.join(cmd))
            res = subprocess.run(cmd, cwd=os.path.abspath('.'), text=True, capture_output=True)
            if res.stdout:
                print(res.stdout.rstrip())
            if res.returncode != 0:
                if res.stderr:
                    print(res.stderr.rstrip())
                raise SystemExit(res.returncode)

        return

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
    if 'wtnet' in models:
        print(f"WTNet return_intermediate_features: {not args.wtnet_no_intermediate}")

    built: Dict[str, torch.nn.Module] = {}
    for m in models:
        if m == 's3r':
            built[m] = build_s3r_net(device, t=args.t, w=args.w, semantic_dim=args.semantic_dim, num_known=args.num_known)
        elif m == 'wtnet':
            built[m] = build_wtnet(device, t=args.t, w=args.w, semantic_dim=args.semantic_dim)
        else:
            raise SystemExit(f"Unsupported model: {m}")

    # Optionally compile models for inference (if requested and supported).
    # We'll attempt two backends: the default (inductor -> triton) and 'aot_eager'.
    # Store compiled variants per-model so we can benchmark both compiled backends.
    compiled_built: Dict[str, Dict[str, Optional[torch.nn.Module]]] = {m: {} for m in built.keys()}
    if args.compile:
        if not hasattr(torch, 'compile'):
            print('torch.compile not available in this PyTorch build; continuing without compile')
        else:
            backend_names = ['default', 'aot_eager']
            for m, model in list(built.items()):
                for backend_name in backend_names:
                    try:
                        kw = {'mode': 'reduce-overhead'}
                        if backend_name != 'default':
                            kw['backend'] = backend_name
                        print(f"Attempting to torch.compile({m}, backend={backend_name}) ...")
                        compiled_model = torch.compile(model, **kw)
                        compiled_built[m][backend_name] = compiled_model
                        print(f"torch.compile succeeded for {m} (backend={backend_name})")
                    except Exception as e:
                        compiled_built[m][backend_name] = None
                        print(f"torch.compile failed for {m} (backend={backend_name}): {e}; will skip this backend")

    print('\n== Params ==')
    for m, model in built.items():
        total, trainable, size_mb = sizeof_params(model)
        print(f"{m:>6s} | total={total:,} | trainable={trainable:,} | approx(fp32)={size_mb:.2f} MB")

    # FLOPs / MACs (shape-dependent). We profile on a representative batch size (the first one).
    print('\n== FLOPs / MACs (per forward, includes batch) ==')
    b_flops = int(batch_sizes[0])
    for m, model in built.items():
        inputs_for_flops = make_inputs(m, device, b=b_flops, t=args.t, w=args.w, fp16=False)

        def _wrap_for_flops(base: torch.nn.Module) -> torch.nn.Module:
            if m != 'wtnet':
                return base
            if args.wtnet_backbone_only:
                return WTNetBackboneOnly(base, return_intermediate_features=(not args.wtnet_no_intermediate))

            class _WTNetAdapterFlops(torch.nn.Module):
                def __init__(self, mdl, return_intermediate: bool):
                    super().__init__()
                    self.mdl = mdl
                    self.return_intermediate = return_intermediate

                def forward(self, *ins, **kw):
                    return self.mdl(*ins, return_intermediate_features=self.return_intermediate, **kw)

            return _WTNetAdapterFlops(base, return_intermediate=(not args.wtnet_no_intermediate))

        # Profile original model only (compile backend doesn't change FLOPs).
        prof_model = _wrap_for_flops(model)
        prof = try_profile_flops(prof_model, inputs_for_flops)
        if prof is None:
            print(f"{m:>6s} | batch={b_flops} | FLOPs=N/A (profiling lib missing or unsupported)")
        else:
            macs, params = prof
            # convention: FLOPs ~= 2*MACs for multiply-add
            gmacs = macs / 1e9
            gflops = (2.0 * macs) / 1e9
            print(f"{m:>6s} | batch={b_flops} | GMACs={gmacs:.3f} | GFLOPs~={gflops:.3f} | params={int(params):,}")

    print('\n== Forward benchmark ==')
    for b in batch_sizes:
        print(f"\n-- batch={b} --")
        for m, model in built.items():
            inputs = make_inputs(m, device, b=b, t=args.t, w=args.w, fp16=(args.fp16 and use_cuda))

            # Prepare candidate variants: compiled backends (if any) followed by original
            variants = []  # list of tuples (variant_tag, module, is_compiled)
            if args.compile:
                backends_map = compiled_built.get(m, {})
                for backend_name, compiled_model in backends_map.items():
                    if compiled_model is not None:
                        variants.append((f"{m}[{backend_name}]", compiled_model, True, backend_name))
            # Always include the original (uncompiled) model as the last resort
            variants.append((f"{m}[orig]", model, False, 'orig'))

            # For each variant (compiled backend or original), benchmark separately
            for var_tag, var_model, is_compiled, backend_name in variants:
                model_base = model
                model_to_use = var_model

                if m == 'wtnet':
                    # Build WTNet adapter wrapping the chosen model variant
                    if args.wtnet_backbone_only:
                        adapter = WTNetBackboneOnly(
                            model_to_use,
                            return_intermediate_features=(not args.wtnet_no_intermediate),
                        )
                    else:
                        class _WTNetAdapter(torch.nn.Module):
                            def __init__(self, mdl, return_intermediate: bool):
                                super().__init__()
                                self.mdl = mdl
                                self.return_intermediate = return_intermediate
                            def forward(self, *ins, **kw):
                                return self.mdl(*ins, return_intermediate_features=self.return_intermediate, **kw)

                        adapter = _WTNetAdapter(model_to_use, return_intermediate=(not args.wtnet_no_intermediate))

                    # Optional verification before timing
                    if args.verify:
                        adapter.eval()
                        with torch.inference_mode():
                            if args.fp16 and use_cuda:
                                with torch.cuda.amp.autocast(dtype=torch.float16):
                                    y = adapter(*inputs)
                            else:
                                y = adapter(*inputs)
                            if isinstance(y, (tuple, list)):
                                y0 = y[0]
                            else:
                                y0 = y
                            y0_fp32 = y0.detach().float()
                            sig = {
                                'shape': tuple(y0.shape),
                                'dtype': str(y0.dtype).replace('torch.', ''),
                                'device': str(y0.device),
                                'mean': float(y0_fp32.mean().cpu()),
                                'std': float(y0_fp32.std(unbiased=False).cpu()),
                                'absmax': float(y0_fp32.abs().max().cpu()),
                            }
                            tagv = var_tag
                            if args.wtnet_backbone_only:
                                tagv = tagv.replace('wtnet', 'wtnet(bb)')
                            print(f"  [verify] {tagv} -> {sig}")

                    try:
                        res = benchmark_forward(
                            adapter,
                            inputs,
                            runs=args.runs,
                            warmup=args.warmup,
                            use_cuda=use_cuda,
                            fp16=(args.fp16 and use_cuda),
                        )
                    except Exception as e:
                        # If compiled variant failed at runtime, and this variant was compiled, retry with original
                        if is_compiled:
                            print(f'Runtime failure with compiled {var_tag}: {e}; retrying with original model')
                            if args.wtnet_backbone_only:
                                adapter = WTNetBackboneOnly(
                                    model_base,
                                    return_intermediate_features=(not args.wtnet_no_intermediate),
                                )
                            else:
                                adapter = _WTNetAdapter(model_base, return_intermediate=(not args.wtnet_no_intermediate))
                            res = benchmark_forward(
                                adapter,
                                inputs,
                                runs=args.runs,
                                warmup=args.warmup,
                                use_cuda=use_cuda,
                                fp16=(args.fp16 and use_cuda),
                            )
                        else:
                            raise
                else:
                    # Non-wtnet models
                    if args.verify:
                        model_to_use.eval()
                        with torch.inference_mode():
                            if args.fp16 and use_cuda:
                                with torch.cuda.amp.autocast(dtype=torch.float16):
                                    y = model_to_use(*inputs)
                            else:
                                y = model_to_use(*inputs)
                            if isinstance(y, (tuple, list)):
                                y0 = y[0]
                            else:
                                y0 = y
                            y0_fp32 = y0.detach().float()
                            sig = {
                                'shape': tuple(y0.shape),
                                'dtype': str(y0.dtype).replace('torch.', ''),
                                'device': str(y0.device),
                                'mean': float(y0_fp32.mean().cpu()),
                                'std': float(y0_fp32.std(unbiased=False).cpu()),
                                'absmax': float(y0_fp32.abs().max().cpu()),
                            }
                            print(f"  [verify] {var_tag} -> {sig}")

                    try:
                        res = benchmark_forward(
                            model_to_use,
                            inputs,
                            runs=args.runs,
                            warmup=args.warmup,
                            use_cuda=use_cuda,
                            fp16=(args.fp16 and use_cuda),
                        )
                    except Exception as e:
                        if is_compiled:
                            print(f'Runtime failure with compiled {var_tag}: {e}; retrying with the original model')
                            model_base.eval()
                            res = benchmark_forward(
                                model_base,
                                inputs,
                                runs=args.runs,
                                warmup=args.warmup,
                                use_cuda=use_cuda,
                                fp16=(args.fp16 and use_cuda),
                            )
                        else:
                            raise

                # print results with the variant tag (clean up tag for backbone shorthand)
                pretty_tag = var_tag
                if m == 'wtnet' and args.wtnet_backbone_only:
                    pretty_tag = pretty_tag.replace('wtnet', 'wtnet(bb)')

                print(
                    f"{pretty_tag:>16s} | mean={res.mean_s*1000:.2f} ms "
                    f"p50={res.p50_s*1000:.2f} ms p90={res.p90_s*1000:.2f} ms p99={res.p99_s*1000:.2f} ms "
                    f"std={res.std_s*1000:.2f} ms | thr={res.throughput:.1f} img/s"
                    + (f" | peak_mem={res.peak_mem_mb:.1f} MB" if res.peak_mem_mb is not None else "")
                )


if __name__ == '__main__':
    main()

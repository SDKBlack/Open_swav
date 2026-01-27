import time
import torch
import sys
import os

repo_root = '/root/autodl-tmp'
swav_root = os.path.join(repo_root, 'data', 'swav-main', 'swav-main')
# Ensure module import paths
sys.path.insert(0, swav_root)
sys.path.insert(0, repo_root)
# Add S3R folder so train.py can import contLoss and other local modules
sys.path.insert(0, os.path.join(repo_root, 'S3R'))

# Imports
try:
    from src.wt_models import wt_net as build_wtnet
except Exception as e:
    print('Failed to import wt_net:', e)
    raise

try:
    from S3R.train import NET as NETClass
except Exception as e:
    print('Failed to import NET:', e)
    raise


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def model_size_mb(m):
    # assume float32
    return count_params(m) * 4 / (1024**2)


def bench_forward(model, forward_fn, inputs_fn, device, n_warmup=5, n_iter=30):
    # warmup
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(n_warmup):
        out = forward_fn(*inputs_fn())
        if device.type == 'cuda':
            torch.cuda.synchronize()
    # timed
    t0 = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(n_iter):
        out = forward_fn(*inputs_fn())
        if device.type == 'cuda':
            torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) / n_iter * 1000.0
    peak_mem = None
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    return avg_ms, peak_mem


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # instantiate NET (from S3R.train)
    net = NETClass(in_channels=1, input_size=[256,256], semantic_dim=128, num_class=18, device=device)
    net.eval()
    net.to(device)

    # instantiate WTNet via builder
    wtnet = build_wtnet(in_channels=1, input_size=[256,256], semantic_dim=128, num_class=18,
                        device=device, wt_levels=1, use_spatial_attn=True,
                        output_dim=128, hidden_mlp=512, nmb_prototypes=54, normalize=True)
    wtnet.eval()
    wtnet.to(device)

    print('\nModel summary (params and size):')
    p_net = count_params(net)
    p_wt = count_params(wtnet)
    print(f'NET params: {p_net:,} ({model_size_mb(net):.2f} MB)')
    print(f'WTNet params: {p_wt:,} ({model_size_mb(wtnet):.2f} MB)')

    batch_size = 2

    # Prepare forward functions and input generators
    def inputs_net():
        x = torch.randn(batch_size, 1, 256, 256, device=device)
        # y: [B, T, W] ; z: [B, W, T]
        y = torch.randn(batch_size, 256, 256, device=device)
        z = torch.randn(batch_size, 256, 256, device=device)
        return (x, y, z)

    def forward_net(x, y, z):
        return net(x, y, z)

    def inputs_wt():
        x = torch.randn(batch_size, 1, 256, 256, device=device)
        # WTNet accepts either tensor or list; we pass a single tensor
        return ( [x], )

    def forward_wt(inputs):
        return wtnet(inputs)

    # Benchmark NET
    print('\nBenchmarking NET forward...')
    avg_net_ms, net_peak = bench_forward(net, forward_net, inputs_net, device)
    print(f'NET avg forward: {avg_net_ms:.2f} ms | peak_mem: {net_peak} MB')

    # Benchmark WTNet
    print('\nBenchmarking WTNet forward...')
    avg_wt_ms, wt_peak = bench_forward(wtnet, forward_wt, inputs_wt, device)
    print(f'WTNet avg forward: {avg_wt_ms:.2f} ms | peak_mem: {wt_peak} MB')

    # Relative
    print('\nRelative (WTNet / NET):')
    print(f'Param ratio (WT/NET): {p_wt / p_net:.3f}')
    print(f'Latency ratio (WT/NET): {avg_wt_ms / avg_net_ms:.3f}')
    if net_peak and wt_peak:
        print(f'Peak mem ratio (WT/NET): {wt_peak / net_peak:.3f}')

if __name__ == "__main__":
    main()

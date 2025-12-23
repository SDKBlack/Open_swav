import time
import torch
import importlib, sys
sys.path.insert(0, '/root/autodl-tmp/swav')
from src import resnet50 as resnet_models
from src.wtnet import WTNet


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure_forward(model, device, bs, iters=100, warmup=20):
    model.eval()
    inp = torch.randn(bs, 3, 224, 224, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.forward_backbone(inp)
    # timed
    times = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model.forward_backbone(inp)
            torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)
    import numpy as np
    return np.mean(times), np.std(times)


def safe_instantiate_resnet(name='resnet18', **kwargs):
    if name not in resnet_models.__dict__:
        raise KeyError(name)
    return resnet_models.__dict__[name](**kwargs)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    # resnet18
    print('\nBuilding resnet18...')
    r = safe_instantiate_resnet('resnet18', normalize=True, hidden_mlp=2048, output_dim=128, nmb_prototypes=0, num_classes=0)
    r = r.to(device)
    print('resnet18 params:', count_params(r))

    # WTNet baseline (no shared stem)
    print('\nBuilding WTNet (no shared stem)...')
    w_no_share = WTNet(normalize=True, hidden_mlp=2048, output_dim=128, nmb_prototypes=0, num_classes=0, use_shared_stem=False)
    w_no_share = w_no_share.to(device)
    print('wtnet (no share) params:', count_params(w_no_share))

    # WTNet with shared stem
    print('\nBuilding WTNet (with shared stem)...')
    w_share = WTNet(normalize=True, hidden_mlp=2048, output_dim=128, nmb_prototypes=0, num_classes=0, use_shared_stem=True, shared_stem_blocks=2)
    w_share = w_share.to(device)
    print('wtnet (shared stem) params:', count_params(w_share))

    models = [('resnet18', r), ('wtnet_no_share', w_no_share), ('wtnet_share', w_share)]
    batch_sizes = [1, 8, 32]

    for name, m in models:
        print('\n=== Model:', name)
        for bs in batch_sizes:
            mean, std = measure_forward(m, device, bs, iters=60, warmup=10)
            print(f'batch={bs} mean={mean:.2f} ms std={std:.2f} ms per-sample={mean/bs:.3f} ms')

if __name__ == '__main__':
    main()

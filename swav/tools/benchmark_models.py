import time
import torch
import torch.nn as nn
from src import resnet50 as resnet_models
from src.wtnet import WTNet


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def bytes_for_dtype(dtype):
    # approximate bytes per element
    if dtype == torch.float32:
        return 4
    if dtype == torch.float16:
        return 2
    if dtype == torch.float64:
        return 8
    return 4


def model_size_mb(model, dtype=torch.float32):
    return count_params(model) * bytes_for_dtype(dtype) / (1024 ** 2)


def measure_forward_time(model, input_shape, device, batch_size=1, iters=50, warmup=10, use_amp=False):
    model.eval()
    bs = batch_size
    x = torch.randn((bs, *input_shape), device=device)
    # one warmup
    with torch.no_grad():
        for _ in range(warmup):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if use_amp and device.type=='cuda':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.time()
            if use_amp and device.type=='cuda':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)  # ms
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # instantiate models
    resnet = resnet_models.resnet18(normalize=True, output_dim=128, hidden_mlp=0, nmb_prototypes=0, num_classes=0)
    wtnet = WTNet(in_channels=3, input_size=[224,224], semantic_dim=128, num_classes=0, output_dim=128, hidden_mlp=0, nmb_prototypes=0)

    resnet = resnet.to(device)
    wtnet = wtnet.to(device)

    print('\nModel parameter counts and sizes:')
    for name, m in [('resnet18', resnet), ('wtnet', wtnet)]:
        params = count_params(m)
        size_mb = model_size_mb(m)
        print(f"{name}: params={params:,}, approx size={size_mb:.2f} MB")

    # benchmark forward time for various batch sizes
    input_shape = (3, 224, 224)
    batch_sizes = [1, 8, 32]
    use_amp = torch.cuda.is_available()

    print('\nForward time benchmarks (avg ms per forward):')
    for name, m in [('resnet18', resnet), ('wtnet', wtnet)]:
        print(f"\n{name}:")
        for bs in batch_sizes:
            avg, std = measure_forward_time(m, input_shape, device, batch_size=bs, iters=30, warmup=10, use_amp=use_amp)
            per_sample = avg / bs
            print(f"  batch={bs}  avg={avg:.2f} ms  std={std:.2f} ms  per-sample={per_sample:.3f} ms")

    print('\nDone')

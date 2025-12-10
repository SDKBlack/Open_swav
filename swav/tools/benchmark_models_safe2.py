import time
import torch
import importlib
import sys
from pathlib import Path

SWAV_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = SWAV_ROOT / 'src'
# ensure parent dir in sys.path so 'src' becomes importable as a package
sys.path.insert(0, str(SWAV_ROOT))

# import via package
resnet50_mod = importlib.import_module('src.resnet50')
wtnet_mod = importlib.import_module('src.wtnet')

resnet18_fn = getattr(resnet50_mod, 'resnet18')
WTNet = getattr(wtnet_mod, 'WTNet')


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def bytes_for_dtype(dtype):
    if dtype == torch.float32:
        return 4
    if dtype == torch.float16:
        return 2
    if dtype == torch.float64:
        return 8
    return 4


def model_size_mb(model, dtype=torch.float32):
    return count_params(model) * bytes_for_dtype(dtype) / (1024 ** 2)


def run_model_get_embedding(model, x, use_amp=False):
    # model.forward may return: embedding OR (embedding, prototypes) OR (embedding, prototypes, logits)
    # Prefer calling backbone + head to avoid model.forward unpacking issues
    if hasattr(model, 'forward_backbone') and hasattr(model, 'forward_head'):
        if use_amp and x.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                backbone = model.forward_backbone(x)
        else:
            backbone = model.forward_backbone(x)
        # forward_head may return embedding or (embedding, prototypes)
        out = model.forward_head(backbone)
    else:
        if use_amp and x.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                out = model(x)
        else:
            out = model(x)
    if isinstance(out, (tuple, list)):
        emb = out[0]
    else:
        emb = out
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    return emb


def measure_forward_time(model, input_shape, device, batch_size=1, iters=30, warmup=10, use_amp=False):
    model.eval()
    bs = batch_size
    x = torch.randn((bs, *input_shape), device=device)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _ = run_model_get_embedding(model, x, use_amp=use_amp)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.time()
            _ = run_model_get_embedding(model, x, use_amp=use_amp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # instantiate models
    resnet = resnet18_fn(normalize=True, output_dim=128, hidden_mlp=0, nmb_prototypes=0, num_classes=0)
    wtnet = WTNet(in_channels=3, input_size=[224,224], semantic_dim=128, num_classes=0, output_dim=128, hidden_mlp=0, nmb_prototypes=0)

    resnet = resnet.to(device)
    wtnet = wtnet.to(device)

    print('\nModel parameter counts and sizes:')
    for name, m in [('resnet18', resnet), ('wtnet', wtnet)]:
        params = count_params(m)
        size_mb = model_size_mb(m)
        print(f"{name}: params={params:,}, approx size={size_mb:.2f} MB")

    input_shape = (3, 224, 224)
    batch_sizes = [1, 8, 32]
    # Disable AMP for reliable dtype behavior in these models
    use_amp = False

    print('\nForward time benchmarks (avg ms per forward):')
    for name, m in [('resnet18', resnet), ('wtnet', wtnet)]:
        print(f"\n{name}:")
        for bs in batch_sizes:
            avg, std = measure_forward_time(m, input_shape, device, batch_size=bs, iters=30, warmup=10, use_amp=use_amp)
            per_sample = avg / bs
            print(f"  batch={bs}  avg={avg:.2f} ms  std={std:.2f} ms  per-sample={per_sample:.3f} ms")

    print('\nDone')

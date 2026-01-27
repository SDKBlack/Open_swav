import os
import sys
import torch
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wtnet import WTNet


def find_latest_dump(dump_dir):
    files = [os.path.join(dump_dir, f) for f in os.listdir(dump_dir) if f.endswith('.pth')]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def tensor_has_nonfinite(x):
    try:
        return not torch.isfinite(x).all()
    except Exception:
        return False


def check_obj_nonfinite(obj):
    # Recursively check tensors inside obj
    if isinstance(obj, torch.Tensor):
        return tensor_has_nonfinite(obj)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if check_obj_nonfinite(v):
                return True
    elif isinstance(obj, dict):
        for v in obj.values():
            if check_obj_nonfinite(v):
                return True
    return False


def main(dump_path=None):
    if dump_path is None:
        dump_path = find_latest_dump('.')
    if dump_path is None or not os.path.isfile(dump_path):
        print(f"No dump file found at: {dump_path}")
        return 2

    print(f"Loading dump: {dump_path}")
    # Some debug dumps contain arbitrary python objects; allow full payload load (weights_only=False)
    dump = torch.load(dump_path, map_location='cpu', weights_only=False)

    model_state = dump.get('model_state') or dump.get('model_state_dict') or dump.get('state_dict') or dump.get('model')
    if model_state is None:
        # Maybe they saved model.module.state_dict()
        # try to detect keys directly in dump
        for k in dump.keys():
            if isinstance(dump[k], dict) and any(isinstance(v, torch.Tensor) for v in dump[k].values()):
                model_state = dump[k]
                break

    if model_state is None:
        print('Could not find model_state in dump. Keys:', list(dump.keys()))
        return 3

    # Infer num_classes and prototypes from state dict
    num_classes = 0
    nmb_prototypes = 0
    use_aux_heads = False
    for k, v in model_state.items():
        if k.endswith('classifier.weight'):
            num_classes = v.shape[0]
        if k.endswith('prototypes.weight'):
            nmb_prototypes = v.shape[0]
        if 'aux_head1' in k or 'aux_head3' in k or 'aux_head5' in k:
            use_aux_heads = True

    print(f"Inferred num_classes={num_classes}, nmb_prototypes={nmb_prototypes}, use_aux_heads={use_aux_heads}")

    # Create model with inferred settings. Many other args use defaults; load_state_dict with strict=False.
    model = WTNet(num_classes=num_classes, nmb_prototypes=nmb_prototypes, use_aux_heads=use_aux_heads)
    model.cpu()

    # Load state dict (allow mismatch to avoid strict failures)
    try:
        model.load_state_dict(model_state, strict=False)
    except Exception as e:
        print('Warning: load_state_dict raised:', e)

    model.eval()

    inputs = dump.get('inputs')
    if inputs is None:
        print('No inputs found in dump. Keys:', list(dump.keys()))
        return 4

    # Ensure inputs is a list of tensors on CPU
    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = [t.cpu() for t in inputs]

    # Prepare crop indices as in WTNet.forward
    sizes = [inp.shape[-1] for inp in inputs]
    # unique consecutive with counts
    import numpy as np
    uniq, counts = np.unique(sizes, return_counts=True)
    idx_crops = np.cumsum(counts)

    first_bad = None
    bad_info = {}

    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            nonlocal first_bad, bad_info
            if first_bad is not None:
                return
            if check_obj_nonfinite(out):
                first_bad = name
                # Collect stats
                def tensor_stats(t):
                    try:
                        import numpy as _np
                        arr = t.detach().cpu().numpy()
                        return {
                            'shape': list(t.shape),
                            'mean': float(_np.nanmean(arr)),
                            'min': float(_np.nanmin(arr)),
                            'max': float(_np.nanmax(arr)),
                            'finite_count': int(_np.isfinite(arr).sum()),
                        }
                    except Exception:
                        # Fallback to torch-based but guarded
                        finite = torch.isfinite(t)
                        finite_count = int(finite.sum().item())
                        vals = t[finite]
                        if vals.numel() == 0:
                            return {'shape': list(t.shape), 'mean': None, 'min': None, 'max': None, 'finite_count': finite_count}
                        return {'shape': list(t.shape), 'mean': float(vals.mean().item()), 'min': float(vals.min().item()), 'max': float(vals.max().item()), 'finite_count': finite_count}

                out_t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor) else None)
                bad_info = {
                    'module': name,
                    'type': type(module).__name__,
                    'output_stats': tensor_stats(out_t) if out_t is not None else None,
                }
                print('Detected non-finite output at module:', name, 'type:', type(module).__name__)
        return hook

    for name, module in model.named_modules():
        # skip top-level module itself
        if name == '':
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    # Run forward by invoking forward_backbone for each crop-group (CPU)
    start = 0
    try:
        for end in idx_crops:
            group = inputs[start:end]
            x = torch.cat(group, dim=0)
            # forward_backbone expects tensor on same device as model
            out, aux = model.forward_backbone(x)
            # quick check
            if check_obj_nonfinite(out) or check_obj_nonfinite(aux):
                if first_bad is None:
                    first_bad = 'after_forward_backbone'
                    bad_info = {'module': 'forward_backbone', 'type': 'forward_backbone', 'output_stats': None}
                    print('Non-finite detected after forward_backbone')
                break
            start = end
    except Exception as e:
        print('Forward raised exception:', e)

    finally:
        for h in handles:
            h.remove()

    report = {
        'dump_path': dump_path,
        'first_bad': first_bad,
        'bad_info': bad_info,
    }
    out_report = os.path.join(os.path.dirname(dump_path), 'replay_nan_report.pth')
    torch.save(report, out_report)
    print('Saved replay report to', out_report)
    if first_bad is not None:
        print('First bad module:', first_bad)
        print('Bad info:', bad_info)
        # Also list parameters of that module that are non-finite (if present)
        mod = dict(model.named_modules()).get(first_bad, None)
        if mod is not None:
            bad_params = []
            for n, p in mod.named_parameters(prefix=first_bad):
                if check_obj_nonfinite(p.data):
                    bad_params.append((n, list(p.data.shape)))
            if bad_params:
                print('Non-finite parameters in module:', bad_params)
    else:
        print('No non-finite outputs detected during replay.')

    return 0


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=str, default=None, help='Path to nan debug dump (.pth)')
    args = ap.parse_args()
    sys.exit(main(args.dump))

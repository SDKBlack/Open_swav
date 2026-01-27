#!/usr/bin/env python3
"""
Replay a saved debug dump (from --debug_nans) and locate the first module whose
forward output becomes non-finite. Saves a small JSON/pt report.

Usage examples:
  python scripts/find_bad_module.py --dump /path/to/debug_nan_forward_epoch49_iter17.pt \
       --ckpt /path/to/checkpoint.pth.tar --device cuda:0 --num_samples 8

This script tries to reconstruct the model using the `params.pkl` saved in the
same dump directory (written by initialize_exp). It imports the project's
`main_swav` and `src.wt_models` to build the same architecture.
"""

import os
import sys
import argparse
import pickle
import json
import traceback
from collections import OrderedDict

import torch
import numpy as np


def add_repo_paths():
    # add likely repo paths so we can import main_swav and src.wt_models
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    probable_paths = [
        os.path.join(repo_root, 'data', 'swav-main', 'swav-main'),
        os.path.join(repo_root, 'swav-main', 'swav-main'),
        repo_root,
        os.getcwd(),
    ]
    for p in probable_paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def safe_tonumpy(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        try:
            return np.array(x)
        except Exception:
            return None


def tensor_stats(t):
    if t is None:
        return None
    if not torch.is_tensor(t):
        return None
    t_cpu = t.detach().cpu()
    finite_mask = torch.isfinite(t_cpu)
    all_finite = bool(finite_mask.all().item())
    any_finite = bool(finite_mask.any().item())
    nan_count = int((~finite_mask).logical_and(torch.isnan(t_cpu)).sum().item()) if t_cpu.numel() > 0 else 0
    inf_pos = int((t_cpu == float('inf')).sum().item()) if t_cpu.numel() > 0 else 0
    inf_neg = int((t_cpu == float('-inf')).sum().item()) if t_cpu.numel() > 0 else 0
    # compute basic stats on finite elements
    finite_vals = t_cpu[finite_mask]
    if finite_vals.numel() > 0:
        mn = float(torch.min(finite_vals).item())
        mx = float(torch.max(finite_vals).item())
        mean = float(torch.mean(finite_vals).item())
    else:
        mn = mx = mean = None
    return {
        'shape': list(t_cpu.shape),
        'dtype': str(t_cpu.dtype),
        'all_finite': all_finite,
        'any_finite': any_finite,
        'nan_count': nan_count,
        'inf_pos': inf_pos,
        'inf_neg': inf_neg,
        'min_finite': mn,
        'max_finite': mx,
        'mean_finite': mean,
    }


def is_tensor_finite(x):
    if not torch.is_tensor(x):
        return True
    return bool(torch.isfinite(x).all().item())


def check_output_finite(o):
    """Recursively check whether output o (tensor / tuple / list / dict) is finite."""
    if torch.is_tensor(o):
        return is_tensor_finite(o)
    elif isinstance(o, (list, tuple)):
        for v in o:
            if not check_output_finite(v):
                return False
        return True
    elif isinstance(o, dict):
        for v in o.values():
            if not check_output_finite(v):
                return False
        return True
    else:
        # non-tensor objects considered finite
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', required=True, help='Path to debug dump .pt file (from --debug_nans)')
    parser.add_argument('--ckpt', default='', help='Path to checkpoint.pth.tar (optional). If empty, looks for <dump_dir>/checkpoint.pth.tar')
    parser.add_argument('--params', default='', help='Path to params.pkl (optional). If empty, looks in dump dir for params.pkl')
    parser.add_argument('--device', default='cuda:0', help='Device to run replay on')
    parser.add_argument('--num_samples', type=int, default=8, help='How many samples from dump to use (<= saved samples)')
    parser.add_argument('--try_fp16', action='store_true', help='Also run replay under autocast (FP16) if available')
    parser.add_argument('--out', default='', help='Path to save results (json). Defaults to <dump>.find_bad_module.json')
    args = parser.parse_args()

    add_repo_paths()
    try:
        import main_swav  # used for some dataset helpers if needed
    except Exception:
        # not fatal
        main_swav = None
    try:
        import src.wt_models as wt_models
    except Exception as e:
        print('Failed to import src.wt_models:', e)
        raise
    from src.utils import restart_from_checkpoint

    dump = torch.load(args.dump, map_location='cpu')
    dump_dir = os.path.dirname(os.path.abspath(args.dump))

    # load params (args) if available
    params_path = args.params if args.params else os.path.join(dump_dir, 'params.pkl')
    params = None
    if os.path.isfile(params_path):
        try:
            with open(params_path, 'rb') as fh:
                params = pickle.load(fh)
            print('Loaded params from', params_path)
        except Exception as e:
            print('Failed to load params.pkl:', e)
    else:
        print('No params.pkl found at', params_path)

    # prepare inputs
    if 'inputs' in dump:
        inputs_saved = dump['inputs']
    else:
        # try older key
        inputs_saved = dump.get('input', None)
    if inputs_saved is None:
        raise RuntimeError('Dump file does not contain saved inputs; cannot replay')

    # inputs_saved may be a list of tensors (multi-crop) or a single tensor
    if isinstance(inputs_saved, (list, tuple)):
        # each element is (Nsave, C, H, W)
        # we will slice first num_samples along batch dim
        inputs_for_model = []
        for t in inputs_saved:
            inputs_for_model.append(t[: args.num_samples])
    elif torch.is_tensor(inputs_saved):
        inputs_for_model = inputs_saved[: args.num_samples]
    else:
        raise RuntimeError('Unexpected inputs format in dump')

    # load model args/constructor
    if params is None:
        raise RuntimeError('params.pkl required to reconstruct model (not found). Please provide --params pointing to the experiment params.pkl')

    # fix some attributes in params to be conservative for replay
    # ensure architecture attributes exist
    # build model similarly to main_swav
    arch = getattr(params, 'arch', 'wt_net')
    feat_dim = getattr(params, 'feat_dim', getattr(params, 'output_dim', 128))
    hidden_mlp = getattr(params, 'hidden_mlp', 512)
    nmb_prototypes = getattr(params, 'nmb_prototypes', 0)
    num_classes = getattr(params, 'num_classes', 0)

    # construct model
    if arch in wt_models.__dict__:
        model = wt_models.__dict__[arch](
            normalize=False,
            hidden_mlp=hidden_mlp,
            output_dim=feat_dim,
            nmb_prototypes=nmb_prototypes,
            eval_mode=True,
            num_classes=num_classes,
            use_spatial_attn=getattr(params, 'use_spatial_attn', False),
            attention_last_k=getattr(params, 'attention_last_k', 0),
            attn_reduction=getattr(params, 'attn_reduction', 8),
            attn_pool_size=getattr(params, 'attn_pool_size', 8),
            use_gem=getattr(params, 'use_gem', False),
            use_cosine_cls=getattr(params, 'use_cosine_cls', False),
            use_cbam=getattr(params, 'use_cbam', False),
            use_recon=getattr(params, 'use_recon', False),
            dropout=getattr(params, 'dropout', 0.0),
        )
    else:
        raise RuntimeError(f'Unsupported arch {arch} for replay script')

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model.to(device)
    model.eval()

    # load checkpoint weights if provided
    ckpt_path = args.ckpt if args.ckpt else os.path.join(dump_dir, 'checkpoint.pth.tar')
    if os.path.isfile(ckpt_path):
        print('Loading checkpoint', ckpt_path)
        # Prefer restart_from_checkpoint but fall back to manual permissive loading
        try:
            restart_from_checkpoint(ckpt_path, run_variables=None, state_dict=model)
        except Exception as e:
            print('restart_from_checkpoint failed (falling back to manual load):', e)
            traceback.print_exc()
            try:
                # Manual permissive loading: filter keys that match model.state_dict()
                # Try to load with weights_only=False for backward compatibility; handle
                # PyTorch safe-unpickling restrictions as a fallback.
                try:
                    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                except TypeError:
                    # older torch versions may not accept weights_only
                    ck = torch.load(ckpt_path, map_location='cpu')
                except Exception:
                    # Try allowing numpy scalar in safe globals (matches utils.py fallback)
                    try:
                        from torch.serialization import add_safe_globals
                        import numpy as _np
                        add_safe_globals([_np._core.multiarray.scalar])
                        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    except Exception:
                        # re-raise the original error to be handled by outer except
                        raise
                if isinstance(ck, dict) and 'state_dict' in ck:
                    sd = ck['state_dict']
                else:
                    sd = ck
                model_sd = model.state_dict()
                filtered = {}
                for k, v in sd.items():
                    kn = k
                    if kn.startswith('module.'):
                        kn = kn[len('module.'):]
                    if kn in model_sd and tuple(model_sd[kn].shape) == tuple(v.shape):
                        filtered[kn] = v
                if len(filtered) > 0:
                    print(f'Loading {len(filtered)} matching keys into model (manual load)')
                    model.load_state_dict(filtered, strict=False)
                else:
                    print('No matching keys found in checkpoint for model (manual load)')
            except Exception as e2:
                print('Manual checkpoint load failed:', e2)
                traceback.print_exc()
    else:
        print('No checkpoint found at', ckpt_path, '; proceeding with random init for missing layers')

    # prepare a small batch for replay
    if isinstance(inputs_for_model, list):
        # multi-crop: pass list to model
        inputs_device = [t.to(device) for t in inputs_for_model]
        bs = inputs_for_model[0].size(0)
    else:
        inputs_device = inputs_for_model.to(device)
        bs = inputs_for_model.size(0)

    results = {}
    for mode in ['fp32'] + (['fp16'] if args.try_fp16 else []):
        print('Running replay mode:', mode)
        acts = []  # list of (order_idx, module_name, stats, finite_bool)
        order = []

        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                try:
                    ok = check_output_finite(out)
                    st = None
                    # save only small stats to limit memory
                    if torch.is_tensor(out):
                        st = tensor_stats(out)
                    elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                        st = tensor_stats(out[0])
                    acts.append({'module': name, 'finite': ok, 'stats': st})
                except Exception as e:
                    acts.append({'module': name, 'finite': False, 'stats': {'error': str(e)}})
            return hook

        # register hooks on all named_modules (skip root '')
        named = list(model.named_modules())
        for n, m in named:
            if n == '':
                continue
            try:
                h = m.register_forward_hook(make_hook(n))
                hooks.append(h)
            except Exception:
                pass

        # run forward once and capture order
        try:
            with torch.no_grad():
                if mode == 'fp16':
                    try:
                        # newer API
                        from torch import amp
                        ctx = amp.autocast(device_type=device.type, enabled=True)
                    except Exception:
                        try:
                            from torch.cuda import amp as amp_local
                            ctx = amp_local.autocast(enabled=True)
                        except Exception:
                            ctx = None
                    if ctx is not None:
                        with ctx:
                            out = model(inputs_device)
                    else:
                        out = model(inputs_device)
                else:
                    out = model(inputs_device)
        except Exception as e:
            print('Forward raised exception during replay:', e)
            traceback.print_exc()
        finally:
            # remove hooks
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # find first module with finite==False
        bad_module = None
        for a in acts:
            if not a.get('finite', True):
                bad_module = a
                break
        results[mode] = {
            'num_hooks': len(acts),
            'first_bad': bad_module,
            'all_acts': acts[:200],  # cap to first 200 records to keep report small
        }
        print(f"Mode {mode}: hooks recorded={len(acts)}, first_bad={(bad_module['module'] if bad_module else None)}")

    out_path = args.out if args.out else args.dump + '.find_bad_module.json'
    try:
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)
        print('Saved report to', out_path)
    except Exception as e:
        print('Failed to write report json:', e)
        # fallback save torch
        torch.save(results, args.dump + '.find_bad_module.pt')
        print('Saved fallback torch report to', args.dump + '.find_bad_module.pt')


if __name__ == '__main__':
    main()

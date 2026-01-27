"""
Lightweight helper to inspect an augmented debug dump and guide full-step replay.

Usage:
    python scripts/replay_full_step.py /path/to/debug_nan_forward_epoch49_iter19.pt

This script currently:
 - Loads the dump and prints top-level keys and shapes
 - Checks for optimizer_state and amp_state presence (required for exact replay)
 - Locates a checkpoint in the same dump directory and reports it

Next steps (automated replay) will be implemented after we have an augmented dump
with optimizer + amp scaler state (this script helps confirm that the dump is adequate).
"""
import os
import sys
import torch
import pprint


def tensor_info(x):
    try:
        return {'shape': tuple(x.shape), 'dtype': str(x.dtype), 'numel': x.numel()}
    except Exception:
        return str(type(x))


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/replay_full_step.py /path/to/dump.pt')
        return
    dump_path = sys.argv[1]
    if not os.path.isfile(dump_path):
        print('Dump file not found:', dump_path)
        return
    try:
        d = torch.load(dump_path, map_location='cpu')
    except Exception as e:
        print('Initial torch.load failed:', repr(e))
        print('Retrying torch.load with weights_only=False (trusted local file).')
        try:
            d = torch.load(dump_path, map_location='cpu', weights_only=False)
        except TypeError:
            # older PyTorch may not accept weights_only arg — re-raise original
            raise
        except Exception as e2:
            # Last resort: try again allowing non-weights-only load
            try:
                torch.serialization.add_safe_globals([__import__('numpy').core.multiarray.scalar])
            except Exception:
                pass
            d = torch.load(dump_path, map_location='cpu', weights_only=False)
    print('Loaded dump:', dump_path)
    print('Top-level keys:')
    for k in sorted(d.keys()):
        v = d[k]
        if torch.is_tensor(v):
            info = tensor_info(v)
        elif isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
            info = [tensor_info(x) for x in v[:3]]
        else:
            info = str(type(v))
        print(f" - {k}: {info}")

    ok = True
    if 'optimizer_state' not in d:
        print('\nWARNING: optimizer_state not present in dump. Full-step replay may be impossible.')
        ok = False
    else:
        print('\noptimizer_state present in dump (good).')
    if 'amp_state' in d or 'amp_scaler_state' in d:
        print('AMP state present in dump (good).')
    else:
        print('AMP state NOT found in dump. If you used FP16, replay may not match runtime exactly.')
        ok = False

    # try to find checkpoint near dump
    dd = os.path.dirname(dump_path)
    candidates = ['checkpoint.pth.tar', 'checkpoint_latest.pth.tar', 'checkpoint_best.pth.tar']
    found_ckpt = None
    for c in candidates:
        p = os.path.join(dd, c)
        if os.path.isfile(p):
            found_ckpt = p
            break
    if found_ckpt:
        print('\nFound checkpoint candidate:', found_ckpt)
    else:
        print('\nNo checkpoint found in dump directory. You will need to provide the checkpoint used during training.')
        ok = False

    print('\nSummary:')
    if ok:
        print('Dump looks sufficient to attempt an exact offline full-step replay.')
        print('Next: run the full-step replay script (to be run) which will:')
        print('  - rebuild the model using saved params.pkl or args,')
        print('  - load checkpoint and optimizer/amp state,')
        print('  - run forward -> (scaled) backward -> unscale -> optimizer.step,')
        print('  - report which stage first becomes non-finite.')
    else:
        print('Dump is missing some required pieces (optimizer/amp/checkpoint).')
        print('Please re-run training with --debug_nans True to capture a richer dump, or provide the checkpoint file path.')


def run_full_step_replay(dump, dump_path, checkpoint_path=None, force_fp32=False, disable_larc=False):
    """Best-effort: reconstruct model+optimizer+amp and run forward->backward->step once.
    This attempts to mimic training step to locate which stage produces non-finite values.
    """
    import pickle
    import torch
    import torch.nn.functional as F
    from src import wt_models
    from src import compat_apex

    # load params.pkl if available
    params_pkl = os.path.join(os.path.dirname(dump_path), 'params.pkl')
    if not os.path.isfile(params_pkl):
        print('params.pkl not found; cannot reconstruct model automatically.')
        return
    args = pickle.load(open(params_pkl, 'rb'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Rebuilding model on device', device)

    # build model (support wt_models and resnet50 mapping similar to main_swav)
    if args.arch in wt_models.__dict__:
        model = wt_models.__dict__[args.arch](
            normalize=False,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
            eval_mode=False,
            num_classes=args.num_classes,
            use_spatial_attn=getattr(args, 'use_spatial_attn', True),
            attention_last_k=getattr(args, 'attention_last_k', 0),
            attn_reduction=getattr(args, 'attn_reduction', 8),
            attn_pool_size=getattr(args, 'attn_pool_size', 8),
            use_gem=getattr(args, 'use_gem', False),
            use_cosine_cls=getattr(args, 'use_cosine_cls', False),
            use_cbam=getattr(args, 'use_cbam', False),
            use_recon=getattr(args, 'use_recon', False),
            dropout=getattr(args, 'dropout', 0.0),
        )
    else:
        try:
            from src import resnet50 as resnet_models
            model = resnet_models.__dict__[args.arch](
                normalize=False,
                hidden_mlp=args.hidden_mlp,
                output_dim=args.feat_dim,
                nmb_prototypes=args.nmb_prototypes,
                num_classes=(args.num_classes if args.use_labels else 0),
            )
        except Exception as e:
            print('Failed to construct model:', e)
            return

    model = model.to(device)

    # allow overriding fp16 usage or disabling LARC for variant tests
    if force_fp32:
        print('Forcing FP32 mode for replay (overriding args.use_fp16)')
        setattr(args, 'use_fp16', False)

    # build optimizer; optionally wrap with LARC unless disabled
    optim = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    if disable_larc:
        print('Running replay with LARC disabled')
    else:
        optim = compat_apex.LARC(optimizer=optim, trust_coefficient=0.001, clip=False)

    # amp fallback
    amp = compat_apex.amp

    # load checkpoint states if present
    if checkpoint_path is None:
        # try default checkpoint in same dir
        possible = os.path.join(os.path.dirname(dump_path), 'checkpoint.pth.tar')
        if os.path.isfile(possible):
            checkpoint_path = possible
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print('Loading checkpoint:', checkpoint_path)
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'state_dict' in ck:
            try:
                model.load_state_dict(ck['state_dict'], strict=False)
                print('Loaded model state')
            except Exception as e:
                print('Model load warning:', e)
        if 'optimizer' in ck:
            try:
                # if LARC was disabled in this replay, the checkpoint optimizer state
                # may not match exactly; attempt to load and warn on mismatch.
                optim.load_state_dict(ck['optimizer'])
                print('Loaded optimizer state')
            except Exception as e:
                print('Optimizer load warning:', e)
        if 'amp' in ck:
            try:
                amp.load_state_dict(ck['amp'])
                print('Loaded amp state')
            except Exception as e:
                print('AMP load warning:', e)

    model.train()

    # prepare inputs from dump
    inputs = dump.get('inputs', None)
    targets = dump.get('targets', None)
    if inputs is None:
        print('No inputs in dump; cannot replay forward.')
        return
    # move inputs to device; inputs may be list of crops
    if isinstance(inputs, (list, tuple)):
        inputs_for_model = [t.to(device) for t in inputs]
        # if model expects a single tensor of concatenated crops, pass as list
        inputs_pass = inputs_for_model
    else:
        inputs_pass = inputs.to(device)

    # Run forward
    try:
        retval = model(inputs_pass)
    except Exception as e:
        print('Forward raised exception:', repr(e))
        return

    # Extract embeddings/outputs similar to training loop
    if isinstance(retval, (tuple, list)):
        if len(retval) == 4:
            embedding, output, cls_logits_full, _semantic = retval
        elif len(retval) == 3:
            embedding, output, cls_logits_full = retval
        elif len(retval) == 2:
            embedding, output = retval
            if args.use_labels and isinstance(output, torch.Tensor) and output.dim() == 2 and output.size(1) == args.num_classes:
                cls_logits_full = output
            else:
                cls_logits_full = None
        else:
            print('Unexpected model forward return length:', len(retval))
            return
    else:
        embedding = retval
        output = None

    # Check forward numeric validity
    def is_finite(t):
        try:
            return torch.isfinite(t).all().item()
        except Exception:
            return False

    if not is_finite(embedding):
        print('Non-finite detected at FORWARD (embedding)')
        return
    if output is not None and not is_finite(output):
        print('Non-finite detected at FORWARD (output)')
        return

    # construct a simple scalar loss to exercise backward/step
    if cls_logits_full is not None and targets is not None:
        t = targets.to(device)
        try:
            loss = F.cross_entropy(cls_logits_full[:t.size(0)], t)
        except Exception:
            loss = embedding.norm()
    else:
        # fallback surrogate loss
        loss = embedding.norm()

    print('Computed surrogate loss:', float(loss.detach().cpu().item()))

    # backward with AMP if requested
    try:
        optim.zero_grad()
    except Exception:
        pass
    try:
        if getattr(args, 'use_fp16', False):
            # amp may be a noop wrapper if not present; scale_loss context manager expected
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    except Exception as e:
        print('Backward raised exception:', repr(e))
        return

    # check gradients
    any_bad_grad = False
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            print('Non-finite gradient in param', n)
            any_bad_grad = True
            break
    if any_bad_grad:
        print('Non-finite detected at BACKWARD (grad)')
        return

    # try optimizer.step()
    try:
        optim.step()
    except Exception as e:
        print('Optimizer.step raised exception:', repr(e))
        return

    # check parameters after step
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all():
            print('Non-finite detected after OPTIMIZER.STEP for param', n)
            return

    print('Full-step replay completed: no NaNs found in forward/backward/step for this surrogate replay.')


if __name__ == '__main__':
    # allow optional --replay flag as second arg
    if len(sys.argv) >= 3 and sys.argv[2] == '--replay':
        # robustly load the dump (same logic as main()) then run replay
        dump_path = sys.argv[1]
        if not os.path.isfile(dump_path):
            print('Dump file not found:', dump_path)
            sys.exit(1)
        try:
            d = torch.load(dump_path, map_location='cpu')
        except Exception as e:
            # retry with weights_only=False like in main()
            try:
                print('Initial torch.load failed:', repr(e))
                print('Retrying torch.load with weights_only=False (trusted local file).')
                d = torch.load(dump_path, map_location='cpu', weights_only=False)
            except Exception:
                try:
                    # best-effort allowlist fallback for numpy scalar globals
                    torch.serialization.add_safe_globals([__import__('numpy').core.multiarray.scalar])
                except Exception:
                    pass
                d = torch.load(dump_path, map_location='cpu', weights_only=False)

        # find checkpoint
        dd = os.path.dirname(dump_path)
        ck = os.path.join(dd, 'checkpoint.pth.tar') if os.path.isfile(os.path.join(dd, 'checkpoint.pth.tar')) else None
        # parse optional flags
        force_fp32 = '--force-fp32' in sys.argv
        disable_larc = '--disable-larc' in sys.argv
        run_full_step_replay(d, dump_path, ck, force_fp32=force_fp32, disable_larc=disable_larc)
    else:
        main()

"""
DDP-aware full-step replay.
This script is intended to be launched with torch.distributed.run / torchrun
so that it spawns multiple processes (one per GPU) and reproduces a single
training step (forward -> backward -> optimizer.step) in a distributed
context using the saved dump + checkpoint from the experiment.

Example:
  PYTHONPATH=/root/autodl-tmp/data/swav-main/swav-main \
  python -m torch.distributed.run --nproc_per_node=3 scripts/replay_full_step_ddp.py /path/to/dump.pt --replay

The script will print per-rank detection logs indicating at which stage
(forward / backward gradients / optimizer.step) non-finite values first appear.
"""
import os
import sys
import time
import torch
import torch.distributed as dist
import traceback

# Minimal helpers copied/adapted from replay_full_step.py

def tensor_info(x):
    try:
        return {'shape': tuple(x.shape), 'dtype': str(x.dtype), 'numel': x.numel()}
    except Exception:
        return str(type(x))


def safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        # retry with weights_only=False (trusted local file)
        try:
            print('[loader] retrying torch.load with weights_only=False:', repr(e))
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            try:
                torch.serialization.add_safe_globals([__import__('numpy').core.multiarray.scalar])
            except Exception:
                pass
            return torch.load(path, map_location='cpu', weights_only=False)


def is_finite_tensor(t):
    try:
        return torch.isfinite(t).all().item()
    except Exception:
        return False


def main():
    if len(sys.argv) < 2:
        print('Usage: scripts/replay_full_step_ddp.py /path/to/dump.pt [--disable-larc] [--force-fp32]')
        return
    dump_path = sys.argv[1]
    disable_larc = '--disable-larc' in sys.argv
    force_fp32 = '--force-fp32' in sys.argv

    # Distributed init
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')

    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device) if torch.cuda.is_available() else None

    def log(*args, **kwargs):
        print(f"[R{rank}]", *args, **kwargs)
        sys.stdout.flush()

    log('Starting DDP replay: rank', rank, 'local_rank', local_rank, 'world_size', world_size)

    if not os.path.isfile(dump_path):
        log('Dump not found:', dump_path)
        return

    # load dump (cpu) then move things to device later
    dump = safe_torch_load(dump_path)
    log('Loaded dump keys:', sorted(list(dump.keys())))

    # load params.pkl
    pkl_path = os.path.join(os.path.dirname(dump_path), 'params.pkl')
    if not os.path.isfile(pkl_path):
        log('params.pkl not found; cannot reconstruct model automatically.')
        return
    import pickle
    args = pickle.load(open(pkl_path, 'rb'))

    # import model factory
    try:
        # ensure project src is importable
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swav-main')))
    except Exception:
        pass

    try:
        from src import wt_models, compat_apex
    except Exception as e:
        # try to import from top-level src
        try:
            from src import wt_models, compat_apex
        except Exception as e2:
            log('Failed to import src modules:', e2)
            return

    # build model
    try:
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
            )
        else:
            from src import resnet50 as resnet_models
            model = resnet_models.__dict__[args.arch](
                normalize=False,
                hidden_mlp=args.hidden_mlp,
                output_dim=args.feat_dim,
                nmb_prototypes=args.nmb_prototypes,
                num_classes=(args.num_classes if args.use_labels else 0),
            )
    except Exception as e:
        log('Failed to construct model:', e)
        traceback.print_exc()
        return

    model = model.to(device)

    # optionally force FP32
    if force_fp32:
        log('Forcing FP32 mode for replay (overriding args.use_fp16)')
        setattr(args, 'use_fp16', False)

    # optimizer and optional LARC
    optim = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    if not disable_larc:
        try:
            optim = compat_apex.LARC(optimizer=optim, trust_coefficient=0.001, clip=False)
        except Exception:
            log('LARC wrapper not available; continuing with raw optimizer')
    else:
        log('LARC disabled for this replay')

    amp = compat_apex.amp

    # load checkpoint if available
    ckpt = os.path.join(os.path.dirname(dump_path), 'checkpoint.pth.tar')
    if os.path.isfile(ckpt):
        log('Loading checkpoint:', ckpt)
        try:
            ck = torch.load(ckpt, map_location=device, weights_only=False)
        except Exception:
            try:
                torch.serialization.add_safe_globals([__import__('numpy').core.multiarray.scalar])
            except Exception:
                pass
            ck = torch.load(ckpt, map_location=device, weights_only=False)
        if 'state_dict' in ck:
            try:
                model.load_state_dict(ck['state_dict'], strict=False)
                log('Loaded model state')
            except Exception as e:
                log('Model load warning:', e)
        if 'optimizer' in ck:
            try:
                optim.load_state_dict(ck['optimizer'])
                log('Loaded optimizer state')
            except Exception as e:
                log('Optimizer load warning:', e)
        if 'amp' in ck:
            try:
                amp.load_state_dict(ck['amp'])
                log('Loaded amp state')
            except Exception as e:
                log('AMP load warning:', e)

    # wrap in DDP
    try:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if torch.cuda.is_available() else None)
        log('Wrapped model in DDP')
    except Exception as e:
        log('Failed to wrap DDP:', e)

    model.train()

    # prepare inputs
    inputs = dump.get('inputs', None)
    targets = dump.get('targets', None)
    if inputs is None:
        log('No inputs in dump; cannot replay forward.')
        return

    if isinstance(inputs, (list, tuple)):
        inputs_for_model = [t.to(device) for t in inputs]
        inputs_pass = inputs_for_model
    else:
        inputs_pass = inputs.to(device)

    # run forward
    try:
        log('Running forward...')
        retval = model(inputs_pass)
    except Exception as e:
        log('Forward raised exception:', repr(e))
        traceback.print_exc()
        return

    # extract embedding/output similar to main
    embedding = None
    output = None
    cls_logits_full = None
    if isinstance(retval, (tuple, list)):
        if len(retval) == 4:
            embedding, output, cls_logits_full, _ = retval
        elif len(retval) == 3:
            embedding, output, cls_logits_full = retval
        elif len(retval) == 2:
            embedding, output = retval
            if args.use_labels and isinstance(output, torch.Tensor) and output.dim() == 2 and output.size(1) == args.num_classes:
                cls_logits_full = output
        else:
            log('Unexpected forward return length:', len(retval))
            return
    else:
        embedding = retval

    # check forward numeric validity
    if not is_finite_tensor(embedding):
        log('Non-finite detected at FORWARD (embedding)')
        return
    if output is not None and not is_finite_tensor(output):
        log('Non-finite detected at FORWARD (output)')
        return

    # build surrogate loss
    import torch.nn.functional as F
    if cls_logits_full is not None and targets is not None:
        t = targets.to(device)
        try:
            loss = F.cross_entropy(cls_logits_full[:t.size(0)], t)
        except Exception:
            loss = embedding.norm()
    else:
        loss = embedding.norm()

    log('Surrogate loss:', float(loss.detach().cpu().item()))

    # backward
    try:
        optim.zero_grad()
    except Exception:
        pass

    try:
        if getattr(args, 'use_fp16', False):
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    except Exception as e:
        log('Backward raised exception:', repr(e))
        traceback.print_exc()
        return

    # check gradients
    any_bad = False
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            log('Non-finite gradient detected in param', n)
            any_bad = True
            break
    if any_bad:
        log('Non-finite detected at BACKWARD (grad)')
        return

    # optimizer step
    try:
        optim.step()
    except Exception as e:
        log('Optimizer.step raised exception:', repr(e))
        traceback.print_exc()
        return

    # check parameters after step
    for n, p in model.named_parameters():
        try:
            if not torch.isfinite(p).all():
                log('Non-finite detected after OPTIMIZER.STEP for param', n)
                return
        except Exception:
            pass

    log('Rank completed full-step replay: no NaNs found in forward/backward/step')

    # barrier to ensure all ranks finish
    try:
        dist.barrier()
    except Exception:
        pass


if __name__ == '__main__':
    main()

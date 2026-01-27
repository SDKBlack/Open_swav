#!/usr/bin/env python3
import os, sys, json
import argparse
import torch
import numpy as np

# helper to find prototype weight in checkpoint state_dict

def find_prototype_weight(sd):
    for k, v in sd.items():
        kl = k.lower()
        if 'protot' in kl and 'weight' in kl:
            arr = v.cpu().numpy()
            if arr.ndim == 2:
                return k, arr
    for k, v in sd.items():
        arr = v.cpu().numpy()
        if arr.ndim == 2:
            return k, arr
    return None, None


def build_dataset(main_swav_mod, index_file, data_root, size=256):
    # ensure main_swav has a minimal args object for subset_ratio and rank
    if not hasattr(main_swav_mod, 'args'):
        class A: pass
        main_swav_mod.args = A()
        main_swav_mod.args.subset_ratio = 1.0
        main_swav_mod.args.rank = 0
    Dataset = main_swav_mod.IndexMultiCropDataset
    ds = Dataset(
        index_file,
        data_root,
        size_crops=[size],
        nmb_crops=[1],
        min_scale_crops=[1.0],
        max_scale_crops=[1.0],
        return_label=True,
        deterministic_single_crop=True,
        ce_use_original=True,
        as_1d=False,
    )
    return ds


def collate_fn(batch):
    # batch: list of tuples (multi_crops, label) because return_label True
    imgs = []
    labels = []
    for item in batch:
        multi, lbl = item
        # multi is list of tensors (n_crops,), we requested single crop so take multi[0]
        img = multi[0]
        imgs.append(img.unsqueeze(0))
        labels.append(int(lbl))
    imgs = torch.cat(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.int64)
    return imgs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--index_train', required=True)
    parser.add_argument('--data_root', required=False, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--arch', default='wt_net')
    parser.add_argument('--device', default='cuda', help='device to run inference on (cuda or cpu)')
    args = parser.parse_args()

    # Ensure the repo's swav-main folder is on PYTHONPATH so we can import main_swav
    repo_root = os.path.dirname(os.path.realpath(__file__))
    probable_paths = [
        os.path.join(repo_root, 'data', 'swav-main', 'swav-main'),
        os.path.join(repo_root, 'swav-main', 'swav-main'),
        repo_root,
        os.getcwd(),
    ]
    for p in probable_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    # import main_swav for dataset helper; import wt_models for model
    try:
        import main_swav
    except Exception as e:
        print('Failed to import main_swav from known locations. Tried:', probable_paths)
        raise
    import src.wt_models as wt_models

    print('Loading dataset...')
    ds = build_dataset(main_swav, args.index_train, args.data_root)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    print('Dataset size:', len(ds))

    print('Loading checkpoint:', args.ckpt)
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu')
    except Exception:
        # Some PyTorch versions restrict allowed globals when loading pickled checkpoints.
        # Retry with weights_only=False to allow full legacy checkpoints (trusted source assumed).
        try:
            ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        except TypeError:
            # older torch may not accept weights_only kwarg; re-raise original
            raise
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    key, proto_w = find_prototype_weight(sd)
    if key is None:
        print('No prototype weight found in checkpoint')
        return
    print('Found prototype weight key:', key, 'shape:', proto_w.shape)

    feat_dim = proto_w.shape[1]
    n_proto = proto_w.shape[0]

    # instantiate WTNet to compute embeddings
    # choose conservative args: in_channels=1, output_dim=feat_dim
    # decide device and build model there
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print(f'Building WTNet to compute embeddings (device={device})...')
    model = wt_models.__dict__[args.arch](in_channels=1, input_size=[256,256], semantic_dim=128, num_class=0, wt_levels=1, use_spatial_attn=False, output_dim=feat_dim, hidden_mlp=512, nmb_prototypes=0, normalize=False, eval_mode=True)
    model.to(device)
    model.eval()

    # load corresponding weights for projection/head/backbone if present in checkpoint
    # We'll attempt to load state_dict keys that match model
    model_sd = model.state_dict()
    # prepare a filtered state_dict
    filtered = {}
    for k, v in sd.items():
        # strip possible 'module.' prefix
        kn = k
        if kn.startswith('module.'):
            kn = kn[len('module.'):]
        if kn in model_sd and v.shape == model_sd[kn].shape:
            filtered[kn] = v
    if len(filtered) > 0:
        print(f'Loading {len(filtered)} matching keys into model')
        model.load_state_dict(filtered, strict=False)
    else:
        print('No matching model weights found to load; proceeding with random init for backbone/proj (embeddings may be meaningless)')

    # compute embeddings for all training samples
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            # imgs: (B, C, H, W)
            imgs = imgs.to(device)
            out = model(imgs)
            # model returns (embedding, prototype_output, cls_logits, semantic) for WTNet
            if isinstance(out, (list, tuple)):
                embedding = out[0]
            else:
                embedding = out
            embedding = embedding.cpu().numpy()
            all_feats.append(embedding)
            all_labels.append(labels.numpy())
    if len(all_feats) == 0:
        print('No features computed (empty dataset?)')
        return
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print('Computed embeddings:', feats.shape)

    # compute logits and assignments
    # proto_w shape: (n_proto, feat_dim)
    logits = feats.dot(proto_w.T)
    proto_ids = np.argmax(logits, axis=1)

    # build mapping
    mapping = []
    for pid in range(n_proto):
        idx = np.where(proto_ids == pid)[0]
        support = int(len(idx))
        counts = {}
        maj_class = None
        maj_count = 0
        if support > 0:
            assigned_labels = labels[idx]
            unique, cts = np.unique(assigned_labels, return_counts=True)
            for u, c in zip(unique, cts):
                counts[int(u)] = int(c)
                if c > maj_count:
                    maj_count = int(c)
                    maj_class = int(u)
        purity = float(maj_count / support) if support > 0 else 0.0
        mapping.append({'prototype': int(pid), 'support': support, 'majority_class': maj_class, 'majority_count': int(maj_count), 'purity': float(purity), 'class_counts': counts})

    out_json = os.path.join(args.out_dir, 'prototype_to_class_mapping_ckpt.json')
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump({'checkpoint': args.ckpt, 'mapping': mapping}, fh, indent=2)
    out_npz = os.path.join(args.out_dir, 'prototype_mapping_ckpt.npz')
    np.savez_compressed(out_npz, mapping=mapping, proto_w=proto_w, feats=feats, labels=labels)

    print('Saved mapping to', out_json)
    # print problematic prototypes
    print('\nPrototype | support | purity | majority_class | majority_count')
    low_usage = []
    low_purity = []
    for m in mapping:
        print(f"{m['prototype']:9d} | {m['support']:7d} | {m['purity']:.4f} | {str(m['majority_class']):14s} | {m['majority_count']:14d}")
        if m['support'] < 50:
            low_usage.append((m['prototype'], m['support']))
        if m['purity'] < 0.7:
            low_purity.append((m['prototype'], m['support'], m['majority_class'], m['purity']))

    print('\nPrototypes with low usage (<50):')
    print(low_usage)
    print('\nPrototypes with low purity (<0.7):')
    print(low_purity)
    print('\nnpz saved at', out_npz)

if __name__ == '__main__':
    main()

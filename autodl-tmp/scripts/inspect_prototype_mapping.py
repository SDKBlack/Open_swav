#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import torch

def find_prototype_weight(sd):
    # sd: state_dict mapping
    for k, v in sd.items():
        kl = k.lower()
        if 'protot' in kl and 'weight' in kl:
            arr = v.cpu().numpy()
            if arr.ndim == 2:
                return k, arr
    # fallback: look for any 2D weight with second dim matching feat dim later
    for k, v in sd.items():
        arr = v.cpu().numpy()
        if arr.ndim == 2:
            return k, arr
    return None, None


def main():
    if len(sys.argv) < 3:
        print('Usage: inspect_prototype_mapping.py <exp_dir_with_test_X> <checkpoint_path>')
        sys.exit(2)
    exp_dir = sys.argv[1]
    ckpt_path = sys.argv[2]

    fx = os.path.join(exp_dir, 'test_X.npy')
    fy = os.path.join(exp_dir, 'test_Y.npy')
    if not os.path.exists(fx):
        print('test_X.npy not found in', exp_dir)
        sys.exit(1)
    test_X = np.load(fx)
    if os.path.exists(fy):
        test_Y = np.load(fy)
    else:
        test_Y = None

    print('Loaded test_X shape:', test_X.shape)
    if test_Y is not None:
        print('Loaded test_Y shape:', test_Y.shape)

    if not os.path.exists(ckpt_path):
        print('Checkpoint not found:', ckpt_path)
        sys.exit(1)
    # Some PyTorch versions require weights_only=False to load full pickled checkpoints
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            # older torch versions may not accept weights_only kwarg; re-raise original error
            raise
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    # if sd keys are prefixed with module., strip for convenience
    # But keep original keys for lookups
    key, w = find_prototype_weight(sd)
    if key is None:
        print('No prototype-like weight found in checkpoint state_dict')
        sys.exit(1)
    print('Found prototype weight key:', key, 'shape:', w.shape)

    # Determine orientation
    # weight shape typically (out_features, in_features) for nn.Linear(in_features, out_features)
    # so prototypes: Linear(128 -> 54) => weight shape (54,128)
    if w.shape[1] == test_X.shape[1]:
        # w is (n_prot, feat_dim)
        proto_w = w
        logits = test_X.dot(proto_w.T)
    elif w.shape[0] == test_X.shape[1]:
        # w is transposed
        proto_w = w.T
        logits = test_X.dot(proto_w.T)
    else:
        # attempt to reshape if possible
        print('Warning: prototype weight dim does not match test_X dim; trying transpose fallback')
        try:
            proto_w = w
            logits = test_X.dot(proto_w.T)
        except Exception as e:
            print('Failed to compute logits:', e)
            sys.exit(1)

    proto_ids = np.argmax(logits, axis=1)

    # infer num_known from labels if possible
    if test_Y is not None:
        try:
            maxlabel = int(np.nanmax(test_Y))
            num_known = int(np.max(test_Y[test_Y >= 0])) + 1 if np.any(test_Y >= 0) else None
        except Exception:
            num_known = None
    else:
        num_known = None

    n_proto = proto_w.shape[0]
    mapping = []
    # For each prototype, compute counts for known classes
    for pid in range(n_proto):
        idx = np.where(proto_ids == pid)[0]
        support = len(idx)
        counts = {}
        maj_class = None
        maj_count = 0
        if support > 0 and test_Y is not None:
            # restrict to known labels (>=0)
            labels = test_Y[idx]
            # if labels contain values >= num_known, normalize unknowns to -1
            if num_known is not None:
                known_mask = labels < num_known
                # focus on known labels
                labels_known = labels[known_mask]
            else:
                labels_known = labels[labels >= 0]
            unique, cts = np.unique(labels_known, return_counts=True)
            for u,c in zip(unique, cts):
                counts[int(u)] = int(c)
                if c > maj_count:
                    maj_count = int(c)
                    maj_class = int(u)
        purity = float(maj_count / support) if support>0 else 0.0
        mapping.append({'prototype': int(pid), 'support': int(support), 'majority_class': maj_class, 'majority_count': int(maj_count), 'purity': float(purity), 'class_counts': counts})

    out_path = os.path.join(exp_dir, 'prototype_to_class_mapping.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump({'checkpoint': ckpt_path, 'mapping': mapping}, fh, indent=2)
    print('Saved mapping to', out_path)

    # print concise table sorted by prototype id
    print('\nPrototype | support | purity | majority_class | majority_count')
    for m in mapping:
        print(f"{m['prototype']:9d} | {m['support']:7d} | {m['purity']:.4f} | {str(m['majority_class']):14s} | {m['majority_count']:14d}")

if __name__ == '__main__':
    main()

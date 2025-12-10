#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

recon_dir = '/root/autodl-tmp/recon_examples'
out_dir = os.path.join(recon_dir, 'plots_robust')
os.makedirs(out_dir, exist_ok=True)

# find bases
bases = []
for fn in sorted(os.listdir(recon_dir)):
    if fn.endswith('_recon_clean.npy'):
        bases.append(fn[:-len('_recon_clean.npy')])

if not bases:
    print('No recon clean files found in', recon_dir)
    raise SystemExit(0)

print('Found bases:', bases[:10])

for base in bases[:6]:
    files = {
        'clean': os.path.join(recon_dir, base + '_recon_clean.npy'),
        '0dB': os.path.join(recon_dir, base + '_recon_snr0dB.npy'),
        '-5dB': os.path.join(recon_dir, base + '_recon_snr-5dB.npy')
    }
    arrs = {}
    for k, p in files.items():
        if os.path.exists(p):
            a = np.load(p)
            # if 1D, skip and mark as missing
            if a.ndim == 1:
                print('Note: file', p, 'is 1D (waveform). Skipping panel', k)
                arrs[k] = None
            else:
                arrs[k] = a.astype(np.float32)
        else:
            arrs[k] = None

    # determine vmax across available arrays on log1p scale
    candidates = [np.max(np.log1p(a)) for a in arrs.values() if a is not None]
    vmax = max(candidates) if candidates else None

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    labels = ['clean', '0 dB', '-5 dB']
    for ax, lab in zip(axs, labels):
        arr = arrs.get(lab if lab=='clean' else (lab if lab=='0 dB' else '-5 dB'))
        if arr is None:
            ax.text(0.5, 0.5, 'missing or 1D', ha='center', va='center')
            ax.axis('off')
            ax.set_title(lab)
            continue
        ax.imshow(np.log1p(arr), origin='lower', aspect='auto', vmax=vmax)
        ax.set_title(lab)
        ax.set_xlabel('frames')
        ax.set_ylabel('freq bins')
    fig.suptitle(base)
    plt.tight_layout()
    outpath = os.path.join(out_dir, f'{base}_recon_compare.png')
    plt.savefig(outpath)
    plt.close(fig)
    print('Saved', outpath)

print('Done')

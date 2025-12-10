#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--recon_dir', type=str, default='/root/autodl-tmp/recon_userparams')
parser.add_argument('--out_dir', type=str, default=None)
parser.add_argument('--n', type=int, default=12)
args = parser.parse_args()

recon_dir = args.recon_dir
out_dir = args.out_dir or os.path.join(recon_dir, 'plots_filtered')
os.makedirs(out_dir, exist_ok=True)

# gather candidate bases where all three mag files exist and are 2D
files = sorted(os.listdir(recon_dir))
bases = []
for fn in files:
    if not fn.endswith('_recon_clean.npy'):
        continue
    base = fn[:-len('_recon_clean.npy')]
    fn0 = os.path.join(recon_dir, base + '_recon_snr0dB.npy')
    fnm5 = os.path.join(recon_dir, base + '_recon_snr-5dB.npy')
    fnc = os.path.join(recon_dir, base + '_recon_clean.npy')
    if os.path.exists(fn0) and os.path.exists(fnm5) and os.path.exists(fnc):
        try:
            a = np.load(fnc)
            b = np.load(fn0)
            c = np.load(fnm5)
            if a.ndim == 2 and b.ndim == 2 and c.ndim == 2:
                bases.append(base)
        except Exception:
            continue

if not bases:
    print('No complete 2D spectrogram bases found in', recon_dir)
    raise SystemExit(0)

for base in bases[:args.n]:
    fn_clean = os.path.join(recon_dir, base + '_recon_clean.npy')
    fn_0 = os.path.join(recon_dir, base + '_recon_snr0dB.npy')
    fn_m5 = os.path.join(recon_dir, base + '_recon_snr-5dB.npy')
    mag_clean = np.load(fn_clean).astype(np.float32)
    mag_0 = np.load(fn_0).astype(np.float32)
    mag_m5 = np.load(fn_m5).astype(np.float32)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['clean', '0 dB', '-5 dB']
    for ax, arr, t in zip(axs, [mag_clean, mag_0, mag_m5], titles):
        im = ax.imshow(np.log1p(arr), origin='lower', aspect='auto')
        ax.set_title(t)
        ax.set_xlabel('frames')
        ax.set_ylabel('freq bins')
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle(base)
    outpath = os.path.join(out_dir, f'{base}_recon_compare.png')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)
    print('Saved', outpath)

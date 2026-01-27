#!/usr/bin/env python3
"""
Small launcher to run training for one or more experiment group index files.

This script intentionally does NOT modify `train.py`. It prepares the working
directory so that the relative paths used by `train.py` (./experiment_groups and
./Data) resolve to the user-provided directories, then imports `train` and calls
`train.train(...)` for each requested index id.

Usage examples:
  # run a single index
  python run_experiments.py --index-ids 1 --data-dir /root/autodl-tmp/S3R/Data --exp-dir /root/autodl-tmp/S3R/experiment_groups

  # run several indices
  python run_experiments.py --index-ids 1,2,3

  # run all indices found under an experiment-groups directory
  python run_experiments.py --all --exp-dir /path/to/exp_groups

Note: `train.train` contains the full training loop (max_epoch is set inside
that function). This launcher does not change the training schedule; it only
automates selecting index files and wiring paths so you don't need to modify
`train.py`.
"""
import argparse
import os
import sys
import shutil
from pathlib import Path


def ensure_symlink(target: str, link_name: str):
    """Create or update a symlink at link_name that points to target."""
    target = os.path.abspath(target)
    link_name = os.path.abspath(link_name)
    # If link exists and points to the same target, nothing to do
    if os.path.islink(link_name):
        current = os.readlink(link_name)
        if os.path.abspath(current) == target:
            return
        os.unlink(link_name)
    # If a real directory/file exists at link_name, back it up (rename)
    if os.path.exists(link_name):
        backup = link_name + '.backup'
        if os.path.exists(backup):
            shutil.rmtree(backup)
        os.rename(link_name, backup)
    os.symlink(target, link_name)


def find_available_indices(exp_dir: str):
    # look for files like '1-known_for_train' and extract the numeric prefix
    ids = set()
    for p in Path(exp_dir).iterdir():
        name = p.name
        if name.startswith('.'):
            continue
        parts = name.split('-')
        if len(parts) >= 2 and parts[0].isdigit():
            ids.add(int(parts[0]))
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description="Run training for selected experiment-group indices (wrapper around train.train)")
    parser.add_argument('--index-ids', type=str, default='', help='Comma-separated list of integer index ids to run (e.g. 1,2,3)')
    parser.add_argument('--all', action='store_true', help='Run all indices found in experiment-groups dir')
    parser.add_argument('--exp-dir', type=str, default='./experiment_groups', help='Path to experiment_groups directory containing index files')
    parser.add_argument('--data-dir', type=str, default='./Data', help='Path to dataset directory (used by index files)')
    # training hyperparams forwarded to train.train
    parser.add_argument('--device', type=str, default='cuda:0', help='torch device')
    parser.add_argument('--semantic-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--margin', type=float, default=8)
    parser.add_argument('--num-known', type=int, default=18)
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument('--len-time', type=int, default=1)
    parser.add_argument('--tips', type=str, default='(run_experiments)')

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    # Ensure paths used by train.py resolve: ./experiment_groups and ./Data
    if os.path.abspath(args.exp_dir) != os.path.abspath(os.path.join(repo_root, 'experiment_groups')):
        if not os.path.exists(args.exp_dir):
            raise FileNotFoundError(f"Provided experiment-groups dir does not exist: {args.exp_dir}")
        ensure_symlink(args.exp_dir, os.path.join(repo_root, 'experiment_groups'))

    if os.path.abspath(args.data_dir) != os.path.abspath(os.path.join(repo_root, 'Data')):
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Provided data dir does not exist: {args.data_dir}")
        ensure_symlink(args.data_dir, os.path.join(repo_root, 'Data'))

    # Determine indices to run
    if args.all:
        ids = find_available_indices(os.path.join(repo_root, 'experiment_groups'))
    else:
        if not args.index_ids:
            parser.error('Either --index-ids or --all must be provided')
        ids = [int(x) for x in args.index_ids.split(',') if x.strip()]

    if len(ids) == 0:
        print('No experiment indices found, exiting.')
        sys.exit(0)

    print(f'Will run indices: {ids}')

    # import train module (do not execute as script)
    import train as tr
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    for idx in ids:
        print(f'=== Starting training for index {idx} ===')
        # ensure model and semantic dirs exist (train.py expects these)
        os.makedirs('./model/S3R/', exist_ok=True)
        os.makedirs('./semantic/S3R/', exist_ok=True)

        Net = tr.NET(in_channels=1, input_size=[int(512 * args.len_time), 512], semantic_dim=args.semantic_dim,
                     num_class=args.num_known, device=device).to(device)

        # Call train.train (this runs the full training loop inside train.py)
        tr.train(net=Net, device=device, semantic_dims=args.semantic_dim, lr=args.lr,
                 batch_size=args.batch_size, margin=args.margin, num_known_class=args.num_known,
                 my_index=idx, gamma=args.gamma, len_time=args.len_time, tips=args.tips)

        print(f'=== Finished training for index {idx} ===')


if __name__ == '__main__':
    main()

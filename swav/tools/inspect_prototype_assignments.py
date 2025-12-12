"""Inspect which samples are assigned to which prototypes (non-distributed).

Usage example:
python swav/tools/inspect_prototype_assignments.py \
  --data_path /root/autodl-tmp/S3R \
  --split_path /root/autodl-tmp/S3R/experiment_groups/1-known_for_train \
  --batch_size 128 \
  --nmb_prototypes 45 \
  --n_batches 1

The script loads one (or n) batches from the dataset, runs the model forward to obtain
projection outputs, computes a local Sinkhorn assignment (no distributed ops), and
prints for each prototype the list of global sample indices assigned to it (by argmax).
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.s3r_dataset import S3RDataset
from src.wtnet import WTNet


def local_sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    """Local (non-distributed) sinkhorn implementation that mirrors main_swav.distributed_sinkhorn.
    out: [B, K] logits
    returns: Q.t().t() -> [B, K] soft assignments (rows sum to 1)
    """
    Q = torch.exp(out / epsilon).t()  # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    Q /= torch.sum(Q)

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()  # B x K


class IndexedS3R(S3RDataset):
    """Wrap S3RDataset to also return the sample index for mapping."""
    def __getitem__(self, index):
        if self.is_train:
            multi_crops, label = super(IndexedS3R, self).__getitem__(index)
            return multi_crops, label, index
        else:
            out, label = super(IndexedS3R, self).__getitem__(index)
            return out, label, index


def collate_fn_train(batch):
    # batch: list of (multi_crops, label, idx)
    n = len(batch)
    multi = batch[0][0]
    n_crops = len(multi)
    # create list of tensors per crop
    crops = [torch.stack([batch[i][0][j] for i in range(n)]) for j in range(n_crops)]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    indices = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return crops, labels, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_batches", type=int, default=1)
    parser.add_argument("--nmb_prototypes", type=int, default=45)
    parser.add_argument("--size_crops", type=int, nargs='+', default=[224])
    parser.add_argument("--nmb_crops", type=int, nargs='+', default=[2])
    parser.add_argument("--min_scale_crops", type=float, nargs='+', default=[0.14])
    parser.add_argument("--max_scale_crops", type=float, nargs='+', default=[1.0])
    parser.add_argument("--use_fp16", action='store_true')
    parser.add_argument("--arch", type=str, default='wtnet')
    parser.add_argument("--checkpoint", type=str, default=None, help='optional checkpoint to load')
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--crops_for_assign", type=int, nargs='+', default=[0,1])
    parser.add_argument("--use_sk_fusion", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--pooling_type", type=str, default='gem')
    args = parser.parse_args()

    dataset = IndexedS3R(
        args.data_path,
        args.split_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        is_train=True,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_train)

    # instantiate model
    model = WTNet(
        in_channels=3,
        input_size=[512,512],
        semantic_dim=128,
        num_classes=0,
        output_dim=128,
        hidden_mlp=0,
        nmb_prototypes=args.nmb_prototypes,
        use_sk_fusion=args.use_sk_fusion,
        pooling_type=args.pooling_type,
    )
    if args.checkpoint is not None:
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older pytorch versions that don't support weights_only
            ckpt = torch.load(args.checkpoint, map_location='cpu')
        # try to load state_dict if present
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        # adapt keys if model wrapped
        new_state = {}
        for k,v in state.items():
            if k.startswith('module.'):
                new_state[k[len('module.'):]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)

    model.to(args.device)
    model.eval()

    n_report = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels, indices) in enumerate(loader):
            # inputs: list of crop tensors each [B, C, H, W]
            # move to device
            inputs = [t.to(args.device) for t in inputs]
            # forward
            ret = model(inputs)
            # ret is (embedding, proto_out) or (embedding, proto_out, logits)
            if len(ret) == 3:
                embedding, proto_out, _ = ret
            else:
                embedding, proto_out = ret

            # proto_out: if MultiPrototypes, it's a list per head; if single Linear then tensor [N, K]
            # We handle only the simple case where prototypes is linear and outputs a tensor per sample
            if isinstance(proto_out, list):
                # assume first head
                out = proto_out[0]
            else:
                out = proto_out

            bs = inputs[0].size(0)
            # For each crop used for assignment, compute assignments
            assignments = {}
            for crop_id in args.crops_for_assign:
                # out currently is [total_samples, K] where total_samples = sum over crops
                start = bs * crop_id
                end = bs * (crop_id + 1)
                if end > out.size(0):
                    print(f"Crop id {crop_id} out of range for this batch (out.size={out.size(0)})")
                    continue
                logits = out[start:end].detach().cpu()
                q = local_sinkhorn(logits)
                # For each sample, pick argmax prototype
                proto_assign = torch.argmax(q, dim=1).tolist()
                for i, p in enumerate(proto_assign):
                    global_idx = int(indices[i].item())
                    assignments.setdefault(p, []).append((global_idx, float(q[i, p].item())))

            # Print mapping for this batch
            K = args.nmb_prototypes
            print(f"Batch {batch_idx}: prototype -> list of (sample_idx, score)")
            for k in range(K):
                lst = assignments.get(k, [])
                if len(lst) > 0:
                    print(f"proto {k}: {lst}")
            n_report += 1
            if n_report >= args.n_batches:
                break


if __name__ == '__main__':
    main()

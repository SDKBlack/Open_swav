import argparse
import os
import torch
import numpy as np
import pandas as pd
from src.utils import bool_flag, init_distributed_mode
from src.s3r_dataset import S3RDataset
from src.eval_openset import evaluate_openset, compute_distances, evaluate_metric, compute_stage2_up, metrics_stage_1
from src.wtnet import WTNet
import src.resnet50 as resnet_models
from logging import getLogger
import logging
from tqdm import tqdm

# Suppress detailed logging for this comparison script
logging.basicConfig(level=logging.ERROR)
logger = getLogger()

def get_metrics(model, train_loader, test_loader, unknown_loader, args, device):
    model.eval()
    
    # 1. Extract features for known train data
    train_features = []
    train_labels = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            if isinstance(inputs, list): inputs = inputs[0]
            inputs = inputs.to(device)
            ret = model(inputs)
            train_features.append(ret[1].cpu())
            train_labels.append(labels)
    train_X = torch.cat(train_features, dim=0)
    train_Y = torch.cat(train_labels, dim=0)
    
    # 2. Extract features for test data
    test_features = []
    test_labels = []
    
    # Known
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            ret = model(inputs)
            test_features.append(ret[1].cpu())
            test_labels.append(labels)
            
    # Unknown
    if unknown_loader:
        with torch.no_grad():
            for inputs, labels in unknown_loader:
                inputs = inputs.to(device)
                ret = model(inputs)
                test_features.append(ret[1].cpu())
                test_labels.append(labels)
                
    test_X = torch.cat(test_features, dim=0)
    test_Y = torch.cat(test_labels, dim=0)
    
    num_known = args.num_classes
    
    # 3. Compute Metrics (Euclidean)
    d_ct, theta = compute_distances(train_X, train_Y, test_X, num_known, metric='euclidean')
    tkr, tur, kp, fkr, mean_acc, label_hat = evaluate_metric(test_Y, d_ct, theta, num_known, "Euclidean")
    
    # 4. Compute Stage 2 UP (Euclidean)
    res = compute_stage2_up(test_X, test_Y, label_hat, theta, num_known)
    
    return {
        'TKR': tkr,
        'TUR': tur,
        'Mean Acc': mean_acc,
        'UP (DB)': res['up_db'],
        'UP (Sil)': res['up_sil']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--test_split_path", type=str, required=True)
    parser.add_argument("--unknown_split_path", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, nargs='+', default=[50, 100, 199])
    
    # Model params (fixed for this experiment)
    parser.add_argument("--arch", default="wtnet", type=str)
    parser.add_argument("--hidden_mlp", default=2048, type=int)
    parser.add_argument("--feat_dim", default=128, type=int)
    parser.add_argument("--nmb_prototypes", default=90, type=int)
    parser.add_argument("--num_classes", type=int, default=18)
    parser.add_argument("--use_freq_pos_enc", type=bool_flag, default=True)
    parser.add_argument("--use_aux_heads", type=bool_flag, default=True)
    
    # Data params
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--workers", default=4, type=int)
    
    # Dummy args for utils
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--gpu_to_work_on", default=0, type=int)
    
    args = parser.parse_args()
    
    # Init datasets once
    train_dataset = S3RDataset(args.data_path, args.split_path, args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops, is_train=False)
    test_dataset = S3RDataset(args.data_path, args.test_split_path, args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops, is_train=False)
    unknown_dataset = S3RDataset(args.data_path, args.unknown_split_path, args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    unknown_loader = torch.utils.data.DataLoader(unknown_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{'Epoch':<10} | {'TKR':<10} | {'TUR':<10} | {'Mean Acc':<10} | {'UP (DB)':<10} | {'UP (Sil)':<10}")
    print("-" * 75)
    
    for epoch in args.epochs:
        ckp_name = f"ckp-{epoch}.pth"
        ckp_path = os.path.join(args.checkpoints_dir, ckp_name)
        
        if not os.path.exists(ckp_path):
            print(f"Checkpoint {ckp_path} not found, skipping.")
            continue
            
        # Build model
        max_input_size = 224
        model = WTNet(
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
            num_classes=args.num_classes,
            use_freq_pos_enc=args.use_freq_pos_enc,
            use_aux_heads=args.use_aux_heads,
            input_size=[max_input_size, max_input_size],
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        
        # Evaluate
        metrics = get_metrics(model, train_loader, test_loader, unknown_loader, args, device)
        
        print(f"{epoch:<10} | {metrics['TKR']:.4f}     | {metrics['TUR']:.4f}     | {metrics['Mean Acc']:.4f}     | {metrics['UP (DB)']:.4f}     | {metrics['UP (Sil)']:.4f}")
        results.append(metrics)

if __name__ == "__main__":
    main()

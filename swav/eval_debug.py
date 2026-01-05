
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from logging import getLogger
import logging

# Setup logger
logger = getLogger()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Import from src
from src.wtnet import WTNet
from src.s3r_dataset import S3RDataset
from src.eval_openset import evaluate_openset

def main():
    # Define args based on train.log
    class Args:
        arch = "wtnet"
        batch_size = 128
        workers = 10
        data_path = "/root/autodl-tmp/S3R"
        split_path = "/root/autodl-tmp/S3R/experiment_groups/1-known_for_train"
        test_split_path = "/root/autodl-tmp/S3R/experiment_groups/1-known_for_test"
        unknown_split_path = "/root/autodl-tmp/S3R/experiment_groups/1-unknown"
        dump_path = "./test_enc_pro54"
        
        # Model params
        feat_dim = 128
        hidden_mlp = 2048
        nmb_prototypes = 54
        num_classes = 18 # Detected in log
        use_aux_heads = True
        pooling_type = "gem"
        
        # WTNet specific
        shared_stem_blocks = 2
        use_shared_stem = False
        use_freq_pos_enc = True
        use_sk_fusion = False
        use_specaugment = False
        spec_time_masks = 2
        spec_freq_masks = 2
        spec_max_time = 40
        spec_max_freq = 30
        
        # Misc
        rank = 0
        gpu_to_work_on = 0
        
        # Eval specific
        size_crops = [224]
        nmb_crops = [2] # Dummy for eval
        min_scale_crops = [0.8]
        max_scale_crops = [1.0]

    args = Args()
    
    # Load Model
    logger.info("Building model...")
    model = WTNet(
        normalize=True,
        output_dim=args.feat_dim,
        hidden_mlp=args.hidden_mlp,
        nmb_prototypes=args.nmb_prototypes,
        num_classes=args.num_classes,
        use_shared_stem=args.use_shared_stem,
        shared_stem_blocks=args.shared_stem_blocks,
        use_sk_fusion=args.use_sk_fusion,
        pooling_type=args.pooling_type,
        use_aux_heads=args.use_aux_heads,
        use_freq_pos_enc=args.use_freq_pos_enc,
        input_size=[224, 224],
    )
    model = model.cuda()
    
    # Load Checkpoint
    checkpoint_path = os.path.join(args.dump_path, "checkpoint.pth.tar")
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint['state_dict']
        # Remove module. prefix if present (DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        logger.error("Checkpoint not found!")
        return

    model.eval()

    # Build Loaders
    logger.info("Building loaders...")
    
    # Train loader for eval (prototypes/centers calculation)
    train_dataset_eval = S3RDataset(
        args.data_path,
        args.split_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        is_train=False
    )
    train_loader_eval = torch.utils.data.DataLoader(
        train_dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Test loader (Knowns)
    test_dataset = S3RDataset(
        args.data_path,
        args.test_split_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        is_train=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Unknown loader
    unknown_dataset = S3RDataset(
        args.data_path,
        args.unknown_split_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        is_train=False
    )
    unknown_loader = torch.utils.data.DataLoader(
        unknown_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Run Evaluation
    logger.info("Starting evaluation...")
    evaluate_openset(model, train_loader_eval, test_loader, unknown_loader, args)

if __name__ == "__main__":
    main()

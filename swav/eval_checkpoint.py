import argparse
import os
import torch
import torch.nn as nn
from src.utils import bool_flag, init_distributed_mode
from src.s3r_dataset import S3RDataset
from src.eval_openset import evaluate_openset
from src.wtnet import WTNet
import src.resnet50 as resnet_models
from logging import getLogger
import logging

logger = getLogger()

def main():
    parser = argparse.ArgumentParser(description="Evaluate SwAV/WTNet Checkpoint")
    
    # Data params
    parser.add_argument("--data_path", type=str, required=True, help="path to dataset repository")
    parser.add_argument("--split_path", type=str, required=True, help="path to train split file (for known stats)")
    parser.add_argument("--test_split_path", type=str, required=True, help="path to test split file")
    parser.add_argument("--unknown_split_path", type=str, required=True, help="path to unknown split file")
    
    # Model params
    parser.add_argument("--arch", default="wtnet", type=str, help="convnet architecture")
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=36, type=int, help="number of prototypes")
    parser.add_argument("--num_classes", type=int, default=0, help="number of classes")
    
    # WTNet specific
    parser.add_argument("--use_shared_stem", type=bool_flag, default=False)
    parser.add_argument("--shared_stem_blocks", type=int, default=2)
    parser.add_argument("--use_sk_fusion", type=bool_flag, default=False)
    parser.add_argument("--pooling_type", type=str, default="gem")
    parser.add_argument("--use_aux_heads", type=bool_flag, default=False)
    parser.add_argument("--use_freq_pos_enc", type=bool_flag, default=False)
    
    # Data loading params
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+")
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint file")
    
    # Misc
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--dump_path", type=str, default=".", help="path to save eval results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Init distributed (needed for some utils, even if single GPU)
    init_distributed_mode(args)

    # Create dump_path if it doesn't exist
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path, exist_ok=True)
    
    # Build model
    if args.arch == 'wtnet':
        max_input_size = max(args.size_crops) if args.size_crops else 224
        model = WTNet(
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
            num_classes=args.num_classes,
            use_shared_stem=args.use_shared_stem,
            shared_stem_blocks=args.shared_stem_blocks,
            use_sk_fusion=args.use_sk_fusion,
            pooling_type=args.pooling_type,
            use_aux_heads=args.use_aux_heads,
            use_freq_pos_enc=args.use_freq_pos_enc,
            input_size=[max_input_size, max_input_size],
        )
    else:
        model = resnet_models.__dict__[args.arch](
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
            num_classes=args.num_classes,
        )
        
    model = model.cuda()
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint_path):
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # Remove 'module.' prefix if present (since we are not wrapping in DDP here yet, or if we do we need to match)
        # If we run with torchrun, we might want DDP. But for simple eval, let's strip.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded model with msg: {msg}")
    else:
        logger.error(f"No checkpoint found at {args.checkpoint_path}")
        return

    # Build datasets
    # Train set (for known stats) - no augmentation
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
    
    # Test set (Known)
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
    
    # Unknown set
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
    
    logger.info("Starting evaluation...")
    evaluate_openset(model, train_loader_eval, test_loader, unknown_loader, args)

if __name__ == "__main__":
    main()

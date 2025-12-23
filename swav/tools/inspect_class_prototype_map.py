import argparse
import os
import torch
import sys
from collections import defaultdict, Counter
from torch.utils.data import DataLoader

# Add root to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.s3r_dataset import S3RDataset
from src.wtnet import WTNet

def get_args():
    parser = argparse.ArgumentParser(description="Inspect Class <-> Prototype Mapping")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--known_split", type=str, required=True, help="Path to known classes split file")
    parser.add_argument("--unknown_split", type=str, default=None, help="Path to unknown classes split file (optional)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    
    # Model params
    parser.add_argument("--arch", type=str, default="wtnet")
    parser.add_argument("--nmb_prototypes", type=int, default=45)
    parser.add_argument("--use_sk_fusion", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--pooling_type", type=str, default="gem")
    
    # Data params
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--size_crops", type=int, nargs='+', default=[224])
    parser.add_argument("--nmb_crops", type=int, nargs='+', default=[1]) # Use 1 crop for eval usually
    parser.add_argument("--min_scale_crops", type=float, nargs='+', default=[0.14])
    parser.add_argument("--max_scale_crops", type=float, nargs='+', default=[1.0])
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()

def load_model(args):
    # Override args to match checkpoint structure inferred from errors
    # Checkpoint has 384 dim features (3 branches * 128), so sk_fusion=False
    # Checkpoint has 384 dim into projection, so pooling didn't expand (gem/avg)
    print("Overriding model args to match checkpoint: use_sk_fusion=False, pooling_type='gem'")
    model = WTNet(
        in_channels=3,
        input_size=[512, 512],
        semantic_dim=128,
        num_classes=0,
        output_dim=128, 
        hidden_mlp=0,
        nmb_prototypes=args.nmb_prototypes,
        use_sk_fusion=False,
        pooling_type='gem',
    )

    
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # Remove module. prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(args.device)
    model.eval()
    return model

def process_loader(loader, model, device, class_to_proto, proto_to_class, prefix=""):
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            # inputs might be a list of crops, take the first one if so
            if isinstance(inputs, list):
                img = inputs[0]
            else:
                img = inputs
            
            img = img.to(device)
            labels = labels.to(device)
            
            # Forward
            ret = model(img)
            # ret is (embedding, proto_out) or (embedding, proto_out, logits)
            if len(ret) == 3:
                _, proto_out, _ = ret
            else:
                _, proto_out = ret
            
            # proto_out: [B, nmb_prototypes]
            # Get hard assignment
            preds = torch.argmax(proto_out, dim=1)
            
            for p, l in zip(preds, labels):
                p_item = p.item()
                l_item = l.item()
                
                class_to_proto[l_item][p_item] += 1
                proto_to_class[p_item][l_item] += 1

def print_stats(class_to_proto, proto_to_class):
    print("\n" + "="*60)
    print("CLASS -> PROTOTYPES MAPPING")
    print("="*60)
    # Sort by class ID
    all_classes = sorted(class_to_proto.keys())
    for cls in all_classes:
        counts = class_to_proto[cls]
        total = sum(counts.values())
        # Sort prototypes by count descending
        sorted_protos = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Format: Class 0 (N=100): p1(50, 50%), p2(30, 30%), ...
        proto_str = ", ".join([f"P{p}({c}, {c/total*100:.1f}%)" for p, c in sorted_protos])
        print(f"Class {cls:<3} (Total {total:<4}): {proto_str}")

    print("\n" + "="*60)
    print("PROTOTYPE -> CLASSES MAPPING")
    print("="*60)
    # Sort by prototype ID
    all_protos = sorted(proto_to_class.keys())
    for p in all_protos:
        counts = proto_to_class[p]
        total = sum(counts.values())
        # Sort classes by count descending
        sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Format: Proto 0 (N=100): c1(50, 50%), c2(30, 30%), ...
        class_str = ", ".join([f"C{c}({cnt}, {cnt/total*100:.1f}%)" for c, cnt in sorted_classes])
        print(f"Proto {p:<3} (Total {total:<4}): {class_str}")

def main():
    args = get_args()
    model = load_model(args)
    
    class_to_proto = defaultdict(Counter)
    proto_to_class = defaultdict(Counter)
    
    # 1. Known Data
    print(f"Processing Known Split: {args.known_split}")
    known_dataset = S3RDataset(
        args.data_path,
        args.known_split,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        is_train=False # No random augs for inspection usually
    )
    known_loader = DataLoader(known_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    process_loader(known_loader, model, args.device, class_to_proto, proto_to_class, prefix="Known")
    
    # 2. Unknown Data (Optional)
    if args.unknown_split:
        print(f"Processing Unknown Split: {args.unknown_split}")
        unknown_dataset = S3RDataset(
            args.data_path,
            args.unknown_split,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            is_train=False
        )
        unknown_loader = DataLoader(unknown_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        process_loader(unknown_loader, model, args.device, class_to_proto, proto_to_class, prefix="Unknown")
        
    print_stats(class_to_proto, proto_to_class)

if __name__ == "__main__":
    main()

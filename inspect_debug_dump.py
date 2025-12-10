import torch
import sys
import os

def inspect_dump(path):
    print(f"Loading {path}...")
    try:
        data = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("Keys in dump:", data.keys())
    
    if 'embedding' in data:
        emb = data['embedding']
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding has NaNs: {torch.isnan(emb).any()}")
        print(f"Embedding has Infs: {torch.isinf(emb).any()}")
        if torch.isnan(emb).any():
            print("  Count NaNs:", torch.isnan(emb).sum().item())
        print(f"Embedding min/max/mean: {emb.min()}/{emb.max()}/{emb.mean()}")

    if 'output' in data:
        out = data['output']
        print(f"Output shape: {out.shape}")
        print(f"Output has NaNs: {torch.isnan(out).any()}")
    
    if 'proto_w' in data:
        print(f"Proto weights shape: {data['proto_w'].shape}")

    if 'centers' in data:
        centers = data['centers']
        print(f"Centers shape: {centers.shape}")
        print(f"Centers has NaNs: {torch.isnan(centers).any()}")
        if torch.isnan(centers).any():
            print("  Count NaNs in centers:", torch.isnan(centers).sum().item())
        print(f"Centers min/max/mean: {centers.min()}/{centers.max()}/{centers.mean()}")

    if 'loss' in data:
        print(f"Loss: {data['loss']}")

    if 'model_state_dict' in data:
        print("Model state dict present.")
        # Check for NaNs in weights
        for k, v in data['model_state_dict'].items():
            if torch.isnan(v).any():
                print(f"NaNs found in model parameter: {k}")

    if 'optimizer_state_dict' in data:
        print("Optimizer state dict present.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_debug_dump.py <path_to_dump>")
        sys.exit(1)
    inspect_dump(sys.argv[1])

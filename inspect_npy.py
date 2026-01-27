import numpy as np
import os

file_path = '/root/autodl-tmp/S3R/Data/0/0-a.npy'
if os.path.exists(file_path):
    data = np.load(file_path)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}")
else:
    print(f"File not found: {file_path}")

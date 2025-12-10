import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class S3RDataset(Dataset):
    def __init__(
        self,
        data_path,
        split_file,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        is_train=True,
        random_erasing_prob=0.0,
    ):
        super(S3RDataset, self).__init__()
        self.data_path = data_path
        self.is_train = is_train
        self.random_erasing_prob = random_erasing_prob
        
        # Read split file
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                path = parts[0]
                label = int(parts[1])
                # Adjust path: ./Data/0/0 -> Data/0/0-a.npy
                # The split file has relative paths starting with ./Data
                # We need to construct the full path
                # Assuming data_path is /root/autodl-tmp/S3R
                # and path in file is ./Data/...
                
                if path.startswith('./'):
                    rel_path = path[2:]
                else:
                    rel_path = path
                
                full_path = os.path.join(data_path, rel_path + '-a.npy')
                self.samples.append((full_path, label))
        
        # Calculate number of classes
        labels = [s[1] for s in self.samples]
        self.num_classes = len(set(labels))
        # Assuming labels are 0-indexed and contiguous, or at least max(labels) + 1 covers them.
        # If labels are not contiguous, we might want max(labels) + 1.
        if len(labels) > 0:
             self.num_classes = max(labels) + 1
        else:
             self.num_classes = 0

        if self.is_train:
            assert len(size_crops) == len(nmb_crops)
            assert len(min_scale_crops) == len(nmb_crops)
            assert len(max_scale_crops) == len(nmb_crops)
            
            self.trans = []
            for i in range(len(size_crops)):
                randomresizedcrop = transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i]),
                )
                
                transform_list = [
                    randomresizedcrop,
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
                ]
                
                if self.random_erasing_prob > 0:
                    transform_list.append(transforms.RandomErasing(p=self.random_erasing_prob, scale=(0.01, 0.20), ratio=(0.3, 3.3), value=0))

                self.trans.extend([transforms.Compose(transform_list)] * nmb_crops[i])
        else:
            # For testing, just resize to the first crop size (usually 224) or keep original?
            # User said: "在测试的时候，只输入原始的数据"
            # But ResNet needs fixed size for batching if we use DataLoader with default collate.
            # If we use batch_size=1, we can keep original.
            # But usually we resize to 224x224 or 256x256.
            # The original is 543x512.
            # I'll resize to 224x224 for consistency with ResNet50 default.
            # Or maybe the user implies NO augmentation, but resizing is necessary for the model?
            # "只输入原始的数据" might mean no random cropping.
            # I'll use Resize to 224x224.
            self.test_trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path, label = self.samples[index]
        
        try:
            data = np.load(path)
        except Exception as e:
            # Handle missing files or errors
            print(f"Error loading {path}: {e}")
            # Return a dummy or skip? 
            # Better to fail loud or return zeros.
            data = np.zeros((543, 512), dtype=np.float32)

        # Data is (543, 512), float16 or float32.
        # Convert to float32
        data = data.astype(np.float32)
        
        # Convert to Tensor (1, H, W)
        data = torch.from_numpy(data).unsqueeze(0)
        
        # Replicate to 3 channels
        data = data.repeat(3, 1, 1)
        
        if self.is_train:
            multi_crops = list(map(lambda trans: trans(data), self.trans))
            return multi_crops, label
        else:
            out = self.test_trans(data)
            return out, label

    def __len__(self):
        return len(self.samples)

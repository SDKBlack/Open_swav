import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as TF

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
        use_specaugment=False,
        spec_freq_masks=0,
        spec_time_masks=0,
        spec_max_freq=0,
        spec_max_time=0,
    ):
        super(S3RDataset, self).__init__()
        self.data_path = data_path
        self.is_train = is_train
        # SpecAugment params
        self.use_specaugment = use_specaugment
        self.spec_freq_masks = spec_freq_masks
        self.spec_time_masks = spec_time_masks
        self.spec_max_freq = spec_max_freq
        self.spec_max_time = spec_max_time
        
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
                # randomresizedcrop = transforms.RandomResizedCrop(
                #     size_crops[i],
                #     scale=(min_scale_crops[i], max_scale_crops[i]),
                # )
                # Disable random crop, use Resize instead
                resize_transform = transforms.Resize((size_crops[i], size_crops[i]))
                
                transform_list = [
                    resize_transform,
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
                ]

                # Optionally apply RandomErasing or SpecAugment later in tensor domain
                # We will append a callable that applies SpecAugment if the dataset was
                # constructed with specaugment parameters (attached as attributes).
                def final_transform(tensor):
                    # tensor: C x H x W (we expect spectrogram-like spatial dims)
                    out = tensor
                    # If specaugment is requested, call dataset-level specaugment
                    if getattr(self, 'use_specaugment', False):
                        out = self._apply_specaugment(out)
                    return out

                transform_list.append(transforms.Lambda(lambda t: final_transform(t)))

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

    # --- SpecAugment helper ---
    def _apply_specaugment(self, tensor):
        """
        Apply frequency and time masking on a tensor of shape C x H x W.
        We interpret H as frequency bins and W as time frames.
        """
        # operate on a copy
        out = tensor.clone()
        # Work on single-channel representation for masking (collapse channels if >1)
        if out.size(0) > 1:
            spec = out.mean(dim=0, keepdim=True)[0]  # H x W
        else:
            spec = out[0]

        H, W = spec.shape
        # Frequency masks
        n_freq = getattr(self, 'spec_freq_masks', 0)
        max_freq = getattr(self, 'spec_max_freq', max(1, H // 4))
        for _ in range(n_freq):
            f = random.randint(0, min(max_freq, H))
            f0 = random.randint(0, max(0, H - f))
            spec[f0:f0+f, :] = 0

        # Time masks
        n_time = getattr(self, 'spec_time_masks', 0)
        max_time = getattr(self, 'spec_max_time', max(1, W // 4))
        for _ in range(n_time):
            t = random.randint(0, min(max_time, W))
            t0 = random.randint(0, max(0, W - t))
            spec[:, t0:t0+t] = 0

        # Broadcast back to channels
        if out.size(0) > 1:
            out = out.clone()
            for c in range(out.size(0)):
                out[c] = spec
        else:
            out[0] = spec

        return out

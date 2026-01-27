
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from thop import profile, clever_format

# Add paths
sys.path.append('/root/autodl-tmp/data/swav-main/swav-main')
sys.path.append('/root/autodl-tmp/WTConv')

# Import WT model
try:
    from src.wt_models import wt_tri_branch_net
except ImportError:
    # If running from root, try adjusting path
    sys.path.append(os.getcwd())
    from src.wt_models import wt_tri_branch_net

# --- Helper Functions from S3R/train.py ---
def position_coding(x):
    num_token, num_dims = x.size(-2), x.size(-1)
    p = torch.zeros((1, num_token, num_dims))
    t = torch.arange(num_token, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(1e4, torch.arange(0, num_dims, 2, dtype=torch.float32) / num_dims)
    p[:, :, 0::2] = torch.sin(t)
    p[:, :, 1::2] = torch.cos(t)
    return p

# --- NET Class from S3R/train.py ---
class NET(nn.Module):
    def __init__(self, in_channels, input_size, semantic_dim, num_class, device):
        super(NET, self).__init__()
        self.input_size = input_size                     # [T, W]
        self.semantic_dim = semantic_dim
        self.num_class = num_class
        self.device = device
        self.in_channels = in_channels
        self.SA_1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_1 = nn.TransformerEncoder(self.SA_1, num_layers=3)
        self.SA_2 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_2 = nn.TransformerEncoder(self.SA_2, num_layers=3)

        self.encoding_to_sa1 = nn.Sequential(
            nn.Linear(self.input_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.encoding_to_sa2 = nn.Sequential(
            nn.Linear(self.input_size[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.sa1_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[0] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        self.sa2_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[1] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        self.total_semantic = nn.Sequential(
            nn.Linear(self.semantic_dim * 3, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU(),
            nn.Linear(self.semantic_dim, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )
        self.encoder_d1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_d3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_d5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_to_semantic = nn.Sequential(
            nn.Linear(128*3, self.semantic_dim*2),
            nn.BatchNorm1d(self.semantic_dim*2),
            nn.ReLU(),
            nn.Linear(self.semantic_dim*2, self.semantic_dim),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )
        self.semantic_to_classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, self.num_class)
        )

    def forward(self, x, y, z):
        encoder1 = self.encoder_d1(x)
        encoder2 = self.encoder_d3(x)
        encoder3 = self.encoder_d5(x)
        encoder_output = torch.cat([encoder1, encoder2, encoder3], dim=1)
        x = F.adaptive_avg_pool2d(encoder_output, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.encoder_to_semantic(x)
        expand_x = F.adaptive_avg_pool2d(encoder_output, (1, 1))
        expand_x = expand_x.view(expand_x.shape[0], -1)
        y = self.encoding_to_sa1(y)                 # [B, T, 64]
        z = self.encoding_to_sa2(z)                 # [B, W, 64]
        y = y + position_coding(y).to(self.device)
        z = z + position_coding(z).to(self.device)
        y = self.SA_1(y)                            # [B, T, 64]
        z = self.SA_2(z)                            # [B, W, 64]
        y = y.view(y.shape[0], -1)                  # [B, T*64]
        z = z.view(z.shape[0], -1)                  # [B, W*64]
        y = self.sa1_to_semantic(y)                 # [B, T*64] -> [B, semantic dim]
        z = self.sa2_to_semantic(z)                 # [B, W*64] -> [B, semantic dim]
        semantic = torch.cat([x, y], dim=1)
        semantic = torch.cat([semantic, z], dim=1)
        semantic = self.total_semantic(semantic)  # [B, 128*3] -> [B, 128]
        predict = self.semantic_to_classifier(semantic)

        return predict, semantic, x, y, z, expand_x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config for NET (from train.py)
    # NET(in_channels=1, input_size=[int(512 * len_time), 512], semantic_dim=semantic_dim, num_class=num_known, device=device)
    # Assuming len_time=1, size=512, semantic_dim=128, num_known=18
    net_model = NET(in_channels=1, input_size=[512, 512], semantic_dim=128, num_class=18, device=device).to(device)
    
    # Config for WTTriBranchNet
    # WTTriBranchNet(normalize=False, output_dim=0, hidden_mlp=0, nmb_prototypes=0, eval_mode=False, num_classes=0)
    # To match NET's classification purpose, we set num_classes=18.
    # NET has a complex head structure (Transformers etc). WTTriBranchNet has a simpler head.
    # We will compare the "backbone" part mostly, but let's instantiate full models.
    wt_model = wt_tri_branch_net(num_classes=18).to(device)
    
    print("="*50)
    print("Model Comparison")
    print("="*50)
    
    # 1. Parameter Count
    net_params = count_parameters(net_model)
    wt_params = count_parameters(wt_model)
    
    print(f"NET Parameters: {net_params:,}")
    print(f"WTConv Model Parameters: {wt_params:,}")
    print(f"Difference: {wt_params - net_params:,}")
    
    # 2. MACs / FLOPs
    input_size = 512
    batch_size = 1
    
    # Dummy inputs
    x_dummy = torch.randn(batch_size, 1, input_size, input_size).to(device)
    y_dummy = torch.randn(batch_size, input_size, input_size).to(device)
    z_dummy = torch.randn(batch_size, input_size, input_size).to(device)
    
    x_wt_dummy = torch.randn(batch_size, 1, input_size, input_size).to(device)
    
    print("-" * 30)
    print("Calculating MACs...")
    
    try:
        # NET MACs
        # thop.profile expects model and inputs=(args,)
        net_macs, net_params_thop = profile(net_model, inputs=(x_dummy, y_dummy, z_dummy), verbose=False)
        net_macs_fmt, net_params_fmt = clever_format([net_macs, net_params_thop], "%.3f")
        print(f"NET MACs: {net_macs_fmt}")
    except Exception as e:
        print(f"NET MACs calculation failed: {e}")

    try:
        # WT Model MACs
        wt_macs, wt_params_thop = profile(wt_model, inputs=(x_wt_dummy,), verbose=False)
        wt_macs_fmt, wt_params_fmt = clever_format([wt_macs, wt_params_thop], "%.3f")
        print(f"WTConv Model MACs: {wt_macs_fmt}")
    except Exception as e:
        print(f"WTConv Model MACs calculation failed: {e}")

if __name__ == "__main__":
    compare_models()

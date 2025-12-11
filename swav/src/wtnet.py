import torch
import torch.nn as nn
import torch.nn.functional as F
from .wtconv.wtconv2d import WTConv2d


class MHSA2D(nn.Module):
    """A simple multi-head self-attention for 2D feature maps.
    Input: (B, C, H, W)
    Output: (B, C, H, W) with residual connection.
    """
    def __init__(self, in_channels, num_heads=8, dropout=0.0):
        super(MHSA2D, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # use 1x1 convs to compute qkv and projection
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # residual scaling
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)  # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        # qkv[:, i] -> (B, heads, head_dim, N)
        q = qkv[:, 0]  # (B, heads, head_dim, N)
        k = qkv[:, 1]
        v = qkv[:, 2]
        # transpose to (B, heads, N, head_dim)
        q = q.permute(0, 1, 3, 2).contiguous()
        k = k.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2).contiguous()

        # compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().reshape(B, C, H, W)
        out = self.proj(out)
        return x + self.gamma * out

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=20.0):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


class WTNet(nn.Module):
    def __init__(self, in_channels=3, input_size=[512, 512], semantic_dim=128, num_classes=0, 
                 output_dim=0, hidden_mlp=0, nmb_prototypes=0, eval_mode=False, normalize=False,
                 use_attention=True, attn_heads=8, use_shared_stem=False, shared_stem_blocks=2):
        super(WTNet, self).__init__()
        
        # SwAV specific params
        self.eval_mode = eval_mode
        self.l2norm = normalize
        self.num_classes = num_classes
        
        # Network params
        self.in_channels = in_channels
        self.semantic_dim = semantic_dim
        self.use_attention = use_attention
        self.attn_heads = attn_heads
        self.use_shared_stem = use_shared_stem
        self.shared_stem_blocks = shared_stem_blocks
        
        # If requested, create a shared stem to reduce repeated computation across branches.
        # The stem will execute the first `shared_stem_blocks` blocks once and then each
        # branch continues from that shared representation.
        if self.use_shared_stem and self.shared_stem_blocks > 0:
            self.shared_stem = self._make_shared_stem(self.shared_stem_blocks)
            # branches start after shared_stem_blocks
            start_block = self.shared_stem_blocks
            self.encoder_d1 = self._make_branch(kernel_size=1, start_block=start_block)
            self.encoder_d3 = self._make_branch(kernel_size=3, start_block=start_block)
            self.encoder_d5 = self._make_branch(kernel_size=5, start_block=start_block)
        else:
            self.shared_stem = None
            self.encoder_d1 = self._make_branch(kernel_size=1, start_block=0)
            self.encoder_d3 = self._make_branch(kernel_size=3, start_block=0)
            self.encoder_d5 = self._make_branch(kernel_size=5, start_block=0)
        
        # Encoder to Semantic (Projection)
        # 128 channels * 3 branches = 384
        self.feature_dim = 128 * 3
        # Attention module (applied on concatenated feature map before pooling)
        if self.use_attention:
            self.attn = MHSA2D(self.feature_dim, num_heads=self.attn_heads)
        else:
            self.attn = None
        
        self.encoder_to_semantic = nn.Sequential(
            nn.Linear(self.feature_dim, self.semantic_dim*2),
            nn.BatchNorm1d(self.semantic_dim*2),
            nn.ReLU(),
            nn.Linear(self.semantic_dim*2, self.semantic_dim),
            nn.BatchNorm1d(self.semantic_dim),
            # nn.ReLU()
        )
        
        # SwAV Projection Head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(self.semantic_dim, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(self.semantic_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # Prototypes
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
            
        # Classifier (optional)
        if num_classes > 0:
            self.classifier = CosineClassifier(self.semantic_dim, num_classes)
        else:
            self.classifier = None

    

    def _make_branch(self, kernel_size, start_block=0):
        """Construct a branch starting from block index `start_block`.
        Block channel progression (pairs):
        block 0: in_channels -> 4
        block 1: 4 -> 8
        block 2: 8 -> 16
        block 3: 16 -> 32
        block 4: 32 -> 64
        block 5: 64 -> 128
        """
        layers = []
        out_list = [4, 8, 16, 32, 64, 128]
        # Determine starting input channel for the first block we will create
        if start_block == 0:
            prev_out = self.in_channels
        else:
            prev_out = out_list[start_block - 1]

        for i in range(start_block, len(out_list)):
            out_c = out_list[i]
            layers.extend(self._make_block(prev_out, out_c, kernel_size, pool=(i != len(out_list)-1)))
            prev_out = out_c

        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _make_shared_stem(self, blocks):
        """Create a shared stem consisting of the first `blocks` blocks (from block 0 upward).
        This is applied once to the input and the resulting tensor is fed to each branch.
        """
        assert blocks >= 1 and blocks <= 6, "shared_stem_blocks must be between 1 and 6"
        layers = []
        out_list = [4, 8, 16, 32, 64, 128]
        prev_out = self.in_channels
        for i in range(blocks):
            out_c = out_list[i]
            layers.extend(self._make_block(prev_out, out_c, 3, pool=(i != blocks-1)))
            prev_out = out_c

        return nn.Sequential(*layers)

    def _make_block(self, in_c, out_c, k, pool=True):
        layers = []
        # WTConv2d(in, in) -> Conv2d(in, out, 1)
        layers.append(WTConv2d(in_c, in_c, kernel_size=k, wt_levels=2))
        if in_c != out_c:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=1))
            
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU())
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers

    def forward_backbone(self, x):
        # If we have a shared stem, run it once and feed the result to each branch.
        if getattr(self, 'shared_stem', None) is not None:
            shared = self.shared_stem(x)
            e1 = self.encoder_d1(shared)
            e3 = self.encoder_d3(shared)
            e5 = self.encoder_d5(shared)
        else:
            e1 = self.encoder_d1(x)
            e3 = self.encoder_d3(x)
            e5 = self.encoder_d5(x)

        # Concatenate
        out = torch.cat([e1, e3, e5], dim=1)  # [B, 128*3, H, W]

        # optional attention on spatial features
        if self.attn is not None:
            out = self.attn(out)

        # Global Pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        # Encoder to Semantic
        out = self.encoder_to_semantic(out)

        return out

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        
        logits = None
        if self.classifier is not None:
            logits = self.classifier(output)
            
        embedding, proto_out = self.forward_head(output)
        
        # Cache backbone features for Boundary Loss
        self._last_backbone = output
        
        if logits is not None:
            return embedding, proto_out, logits
        return embedding, proto_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from .wtconv.wtconv2d import WTConv2d


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w

        return out

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

class ArcFaceClassifier(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super(ArcFaceClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input, dim=1)
        # normalize weights
        W = F.normalize(self.weight, dim=1)
        # dot product
        cosine = F.linear(x, W)
        
        if label is None:
            return cosine * self.s
            
        # ArcFace
        # cos(theta + m)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = cosine * (1.0 - one_hot) + target_logits * one_hot
        output *= self.s
        return output


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
            # self.attn = MHSA2D(self.feature_dim, num_heads=self.attn_heads)
            self.attn = CoordinateAttention(self.feature_dim, self.feature_dim)
        else:
            self.attn = None
        
        # GeM Pooling
        self.gem = GeM()
        
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
            # self.classifier = CosineClassifier(self.semantic_dim, num_classes)
            self.classifier = ArcFaceClassifier(self.semantic_dim, num_classes)
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
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.gem(out)
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

    def forward(self, inputs, labels=None):
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
            # If labels are provided, we need to expand them to match the output size
            # output size is sum(n_crops * bs)
            # labels size is bs
            # We assume labels correspond to the first bs samples (and repeated for other crops)
            # Actually, inputs is a list of crops. Each crop has batch_size samples.
            # The labels are for the images.
            # So if we have N crops, we have N * bs samples.
            # The labels should be repeated N times.
            
            if labels is not None:
                bs = labels.size(0)
                total_bs = output.size(0)
                if total_bs > bs:
                    # Repeat labels
                    n_repeats = total_bs // bs
                    # Check if exact multiple
                    if total_bs % bs == 0:
                        labels_expanded = labels.repeat(n_repeats)
                        logits = self.classifier(output, labels_expanded)
                    else:
                        # Fallback or error? 
                        # If batch sizes differ across crops (unlikely in SwAV), we might have issues.
                        # For now, assume standard SwAV setup.
                        # If not matching, pass None to get cosine similarity without margin
                        logits = self.classifier(output)
                else:
                    logits = self.classifier(output, labels)
            else:
                logits = self.classifier(output)
            
        embedding, proto_out = self.forward_head(output)
        
        # Cache backbone features for Boundary Loss
        self._last_backbone = output
        
        if logits is not None:
            return embedding, proto_out, logits
        return embedding, proto_out

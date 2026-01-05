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
    def __init__(self, in_features, num_classes, s=20.0, m=0.5):
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


class MPNCOV(nn.Module):
    def __init__(self, iter_num=3):
        super(MPNCOV, self).__init__()
        self.iter_num = iter_num

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        x = x.view(B, C, -1) # B, C, N
        N = H * W
        x = x - x.mean(dim=2, keepdim=True)
        sigma = torch.bmm(x, x.transpose(1, 2)) / (N - 1) # B, C, C
        
        # Matrix Square Root via Newton-Schulz
        # Pre-normalization for stability
        trace = sigma.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) # B, 1, 1
        sigma = sigma / (trace + 1e-6)
        
        Y = sigma
        I = torch.eye(C, device=sigma.device).unsqueeze(0).expand(B, C, C)
        Z = I
        
        for i in range(self.iter_num):
            T = 0.5 * (3.0 * I - torch.bmm(Z, Y))
            Y = torch.bmm(Y, T)
            Z = torch.bmm(T, Z)
            
        sigma_sqrt = Y * torch.sqrt(trace + 1e-6)
        
        # Upper triangular part
        triu_indices = torch.triu_indices(C, C, device=sigma.device)
        out = sigma_sqrt[:, triu_indices[0], triu_indices[1]]
        
        return out # B, C*(C+1)/2

class SelectiveKernelFusion(nn.Module):
    def __init__(self, channels, branches=3, reduction=16):
        super(SelectiveKernelFusion, self).__init__()
        self.channels = channels
        d = max(channels // reduction, 32)
        self.fc1 = nn.Linear(channels, d)
        self.fc2 = nn.Linear(d, channels * branches)
        self.softmax = nn.Softmax(dim=1)
        self.branches = branches

    def forward(self, x_list):
        # x_list: list of [B, C, H, W]
        batch_size = x_list[0].shape[0]
        
        # Fuse: Element-wise Sum
        U = sum(x_list) # [B, C, H, W]
        
        # Global Avg Pool
        s = U.mean([-2, -1]) # [B, C]
        
        # Compact feature
        z = self.fc1(s) # [B, d]
        z = F.relu(z)
        
        # Attention weights
        weights = self.fc2(z) # [B, C*branches]
        weights = weights.view(batch_size, self.branches, self.channels)
        weights = self.softmax(weights) # [B, branches, C]
        
        # Weighted Sum
        V = 0
        for i, x in enumerate(x_list):
            w = weights[:, i, :].unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
            V += w * x
            
        return V

class WTNet(nn.Module):
    def __init__(self, in_channels=3, input_size=[512, 512], semantic_dim=512, num_classes=0, 
                 output_dim=0, hidden_mlp=0, nmb_prototypes=0, eval_mode=False, normalize=False,
                 use_shared_stem=False, shared_stem_blocks=2,
                 use_sk_fusion=False, pooling_type='gem', use_aux_heads=False,
                 use_freq_pos_enc=False):
        super(WTNet, self).__init__()
        
        # SwAV specific params
        self.eval_mode = eval_mode
        self.l2norm = normalize
        self.num_classes = num_classes
        self.use_aux_heads = use_aux_heads
        self.use_freq_pos_enc = use_freq_pos_enc
        self.input_size = input_size
        
        # Network params
        self.in_channels = in_channels
        self.semantic_dim = semantic_dim
        self.use_shared_stem = use_shared_stem
        self.shared_stem_blocks = shared_stem_blocks
        self.use_sk_fusion = use_sk_fusion
        self.pooling_type = pooling_type
        
        # Frequency Positional Encoding
        if self.use_freq_pos_enc:
            # Assuming input_size[0] is the max frequency dimension
            max_freq = input_size[0]
            self.pos_branch = nn.Sequential(
                nn.Linear(max_freq, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.pos_dim = 64
        else:
            self.pos_branch = None
            self.pos_dim = 0
        
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
        
        # Feature Fusion
        if self.use_sk_fusion:
            self.sk_fusion = SelectiveKernelFusion(128, branches=3)
            self.feature_dim = 128
        else:
            self.sk_fusion = None
            # 128 channels * 3 branches = 384
            self.feature_dim = 128 * 3

        # Pooling
        if self.pooling_type == 'mpn':
            self.pool = MPNCOV()
            # MPN-COV output dimension is C*(C+1)/2
            self.feature_dim = self.feature_dim * (self.feature_dim + 1) // 2
        elif self.pooling_type == 'gem':
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.encoder_to_semantic = nn.Sequential(
            nn.Linear(self.feature_dim, self.semantic_dim*2),
            nn.BatchNorm1d(self.semantic_dim*2),
            nn.ReLU(),
            nn.Linear(self.semantic_dim*2, self.semantic_dim),
            nn.BatchNorm1d(self.semantic_dim),
            # nn.ReLU()
        )
        
        # Auxiliary Heads
        if self.use_aux_heads and self.num_classes > 0:
            # Each branch outputs 128 channels
            # We need a pooling layer and a linear classifier for each
            # We'll reuse the same pooling type as main branch for consistency, 
            # but we need separate instances if they have learnable params (like GeM)
            # MPNCOV might be too heavy for aux heads? Let's stick to main pooling type.
            
            def make_aux_head(in_dim):
                pool = None
                feat_dim = in_dim
                if self.pooling_type == 'mpn':
                    pool = MPNCOV()
                    feat_dim = in_dim * (in_dim + 1) // 2
                elif self.pooling_type == 'gem':
                    pool = GeM()
                else:
                    pool = nn.AdaptiveAvgPool2d((1, 1))
                # If this pool is GeM and we're creating an auxiliary head, freeze its learned p
                # to avoid unstable gradients coming from auxiliary classifier losses.
                if isinstance(pool, GeM):
                    try:
                        pool.p.requires_grad = False
                    except Exception:
                        pass

                return nn.Sequential(
                    pool,
                    nn.Flatten(),
                    nn.Linear(feat_dim, self.semantic_dim),
                    nn.BatchNorm1d(self.semantic_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.semantic_dim, self.num_classes)
                )

            self.aux_head1 = make_aux_head(128)
            self.aux_head3 = make_aux_head(128)
            self.aux_head5 = make_aux_head(128)
        else:
            self.aux_head1 = None
            self.aux_head3 = None
            self.aux_head5 = None
        
        # SwAV Projection Head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(self.semantic_dim + self.pos_dim, output_dim)
        else:
            self.projection_head = nn.Linear(self.semantic_dim + self.pos_dim, output_dim)
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.semantic_dim, hidden_mlp),
            #     nn.BatchNorm1d(hidden_mlp),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(hidden_mlp, output_dim),
            # )

        # Prototypes
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
            
        # Classifier (optional)
        if num_classes > 0:
            # self.classifier = CosineClassifier(self.semantic_dim, num_classes)
            self.classifier = ArcFaceClassifier(self.semantic_dim + self.pos_dim, num_classes)
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
            layers.append(nn.MaxPool2d(kernel_size=1, stride=2))
        return layers

    def forward_backbone(self, x):
        # Position Branch
        pos_feat = None
        if self.use_freq_pos_enc and self.pos_branch is not None:
            # x: B, C, H, W
            # Frequency profile: mean over Channel and Time (W)
            # We want to capture energy distribution along Frequency (H)
            freq_profile = x.mean(dim=[1, 3]) # B, H
            
            # Interpolate to match max_freq (input_size[0])
            target_H = self.input_size[0]
            if freq_profile.shape[1] != target_H:
                freq_profile = F.interpolate(freq_profile.unsqueeze(1), size=target_H, mode='linear', align_corners=False).squeeze(1)
            
            pos_feat = self.pos_branch(freq_profile) # B, 64

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

        # Auxiliary Heads Forward
        aux_logits = {}
        if self.use_aux_heads and self.aux_head1 is not None:
            aux_logits['1'] = self.aux_head1(e1)
            aux_logits['3'] = self.aux_head3(e3)
            aux_logits['5'] = self.aux_head5(e5)

        # Concatenate or Fuse
        if self.sk_fusion is not None:
            out = self.sk_fusion([e1, e3, e5]) # [B, 128, H, W]
        else:
            out = torch.cat([e1, e3, e5], dim=1)  # [B, 128*3, H, W]

        # Global Pooling
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.pool(out)
        out = out.view(out.size(0), -1)

        # Encoder to Semantic
        out = self.encoder_to_semantic(out)

        if pos_feat is not None:
            out = torch.cat([out, pos_feat], dim=1)

        return out, aux_logits

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x, None

    def forward(self, inputs, labels=None):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        aux_logits_list = []
        for end_idx in idx_crops:
            _out, _aux = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            
            # Collect aux logits if available
            if _aux:
                aux_logits_list.append(_aux)
                
            start_idx = end_idx
        
        # Concatenate aux logits across crops
        final_aux_logits = {}
        if aux_logits_list:
            for k in aux_logits_list[0].keys():
                final_aux_logits[k] = torch.cat([d[k] for d in aux_logits_list], dim=0)
        
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
            if self.use_aux_heads:
                return embedding, proto_out, logits, final_aux_logits
            return embedding, proto_out, logits
        
        if self.use_aux_heads:
            return embedding, proto_out, final_aux_logits
        return embedding, proto_out

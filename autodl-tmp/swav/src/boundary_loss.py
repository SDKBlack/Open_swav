import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=None, pos_thresh=0.5, neg_thresh=1.0, proto_thresh=1.0):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.proto_thresh = proto_thresh

        # Learnable prototypes for known classes.
        # We support lazy initialization: if feat_dim is provided we initialize now,
        # otherwise we'll create prototypes on first forward() using the actual
        # feature dimensionality from the backbone. This avoids shape mismatch
        # when the backbone feature dim differs from user-provided config.
        if feat_dim is not None:
            self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
            nn.init.xavier_uniform_(self.prototypes)
        else:
            # Will be created on demand in forward()
            self.prototypes = None

    def forward(self, features, labels):
        """
        Args:
            features: [N, D] backbone features
            labels: [N] class labels
        """
        # Normalize features and prototypes
        features = F.normalize(features, dim=1)

        # Lazy initialization: if prototypes were not constructed (or have
        # mismatched feature dimension), create/recreate them to match the
        # incoming feature dimensionality.
        if self.prototypes is None or self.prototypes.shape[1] != features.size(1):
            feat_dim_in = features.size(1)
            p = nn.Parameter(torch.randn(self.num_classes, feat_dim_in, device=features.device, dtype=features.dtype))
            nn.init.xavier_uniform_(p)
            # assign to module so it's registered as a parameter
            self.prototypes = p

        # make sure prototypes are on the same device and dtype as features
        proto = self.prototypes
        if proto.device != features.device or proto.dtype != features.dtype:
            proto = proto.to(device=features.device, dtype=features.dtype)
            # reassign to ensure the module has a parameter on the correct device
            self.prototypes = nn.Parameter(proto)
        prototypes = F.normalize(self.prototypes, dim=1)

        # 1. Calculate Euclidean Distances
        # dist(u, v)^2 = 2 - 2(u.v)
        
        # Sample-to-Prototype Similarity
        sim_sp = torch.mm(features, prototypes.t()) # [N, K]
        dist_sp = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_sp, min=1e-6)) # [N, K]

        # Prototype-to-Prototype Similarity
        sim_pp = torch.mm(prototypes, prototypes.t()) # [K, K]
        dist_pp = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_pp, min=1e-6)) # [K, K]

        # 2. Masks
        num_prototypes = self.num_classes
        
        # Filter valid labels
        valid_mask = labels < num_prototypes
        if not valid_mask.all():
            features = features[valid_mask]
            labels = labels[valid_mask]
            dist_sp = dist_sp[valid_mask]
            if len(labels) == 0:
                return torch.tensor(0.0, device=features.device)

        pos_mask = F.one_hot(labels, num_classes=num_prototypes).bool()
        neg_mask = ~pos_mask

        # 3. Calculate Losses
        
        # Term 1: Distance to Positive Class > pos_thresh
        pos_dists = dist_sp[pos_mask]
        loss_pos = torch.mean(F.relu(pos_dists - self.pos_thresh))

        # Term 2: Distance to Negative Class < neg_thresh
        neg_dists = dist_sp[neg_mask]
        loss_neg = torch.mean(F.relu(self.neg_thresh - neg_dists))

        # Term 3: Distance between Prototypes < proto_thresh
        eye_mask = torch.eye(num_prototypes, device=prototypes.device, dtype=torch.bool)
        inter_proto_dists = dist_pp[~eye_mask]
        loss_proto = torch.mean(F.relu(self.proto_thresh - inter_proto_dists))

        return loss_pos + loss_neg + loss_proto

    def forward_virtual(self, features):
        """
        Compute loss for virtual unknown samples (e.g. from Mixup).
        These samples should be far from all known class prototypes.
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Lazy initialization (same as forward)
        if self.prototypes is None or self.prototypes.shape[1] != features.size(1):
            feat_dim_in = features.size(1)
            p = nn.Parameter(torch.randn(self.num_classes, feat_dim_in, device=features.device, dtype=features.dtype))
            nn.init.xavier_uniform_(p)
            self.prototypes = p

        # make sure prototypes are on the same device and dtype as features
        proto = self.prototypes
        if proto.device != features.device or proto.dtype != features.dtype:
            proto = proto.to(device=features.device, dtype=features.dtype)
            self.prototypes = nn.Parameter(proto)
        prototypes = F.normalize(self.prototypes, dim=1)

        # Sample-to-Prototype Similarity
        sim_sp = torch.mm(features, prototypes.t()) # [N, K]
        dist_sp = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_sp, min=1e-6)) # [N, K]

        # All prototypes are negative for virtual unknowns
        # We want dist_sp > neg_thresh
        # Loss = mean(relu(neg_thresh - dist_sp))
        loss_virtual = torch.mean(F.relu(self.neg_thresh - dist_sp))

        return loss_virtual

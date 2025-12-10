import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, pos_thresh=0.5, neg_thresh=1.0, proto_thresh=1.0):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.proto_thresh = proto_thresh
        
        # Learnable prototypes for known classes
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features, labels):
        """
        Args:
            features: [N, D] backbone features
            labels: [N] class labels
        """
        # Normalize features and prototypes
        features = F.normalize(features, dim=1)

        # make sure prototypes are on the same device and dtype as features
        proto = self.prototypes
        if proto.device != features.device or proto.dtype != features.dtype:
            proto = proto.to(device=features.device, dtype=features.dtype)
        prototypes = F.normalize(proto, dim=1)

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

"""Loss functions for SuperGlue Light model"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperGlueLightLoss(nn.Module):
    def __init__(self, pos_margin: float = 1.0, neg_margin: float = 0.2, geo_weight: float = 0.5):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.geo_weight = geo_weight

    def contrastive_loss(self, desc1: torch.Tensor, desc2: torch.Tensor, matches: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between descriptor pairs
        Args:
            desc1: Descriptors from first image (B, N, D)
            desc2: Descriptors from second image (B, N, D)
            matches: Ground truth matches (B, N, N) with 1 for matches, 0 otherwise
        Returns:
            Contrastive loss value
        """
        # Normalize descriptors
        desc1 = F.normalize(desc1, p=2, dim=2)
        desc2 = F.normalize(desc2, p=2, dim=2)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(desc1, desc2.transpose(1, 2))  # (B, N, N)
        
        # Positive pairs loss
        pos_loss = matches * torch.clamp(self.pos_margin - sim_matrix, min=0)
        
        # Negative pairs loss
        neg_loss = (1 - matches) * torch.clamp(sim_matrix - self.neg_margin, min=0)
        
        # Combine losses
        loss = (pos_loss + neg_loss).mean()
        
        return loss

    def geometric_verification_loss(self, kpts1: torch.Tensor, kpts2: torch.Tensor, 
                                  matches: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Compute geometric verification loss using homography
        Args:
            kpts1: Keypoints from first image (B, N, 2)
            kpts2: Keypoints from second image (B, N, 2)
            matches: Predicted matches (B, N, N)
            H: Ground truth homography matrices (B, 3, 3)
        Returns:
            Geometric verification loss value
        """
        batch_size = kpts1.shape[0]
        num_points = kpts1.shape[1]
        
        # Convert keypoints to homogeneous coordinates
        ones = torch.ones(batch_size, num_points, 1, device=kpts1.device)
        kpts1_h = torch.cat([kpts1, ones], dim=2)  # (B, N, 3)
        kpts2_h = torch.cat([kpts2, ones], dim=2)  # (B, N, 3)
        
        # Transform keypoints using homography
        kpts1_transformed = torch.matmul(H, kpts1_h.transpose(1, 2)).transpose(1, 2)  # (B, N, 3)
        
        # Convert back to euclidean coordinates
        kpts1_transformed = kpts1_transformed[..., :2] / kpts1_transformed[..., 2:3]
        
        # Compute reprojection error
        reproj_error = torch.norm(kpts2[..., :2] - kpts1_transformed, dim=2)  # (B, N)
        
        # Weight error by predicted match probabilities
        match_probs = matches.max(dim=2)[0]  # (B, N)
        weighted_error = match_probs * reproj_error
        
        return weighted_error.mean()

    def forward(self, pred: dict, target: dict) -> torch.Tensor:
        """Forward pass
        Args:
            pred: Dictionary containing model predictions
                - descriptors0: Descriptors from first image (B, N, D)
                - descriptors1: Descriptors from second image (B, N, D)
                - matches: Predicted matches (B, N, N)
            target: Dictionary containing ground truth
                - matches: Ground truth matches (B, N, N)
                - H: Ground truth homography matrices (B, 3, 3)
        Returns:
            Total loss value
        """
        # Compute contrastive loss
        contrast_loss = self.contrastive_loss(
            pred['descriptors0'], 
            pred['descriptors1'], 
            target['matches']
        )
        
        # Compute geometric verification loss
        geo_loss = self.geometric_verification_loss(
            pred['keypoints0'],
            pred['keypoints1'],
            pred['matches'],
            target['H']
        )
        
        # Combine losses
        total_loss = contrast_loss + self.geo_weight * geo_loss
        
        return total_loss 

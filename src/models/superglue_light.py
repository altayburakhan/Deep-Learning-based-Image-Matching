import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Dict, Tuple

class KeypointEncoder(nn.Module):
    """Lightweight keypoint encoder based on ResNet18"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet18 and remove final layers
        resnet = resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        
        # Keypoint head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Score map
        )
        
        # Descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 1)  # 256-dim descriptors
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Store original size for upsampling
        orig_h, orig_w = x.shape[2:]
        
        # Extract features
        features = self.backbone(x)
        
        # Generate keypoint scores and descriptors
        scores = self.keypoint_head(features)
        descriptors = self.descriptor_head(features)
        
        # Upsample scores to match input resolution
        scores = F.interpolate(scores, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
        
        return scores, descriptors

class AttentionLayer(nn.Module):
    """Multi-head attention layer for feature matching"""
    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # MLP for position-wise feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Self attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention
        attn_output, _ = self.cross_attn(x, y, y)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.mlp(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class SuperGlueLight(nn.Module):
    """Lightweight SuperGlue model for mobile AR"""
    def __init__(self, 
                 descriptor_dim: int = 256,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 l1_factor: float = 1e-5,
                 l2_factor: float = 1e-4):
        super().__init__()
        
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        
        self.keypoint_encoder = KeypointEncoder()
        
        # Positional encoding for keypoints
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, 32),  # (x, y, score, scale) -> 32
            nn.ReLU(),
            nn.Linear(32, descriptor_dim)
        )
        
        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(descriptor_dim, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # Final MLP for matching scores
        self.final_proj = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.ReLU(),
            nn.Linear(descriptor_dim, 1)
        )

    def get_regularization_loss(self) -> torch.Tensor:
        """Calculate L1 and L2 regularization losses
        Returns:
            Combined regularization loss
        """
        l1_loss = 0
        l2_loss = 0
        
        for param in self.parameters():
            if param.requires_grad:
                # L1 regularization
                l1_loss += torch.sum(torch.abs(param))
                # L2 regularization
                l2_loss += torch.sum(param ** 2)
        
        return self.l1_factor * l1_loss + self.l2_factor * l2_loss

    def forward(self, data: Dict) -> Dict:
        """Forward pass
        Args:
            data: Dictionary containing:
                - image0: First image tensor (B, C, H, W)
                - image1: Second image tensor (B, C, H, W)
                - keypoints0: Keypoints in first image (B, N, 2)
                - keypoints1: Keypoints in second image (B, N, 2)
        Returns:
            Dictionary containing:
                - scores0: Keypoint scores for image0 (B, H, W)
                - scores1: Keypoint scores for image1 (B, H, W)
                - descriptors0: Descriptors for image0 (B, N, D)
                - descriptors1: Descriptors for image1 (B, N, D)
                - matches: Matching scores matrix (B, N, N)
                - reg_loss: Regularization loss
        """
        # Extract keypoints and descriptors
        scores0, desc0 = self.keypoint_encoder(data['image0'])
        scores1, desc1 = self.keypoint_encoder(data['image1'])
        
        batch_size = scores0.shape[0]
        device = scores0.device
        
        # Get keypoint positions and scale them to feature map size
        kpts0 = data['keypoints0'].clone()  # (B, N, 2)
        kpts1 = data['keypoints1'].clone()
        
        # Scale keypoint coordinates to feature map size
        h_scale = desc0.shape[2] / scores0.shape[2]
        w_scale = desc0.shape[3] / scores0.shape[3]
        kpts0_scaled = torch.stack([
            kpts0[..., 0] * w_scale,
            kpts0[..., 1] * h_scale
        ], dim=-1)
        kpts1_scaled = torch.stack([
            kpts1[..., 0] * w_scale,
            kpts1[..., 1] * h_scale
        ], dim=-1)
        
        # Clamp scaled coordinates to feature map bounds
        kpts0_scaled = torch.clamp(kpts0_scaled, 0, desc0.shape[3] - 1)
        kpts1_scaled = torch.clamp(kpts1_scaled, 0, desc1.shape[3] - 1)
        
        # Get scores at keypoint locations
        kpts0_scores = []
        kpts1_scores = []
        
        for b in range(batch_size):
            score0 = scores0[b, 0][kpts0[b, :, 1].long(), kpts0[b, :, 0].long()]  # (N,)
            score1 = scores1[b, 0][kpts1[b, :, 1].long(), kpts1[b, :, 0].long()]
            kpts0_scores.append(score0)
            kpts1_scores.append(score1)
        
        kpts0_scores = torch.stack(kpts0_scores)  # (B, N)
        kpts1_scores = torch.stack(kpts1_scores)
        
        # Create position features (x, y, score, scale)
        pos0 = torch.cat([
            kpts0,  # Use original coordinates for position encoding
            kpts0_scores.unsqueeze(-1),
            torch.ones_like(kpts0_scores.unsqueeze(-1))  # scale
        ], dim=-1)  # (B, N, 4)
        
        pos1 = torch.cat([
            kpts1,
            kpts1_scores.unsqueeze(-1),
            torch.ones_like(kpts1_scores.unsqueeze(-1))
        ], dim=-1)
        
        # Encode positions
        pos_enc0 = self.pos_encoder(pos0)  # (B, N, D)
        pos_enc1 = self.pos_encoder(pos1)
        
        # Get descriptors at keypoint locations using scaled coordinates
        desc0_list = []
        desc1_list = []
        
        for b in range(batch_size):
            d0 = desc0[b, :, kpts0_scaled[b, :, 1].long(), kpts0_scaled[b, :, 0].long()].t()  # (N, D)
            d1 = desc1[b, :, kpts1_scaled[b, :, 1].long(), kpts1_scaled[b, :, 0].long()].t()
            desc0_list.append(d0)
            desc1_list.append(d1)
        
        desc0 = torch.stack(desc0_list)  # (B, N, D)
        desc1 = torch.stack(desc1_list)
        
        # Add position encodings to descriptors
        desc0 = desc0 + pos_enc0
        desc1 = desc1 + pos_enc1
        
        # Apply attention layers
        for layer in self.attention_layers:
            desc0 = layer(desc0.transpose(0, 1), desc1.transpose(0, 1)).transpose(0, 1)
            desc1 = layer(desc1.transpose(0, 1), desc0.transpose(0, 1)).transpose(0, 1)
        
        # Compute matching scores
        scores = torch.matmul(desc0, desc1.transpose(1, 2))  # (B, N, N)
        
        # Add regularization loss to outputs
        outputs = {
            'scores0': scores0,
            'scores1': scores1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'matches': scores,
            'reg_loss': self.get_regularization_loss()
        }
        
        return outputs

def build_superglue_light(config: Dict = None) -> nn.Module:
    """Build SuperGlue Light model
    Args:
        config: Model configuration dictionary
    Returns:
        SuperGlue Light model
    """
    if config is None:
        config = {
            'descriptor_dim': 256,
            'nhead': 4,
            'num_layers': 3,
            'dropout': 0.1
        }
    
    return SuperGlueLight(
        descriptor_dim=config['descriptor_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ) 

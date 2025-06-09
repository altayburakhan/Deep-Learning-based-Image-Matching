import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict

class ImageMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize backbone
        if config.backbone == "resnet18":
            backbone = models.resnet18(pretrained=config.pretrained)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc
        
        # Keypoint detection head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)  # Score map
        )
        
        # Descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(512, config.descriptor_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.descriptor_dim),
            nn.ReLU(),
            nn.Conv2d(config.descriptor_dim, config.descriptor_dim, kernel_size=1)
        )
        
        # Matcher (SuperGlue-inspired attention mechanism)
        self.matcher = AttentionMatcher(
            descriptor_dim=config.descriptor_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            img1: First image tensor of shape (B, C, H, W)
            img2: Second image tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing:
                - keypoints1, keypoints2: Detected keypoints
                - descriptors1, descriptors2: Feature descriptors
                - matches: Matching scores between keypoints
        """
        # Extract features
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        
        # Detect keypoints
        kp1 = self.keypoint_head(feat1)
        kp2 = self.keypoint_head(feat2)
        
        # Extract descriptors
        desc1 = self.descriptor_head(feat1)
        desc2 = self.descriptor_head(feat2)
        
        # Match descriptors
        matches = self.matcher(desc1, desc2)
        
        return {
            'keypoints1': kp1,
            'keypoints2': kp2,
            'descriptors1': desc1,
            'descriptors2': desc2,
            'matches': matches
        }

class AttentionMatcher(nn.Module):
    def __init__(self, descriptor_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=descriptor_dim,
                nhead=num_heads,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
    def forward(self, desc1: torch.Tensor, desc2: torch.Tensor) -> torch.Tensor:
        """
        Match descriptors using attention mechanism.
        
        Args:
            desc1: Descriptors from first image (B, D, H, W)
            desc2: Descriptors from second image (B, D, H, W)
            
        Returns:
            Matching scores between keypoints
        """
        # Reshape descriptors for transformer
        b, d, h, w = desc1.shape
        desc1 = desc1.flatten(2).permute(2, 0, 1)  # (H*W, B, D)
        desc2 = desc2.flatten(2).permute(2, 0, 1)  # (H*W, B, D)
        
        # Concatenate descriptors
        desc_concat = torch.cat([desc1, desc2], dim=0)
        
        # Apply transformer
        attended = self.transformer(desc_concat)
        
        # Split back and compute matching scores
        desc1_att, desc2_att = torch.split(attended, [h*w, h*w], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(desc1_att.permute(1, 0, 2), desc2_att.permute(1, 2, 0))
        
        return sim_matrix 
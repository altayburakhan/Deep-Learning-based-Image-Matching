"""Model configuration module"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Model architecture
    backbone: str = 'resnet18'
    pretrained: bool = True
    feature_dim: int = 256
    descriptor_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    attention_dropout: float = 0.1
    dropout: float = 0.1
    
    # Image parameters
    image_size: Tuple[int, int] = (256, 256)
    num_channels: int = 3
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    
    # Loss weights
    keypoint_loss_weight: float = 1.0
    descriptor_loss_weight: float = 1.0
    match_loss_weight: float = 1.0
    
    # Optimizer parameters
    optimizer_name: str = 'adam'
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        'betas': (0.9, 0.999),
        'eps': 1e-8
    })
    
    # Dataset parameters
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    num_workers: int = 4
    
    # Augmentation parameters
    use_augmentation: bool = True
    augmentation_params: Dict[str, float] = field(default_factory=lambda: {
        'p_motion': 0.4,
        'p_noise': 0.5,
        'p_perspective': 0.4
    })
    
    def update(self, **kwargs):
        """Update config parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}") 

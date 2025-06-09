from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Image settings
    image_size: int = 256
    channels: int = 3
    
    # Feature extractor settings
    backbone: str = "resnet18"
    pretrained: bool = True
    feature_dim: int = 256
    
    # Keypoint detection settings
    num_keypoints: int = 500
    nms_radius: int = 4
    
    # Descriptor settings
    descriptor_dim: int = 128
    
    # Matcher settings
    num_layers: int = 3
    num_heads: int = 4
    attention_dropout: float = 0.1
    
    # Training settings
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    
    # Data augmentation
    use_augmentation: bool = True
    contrast_limit: float = 0.2
    brightness_limit: float = 0.2
    max_rotation: float = 15.0  # degrees
    
    # Loss weights
    keypoint_loss_weight: float = 1.0
    descriptor_loss_weight: float = 1.0
    match_loss_weight: float = 1.0 

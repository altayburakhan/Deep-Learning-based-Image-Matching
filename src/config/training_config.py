"""Training configuration module"""
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Data parameters
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9
    
    # Optimizer parameters
    optimizer_name: str = 'adam'
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        'betas': (0.9, 0.999),
        'eps': 1e-8
    })
    
    # Learning rate scheduler
    scheduler_name: Optional[str] = 'cosine'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 100,
        'eta_min': 1e-6
    })
    
    # Loss function
    loss_name: str = 'cross_entropy'
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Model parameters
    model_name: str = 'resnet50'
    pretrained: bool = True
    num_classes: int = 10
    dropout_rate: float = 0.5
    
    # Regularization
    use_weight_decay: bool = True
    use_dropout: bool = True
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        'random_crop': True,
        'random_flip': True,
        'color_jitter': True
    })
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    def __post_init__(self):
        """Initialize components after dataclass initialization"""
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
    def _create_optimizer(self, parameters=None) -> optim.Optimizer:
        """Create optimizer instance
        Args:
            parameters: Model parameters to optimize
        Returns:
            Optimizer instance
        """
        if parameters is None:
            # Create dummy parameters for initialization
            parameters = nn.Linear(1, 1).parameters()
            
        optimizer_map = {
            'adam': (optim.Adam, {'betas': (0.9, 0.999), 'eps': 1e-8}),
            'sgd': (optim.SGD, {'momentum': self.momentum}),
            'adamw': (optim.AdamW, {'betas': (0.9, 0.999), 'eps': 1e-8}),
            'rmsprop': (optim.RMSprop, {'momentum': self.momentum})
        }
        
        optimizer_cls, default_params = optimizer_map.get(self.optimizer_name.lower(), (None, None))
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
            
        # Merge default parameters with user-provided parameters
        params = default_params.copy()
        if self.optimizer_params:
            # Only include parameters that are valid for this optimizer
            valid_params = {k: v for k, v in self.optimizer_params.items() 
                          if k in optimizer_cls.__init__.__code__.co_varnames}
            params.update(valid_params)
            
        return optimizer_cls(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay if self.use_weight_decay else 0,
            **params
        )
        
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler
        Returns:
            Scheduler instance if specified, else None
        """
        if not self.scheduler_name:
            return None
            
        scheduler_map = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR,
            'step': optim.lr_scheduler.StepLR,
            'plateau': optim.lr_scheduler.ReduceLROnPlateau,
            'warmup': optim.lr_scheduler.OneCycleLR
        }
        
        scheduler_cls = scheduler_map.get(self.scheduler_name.lower())
        if scheduler_cls is None:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
            
        return scheduler_cls(
            self.optimizer,
            **self.scheduler_params
        )
        
    def _create_criterion(self) -> nn.Module:
        """Create loss function
        Returns:
            Loss function instance
        """
        loss_map = {
            'cross_entropy': nn.CrossEntropyLoss,
            'bce': nn.BCEWithLogitsLoss,
            'mse': nn.MSELoss,
            'l1': nn.L1Loss
        }
        
        loss_cls = loss_map.get(self.loss_name.lower())
        if loss_cls is None:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")
            
        return loss_cls(**self.loss_params)
        
    def update(self, **kwargs):
        """Update configuration parameters
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
                
        # Reinitialize components if necessary
        self.__post_init__()
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Create config from YAML file
        Args:
            yaml_path: Path to YAML config file
        Returns:
            TrainingConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)
        
    def to_yaml(self, yaml_path: str):
        """Save config to YAML file
        Args:
            yaml_path: Output YAML file path
        """
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
        
        # Remove non-serializable objects
        config_dict.pop('optimizer', None)
        config_dict.pop('scheduler', None)
        config_dict.pop('criterion', None)
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

# Default configuration
default_config = TrainingConfig() 

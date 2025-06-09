"""Overfitting prevention and regularization module"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from torch.nn.modules.loss import _Loss

class RegularizedLoss(_Loss):
    """Loss function with regularization"""
    def __init__(self,
                 base_criterion: _Loss,
                 model: nn.Module,
                 l1_lambda: float = 1e-5,
                 l2_lambda: float = 1e-4,
                 feature_reg_lambda: float = 1e-3):
        """Initialize regularized loss
        Args:
            base_criterion: Base loss function
            model: Model to regularize
            l1_lambda: L1 regularization strength
            l2_lambda: L2 regularization strength
            feature_reg_lambda: Feature regularization strength
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.model = model
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.feature_reg_lambda = feature_reg_lambda
        
    def forward(self, 
                outputs: torch.Tensor,
                targets: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass with regularization
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            features: Optional intermediate features for regularization
        Returns:
            Total loss and loss components
        """
        # Base loss
        base_loss = self.base_criterion(outputs, targets)
        
        # L1 regularization
        l1_reg = torch.tensor(0., device=outputs.device)
        for param in self.model.parameters():
            l1_reg += torch.norm(param, p=1)
        l1_loss = self.l1_lambda * l1_reg
        
        # L2 regularization
        l2_reg = torch.tensor(0., device=outputs.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, p=2)
        l2_loss = self.l2_lambda * l2_reg
        
        # Feature regularization (if features provided)
        feature_loss = torch.tensor(0., device=outputs.device)
        if features is not None:
            # Encourage sparse and decorrelated features
            feature_loss = self.feature_reg_lambda * (
                torch.norm(features, p=1) +  # Sparsity
                torch.norm(torch.matmul(features.T, features) - 
                         torch.eye(features.shape[1], device=features.device))  # Decorrelation
            )
        
        # Total loss
        total_loss = base_loss + l1_loss + l2_loss + feature_loss
        
        # Return loss components for logging
        loss_components = {
            'base_loss': base_loss.item(),
            'l1_loss': l1_loss.item(),
            'l2_loss': l2_loss.item(),
            'feature_loss': feature_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min',
                 baseline: Optional[float] = None):
        """Initialize early stopping
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            baseline: Optional baseline value to beat
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, value: float, epoch: int) -> bool:
        """Check if training should stop
        Args:
            value: Current metric value
            epoch: Current epoch number
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
            
        # Check baseline
        if self.baseline is not None:
            if self.mode == 'min':
                improved = improved and value < self.baseline
            else:
                improved = improved and value > self.baseline
                
        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered. Best value: {self.best_value:.6f} at epoch {self.best_epoch}")
                return True
            return False

class ArchitectureSimplifier:
    """Model architecture simplification"""
    def __init__(self, model: nn.Module):
        """Initialize simplifier
        Args:
            model: Model to simplify
        """
        self.model = model
        
    def analyze_complexity(self) -> Dict[str, int]:
        """Analyze model complexity
        Returns:
            Dictionary of complexity metrics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        layer_counts = {}
        for name, module in self.model.named_modules():
            layer_type = module.__class__.__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layer_counts': layer_counts
        }
        
    def suggest_simplification(self) -> List[str]:
        """Suggest architecture simplifications
        Returns:
            List of simplification suggestions
        """
        complexity = self.analyze_complexity()
        suggestions = []
        
        # Check total parameters
        if complexity['total_params'] > 10_000_000:  # 10M params
            suggestions.append("Consider reducing model size (>10M parameters)")
            
        # Check layer distribution
        layer_counts = complexity['layer_counts']
        
        # Check for deep networks
        conv_layers = layer_counts.get('Conv2d', 0)
        if conv_layers > 20:
            suggestions.append(f"Deep network detected ({conv_layers} conv layers). Consider reducing depth.")
            
        # Check for wide layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if module.out_features > 2048:
                    suggestions.append(f"Wide layer detected in {name} ({module.out_features} units)")
                    
        return suggestions
        
    def add_dropout(self, 
                    p: float = 0.1,
                    target_layers: Optional[List[str]] = None) -> nn.Module:
        """Add dropout layers to model
        Args:
            p: Dropout probability
            target_layers: Optional list of layer names to add dropout after
        Returns:
            Modified model
        """
        if target_layers is None:
            target_layers = ['Linear', 'Conv2d']
            
        for name, module in self.model.named_children():
            if any(isinstance(module, getattr(nn, layer_type)) for layer_type in target_layers):
                # Add dropout after target layer
                setattr(self.model, name, nn.Sequential(
                    module,
                    nn.Dropout(p)
                ))
                
        return self.model

def create_regularized_model(model: nn.Module,
                           config: object) -> Tuple[nn.Module, RegularizedLoss]:
    """Create regularized model with overfitting prevention
    Args:
        model: Base model
        config: Configuration object
    Returns:
        Tuple of (regularized model, regularized loss)
    """
    # Add dropout layers
    simplifier = ArchitectureSimplifier(model)
    model = simplifier.add_dropout(p=config.dropout)
    
    # Print complexity analysis
    complexity = simplifier.analyze_complexity()
    print("Model complexity analysis:")
    print(f"Total parameters: {complexity['total_params']:,}")
    print(f"Trainable parameters: {complexity['trainable_params']:,}")
    
    # Get simplification suggestions
    suggestions = simplifier.suggest_simplification()
    if suggestions:
        print("\nArchitecture simplification suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
            
    # Create regularized loss
    base_criterion = getattr(nn, config.loss_type)()
    regularized_loss = RegularizedLoss(
        base_criterion=base_criterion,
        model=model,
        l1_lambda=config.l1_reg,
        l2_lambda=config.l2_reg,
        feature_reg_lambda=config.feature_reg
    )
    
    return model, regularized_loss 

"""Model visualization and analysis module"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import confusion_matrix
import shap
import lime.lime_image
from captum.attr import LayerGradCam, Saliency, IntegratedGradients
import wandb
from torchviz import make_dot
from collections import defaultdict
from pathlib import Path
import logging

class ModelVisualizer:
    """Model visualization utilities"""
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 class_names: List[str]):
        """Initialize visualizer
        Args:
            model: Model to visualize
            device: Device to run computations on
            class_names: List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.logger = logging.getLogger(__name__)
        self.history = defaultdict(list)
        
    def plot_training_curves(self,
                           metrics: Dict[str, List[float]],
                           save_path: Optional[str] = None) -> None:
        """Plot training curves
        Args:
            metrics: Dictionary of metric names to lists of values
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        for name, values in metrics.items():
            plt.plot(values, label=name)
            
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """Plot confusion matrix
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save plot
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_feature_importance(self,
                              importance_scores: np.ndarray,
                              feature_names: List[str],
                              save_path: Optional[str] = None) -> None:
        """Plot feature importance scores
        Args:
            importance_scores: Array of importance scores
            feature_names: List of feature names
            save_path: Optional path to save plot
        """
        # Sort by importance
        idx = np.argsort(importance_scores)
        names = np.array(feature_names)[idx]
        scores = importance_scores[idx]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(scores)), scores)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_activation_distributions(self,
                                   activations: Dict[str, torch.Tensor],
                                   save_dir: Optional[str] = None) -> None:
        """Plot activation distributions for each layer
        Args:
            activations: Dictionary of layer names to activation tensors
            save_dir: Optional directory to save plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        for name, activation in activations.items():
            plt.figure(figsize=(8, 6))
            
            # Convert to numpy and flatten
            values = activation.detach().cpu().numpy().flatten()
            
            # Plot histogram
            plt.hist(values, bins=50, density=True)
            plt.xlabel('Activation Value')
            plt.ylabel('Density')
            plt.title(f'Activation Distribution: {name}')
            
            if save_dir:
                plt.savefig(save_dir / f'{name}_distribution.png')
                plt.close()
            else:
                plt.show()
                
    def visualize_attention(self,
                          attention_weights: torch.Tensor,
                          save_path: Optional[str] = None) -> None:
        """Visualize attention weights
        Args:
            attention_weights: Attention weight tensor [head, query, key]
            save_path: Optional path to save plot
        """
        # Convert to numpy
        weights = attention_weights.detach().cpu().numpy()
        
        # Plot each attention head
        num_heads = weights.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 4))
        
        if num_heads == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            sns.heatmap(
                weights[i],
                ax=ax,
                cmap='viridis',
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(f'Head {i+1}')
            
        plt.suptitle('Attention Weights')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_model_comparison(self,
                            metrics: Dict[str, Dict[str, float]],
                            save_path: Optional[str] = None) -> None:
        """Plot model comparison
        Args:
            metrics: Dictionary of model names to metric dictionaries
            save_path: Optional path to save plot
        """
        # Prepare data
        models = list(metrics.keys())
        metric_names = list(metrics[models[0]].keys())
        
        # Create subplots
        fig, axes = plt.subplots(1, len(metric_names), figsize=(5*len(metric_names), 6))
        
        if len(metric_names) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            
            axes[i].bar(models, values)
            axes[i].set_title(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def save_visualization_report(self,
                                metrics: Dict[str, List[float]],
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                save_dir: str) -> None:
        """Save comprehensive visualization report
        Args:
            metrics: Training metrics
            y_true: True labels
            y_pred: Predicted labels
            save_dir: Directory to save report
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot and save training curves
        self.plot_training_curves(
            metrics,
            save_path=str(save_dir / 'training_curves.png')
        )
        
        # Plot and save confusion matrix
        self.plot_confusion_matrix(
            y_true,
            y_pred,
            save_path=str(save_dir / 'confusion_matrix.png')
        )
        
        self.logger.info(f"Visualization report saved to {save_dir}")

    def print_model_summary(self) -> None:
        """Print model architecture summary"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\nModel Summary:")
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("\nLayer Details:")
        print("-" * 50)
        
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {module.__class__.__name__} ({params:,} parameters)")
            
        # Create model graph visualization
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            graph = make_dot(self.model(dummy_input), 
                           params=dict(self.model.named_parameters()))
            graph.render("model_graph", format="png")
        except:
            print("Could not generate model graph visualization")
            
    def compute_gradcam(self,
                       image: torch.Tensor,
                       target_layer: nn.Module,
                       target_class: Optional[int] = None) -> np.ndarray:
        """Compute Grad-CAM visualization
        Args:
            image: Input image tensor
            target_layer: Layer to compute Grad-CAM for
            target_class: Optional target class index
        Returns:
            Grad-CAM heatmap
        """
        gradcam = LayerGradCam(self.model, target_layer)
        attribution = gradcam.attribute(image, target=target_class)
        heatmap = attribution.squeeze().cpu().detach().numpy()
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap)
        
        return heatmap
        
    def compute_feature_importance(self,
                                 images: torch.Tensor,
                                 method: str = 'shap') -> np.ndarray:
        """Compute feature importance scores
        Args:
            images: Batch of input images
            method: 'shap' or 'lime'
        Returns:
            Feature importance scores
        """
        if method == 'shap':
            # Use KernelExplainer for black-box explanation
            background = images[:100].cpu().numpy()  # Use subset for background
            explainer = shap.KernelExplainer(
                lambda x: self.model(torch.tensor(x).to(self.device)).cpu().detach().numpy(),
                background
            )
            shap_values = explainer.shap_values(images[0].cpu().numpy())
            return shap_values
            
        elif method == 'lime':
            explainer = lime.lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                images[0].cpu().numpy().transpose(1, 2, 0),
                lambda x: self.model(torch.tensor(x).to(self.device)).cpu().detach().numpy(),
                top_labels=5,
                hide_color=0,
                num_samples=1000
            )
            return explanation.local_exp
            
        else:
            raise ValueError(f"Unsupported feature importance method: {method}")
            
    def plot_activation_histograms(self,
                                 images: torch.Tensor,
                                 save_path: Optional[str] = None) -> None:
        """Plot activation histograms for each layer
        Args:
            images: Batch of input images
            save_path: Optional path to save plots
        """
        self.model.eval()
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
            
        # Register hooks
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                handles.append(module.register_forward_hook(hook_fn(name)))
                
        # Forward pass
        with torch.no_grad():
            self.model(images)
            
        # Plot histograms
        num_layers = len(activations)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 4*num_layers))
        if num_layers == 1:
            axes = [axes]
            
        for ax, (name, activation) in zip(axes, activations.items()):
            values = activation.cpu().numpy().flatten()
            ax.hist(values, bins=50)
            ax.set_title(f'Layer: {name}')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Count')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Log to wandb if available
        try:
            wandb.log({
                "activation_histograms": wandb.Image(plt),
                **{f"activation_stats/{name}": {
                    'mean': activation.mean().item(),
                    'std': activation.std().item(),
                    'min': activation.min().item(),
                    'max': activation.max().item()
                } for name, activation in activations.items()}
            })
        except:
            pass
            
    def compare_models(self,
                      other_models: List[nn.Module],
                      metrics: Dict[str, List[float]],
                      model_names: Optional[List[str]] = None) -> None:
        """Compare multiple models
        Args:
            other_models: List of models to compare
            metrics: Dictionary of metric names to lists of values
            model_names: Optional list of model names
        """
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(other_models) + 1)]
            
        # Compare architectures
        print("\nModel Architecture Comparison:")
        print("=" * 50)
        for name, model in zip(model_names, [self.model] + other_models):
            params = sum(p.numel() for p in model.parameters())
            print(f"\n{name}:")
            print(f"Total parameters: {params:,}")
            print(f"Layer composition:")
            for layer_name, module in model.named_children():
                print(f"  {layer_name}: {module.__class__.__name__}")
                
        # Compare metrics
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.8 / (len(other_models) + 1)
        
        for i, (name, model) in enumerate(zip(model_names, [self.model] + other_models)):
            values = [metrics[metric][i] for metric in metrics.keys()]
            plt.bar(x + i*width, values, width, label=name)
            
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Model Comparison')
        plt.xticks(x + width*(len(other_models))/2, list(metrics.keys()))
        plt.legend()
        plt.tight_layout()
        
        # Log to wandb if available
        try:
            wandb.log({
                "model_comparison": wandb.Image(plt),
                **{f"comparison/{name}/{metric}": value 
                   for name, model_metrics in zip(model_names, metrics.values())
                   for metric, value in model_metrics.items()}
            })
        except:
            pass 

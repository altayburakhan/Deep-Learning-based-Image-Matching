"""Test script for SuperGlue Light model"""
import torch
import torch.nn as nn
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.superglue_light import build_superglue_light

def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """Print tensor information
    Args:
        name: Tensor name
        tensor: PyTorch tensor
    """
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min/Max: {tensor.min():.3f}/{tensor.max():.3f}")

def test_model():
    """Test the SuperGlue Light model"""
    try:
        print("DEBUG: Starting test_model function")
        print("Testing SuperGlue Light model...")
        
        # Create sample input data
        print("DEBUG: Creating sample input data")
        batch_size = 2
        height, width = 256, 256
        num_keypoints = 100
        
        print("\nCreating sample input data:")
        data = {
            'image0': torch.randn(batch_size, 3, height, width),
            'image1': torch.randn(batch_size, 3, height, width),
            'keypoints0': torch.randint(0, min(height, width), (batch_size, num_keypoints, 2)),
            'keypoints1': torch.randint(0, min(height, width), (batch_size, num_keypoints, 2))
        }
        
        print("DEBUG: Created input tensors")
        for k, v in data.items():
            print_tensor_info(k, v)
        
        # Build model
        print("DEBUG: Building model...")
        model = build_superglue_light()
        print("DEBUG: Model built successfully")
        
        # Print model summary
        print("\nModel architecture:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Print regularization factors
        print("\nRegularization:")
        print(f"L1 factor: {model.l1_factor}")
        print(f"L2 factor: {model.l2_factor}")
        
        # Test forward pass
        print("\nDEBUG: Testing forward pass...")
        with torch.no_grad():
            outputs = model(data)
        print("DEBUG: Forward pass completed")
        
        # Check outputs
        print("\nOutput tensors:")
        for k, v in outputs.items():
            if k == 'reg_loss':
                print(f"{k}:")
                print(f"  Value: {v.item():.6f}")
            else:
                print_tensor_info(k, v)
            
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1) 

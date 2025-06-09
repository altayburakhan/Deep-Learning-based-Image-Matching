"""Trainer module for image matching model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import os

class Trainer:
    def __init__(self, model, config, train_dataset, val_dataset, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Enable multi-GPU training if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        # Setup data loaders with optimized settings
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count()),  # Optimize number of workers
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch next batches
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,  # Larger batch size for validation
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True
        )
        
        # Setup optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Use OneCycleLR scheduler for better convergence
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # Warm-up for 30% of training
            div_factor=25,
            final_div_factor=1000
        )
        
        # Initialize loss functions
        self.keypoint_loss = nn.MSELoss()
        self.descriptor_loss = nn.TripletMarginLoss(margin=1.0)
        self.match_loss = nn.CrossEntropyLoss()
        
        # Set loss weights
        self.keypoint_loss_weight = getattr(config, 'keypoint_loss_weight', 1.0)
        self.descriptor_loss_weight = getattr(config, 'descriptor_loss_weight', 1.0)
        self.match_loss_weight = getattr(config, 'match_loss_weight', 1.0)
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Move data to device
            img1 = batch['image1'].to(self.device)
            img2 = batch['image2'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(img1, img2)
                loss = self._compute_loss(outputs)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights and scheduler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad(), autocast():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                img1 = batch['image1'].to(self.device)
                img2 = batch['image2'].to(self.device)
                
                # Forward pass
                outputs = self.model(img1, img2)
                loss = self._compute_loss(outputs)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def _compute_loss(self, outputs):
        """Compute combined loss from model outputs."""
        keypoint_loss = self.keypoint_loss(outputs['keypoints1'], outputs['keypoints2'])
        descriptor_loss = self.descriptor_loss(
            outputs['descriptors1'],
            outputs['descriptors2'],
            outputs['descriptors2'].roll(1, dims=0)  # Negative samples
        )
        
        # Create target for match loss - identity matrix for each item in batch
        batch_size = outputs['matches'].shape[0]
        num_keypoints = outputs['matches'].shape[1]
        target = torch.eye(num_keypoints, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape predictions and targets for CrossEntropyLoss
        matches_flat = outputs['matches'].reshape(-1, num_keypoints)
        target_flat = target.reshape(-1, num_keypoints)
        match_loss = self.match_loss(matches_flat, target_flat)
        
        total_loss = (
            self.keypoint_loss_weight * keypoint_loss +
            self.descriptor_loss_weight * descriptor_loss +
            self.match_loss_weight * match_loss
        )
        
        return total_loss
    
    def train(self, num_epochs):
        """Train the model for specified number of epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Log metrics
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        save_path = Path('checkpoints') / filename
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)
        
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(Path('checkpoints') / filename)
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['config'] 

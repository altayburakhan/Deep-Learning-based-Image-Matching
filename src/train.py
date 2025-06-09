import torch
import wandb
from pathlib import Path
import argparse

from models.image_matcher import ImageMatcher
from data.dataset import ImageMatchingDataset
from training.trainer import Trainer
from config.model_config import ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train image matching model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use for training (cuda/cpu)')
    parser.add_argument('--wandb_project', type=str, default='image-matching',
                      help='W&B project name')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Initialize W&B
    wandb.init(project=args.wandb_project)
    
    # Create config
    config = ModelConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Create datasets
    train_dataset = ImageMatchingDataset(
        root_dir=args.data_dir,
        config=config,
        split='train'
    )
    
    val_dataset = ImageMatchingDataset(
        root_dir=args.data_dir,
        config=config,
        split='val'
    )
    
    # Create model
    model = ImageMatcher(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=args.device
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        config = trainer.load_checkpoint(args.checkpoint)
        print(f"Resumed from checkpoint: {args.checkpoint}")
    
    # Train model
    trainer.train(args.num_epochs)
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    
    # Close W&B
    wandb.finish()

if __name__ == '__main__':
    main() 

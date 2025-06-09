"""Script to prepare and validate the image matching dataset"""
import sys
from pathlib import Path
import logging
import argparse
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.training_config import TrainingConfig
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import create_dataloaders

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_preparation.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Prepare image matching dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Output directory for processed data')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting dataset preparation...")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize config and preprocessor
        config = TrainingConfig()
        preprocessor = DataPreprocessor(config)
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        valid_pairs, valid_homographies, statistics = preprocessor.prepare_dataset(args.data_dir)
        
        # Save dataset statistics
        stats_file = output_dir / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'mean': statistics['mean'].tolist(),
                'std': statistics['std'].tolist(),
                'num_pairs': len(valid_pairs)
            }, f, indent=2)
        
        # Create and test dataloaders
        logger.info("Testing data loading...")
        train_loader, val_loader, test_loader = create_dataloaders(args.data_dir, config)
        
        # Verify dataloaders
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Test a batch
        sample_batch = next(iter(train_loader))
        logger.info("\nSample batch shapes:")
        for k, v in sample_batch.items():
            logger.info(f"{k}: {v.shape}")
        
        logger.info("\nDataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 

"""Test script for data processing and feature engineering"""
import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import DataProcessor
from src.data.augmentation import DataAugmenter
from src.features.engineering import FeatureEngineer

class TestDataProcessing(unittest.TestCase):
    """Test suite for data processing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create dummy data
        cls.num_samples = 1000
        cls.num_features = 20
        cls.num_classes = 10
        
        # Numeric features
        cls.numeric_data = np.random.randn(cls.num_samples, cls.num_features)
        
        # Categorical features
        cls.categorical_data = np.random.randint(0, 5, (cls.num_samples, 5))
        
        # Create missing values
        cls.data_with_missing = cls.numeric_data.copy()
        mask = np.random.random(cls.data_with_missing.shape) < 0.1
        cls.data_with_missing[mask] = np.nan
        
        # Create image data
        cls.image_data = np.random.randint(0, 255, (cls.num_samples, 3, 224, 224))
        
        # Create labels
        cls.labels = np.random.randint(0, cls.num_classes, cls.num_samples)
        
    def test_data_processor(self):
        """Test data processor functionality"""
        processor = DataProcessor(
            numeric_features=list(range(self.num_features)),
            categorical_features=list(range(5))
        )
        
        # Test missing value handling
        processed_data = processor.handle_missing_values(self.data_with_missing)
        self.assertFalse(np.isnan(processed_data).any())
        
        # Test normalization
        normalized_data = processor.normalize_features(self.numeric_data)
        self.assertAlmostEqual(normalized_data.mean(), 0, places=1)
        self.assertAlmostEqual(normalized_data.std(), 1, places=1)
        
        # Test categorical encoding
        encoded_data = processor.encode_categorical(self.categorical_data)
        self.assertEqual(encoded_data.shape[1], 5 * 5)  # One-hot encoding
        
        # Test data splitting
        train_data, val_data, test_data = processor.train_val_test_split(
            self.numeric_data,
            self.labels,
            val_size=0.15,
            test_size=0.15
        )
        self.assertEqual(len(train_data), int(0.7 * self.num_samples))
        
    def test_data_augmenter(self):
        """Test data augmentation functionality"""
        augmenter = DataAugmenter(
            image_size=(224, 224),
            use_gpu=torch.cuda.is_available()
        )
        
        # Convert to torch tensor
        image_tensor = torch.from_numpy(self.image_data).float()
        
        # Test basic augmentations
        augmented = augmenter.apply_augmentation(image_tensor)
        self.assertEqual(augmented.shape, image_tensor.shape)
        
        # Test motion blur
        blurred = augmenter.apply_motion_blur(image_tensor)
        self.assertEqual(blurred.shape, image_tensor.shape)
        
        # Test camera noise
        noisy = augmenter.apply_camera_noise(image_tensor)
        self.assertEqual(noisy.shape, image_tensor.shape)
        
        # Test perspective transform
        transformed = augmenter.apply_perspective(image_tensor)
        self.assertEqual(transformed.shape, image_tensor.shape)
        
    def test_feature_engineer(self):
        """Test feature engineering functionality"""
        engineer = FeatureEngineer(
            polynomial_degree=2,
            use_scaling=True,
            max_features=100
        )
        
        # Test polynomial features
        poly_features = engineer.generate_polynomial_features(self.numeric_data)
        self.assertGreater(poly_features.shape[1], self.numeric_data.shape[1])
        
        # Test feature scaling
        scaled_features = engineer.scale_features(self.numeric_data)
        self.assertAlmostEqual(scaled_features.mean(), 0, places=1)
        self.assertAlmostEqual(scaled_features.std(), 1, places=1)
        
        # Test categorical embeddings
        embedded = engineer.create_embeddings(
            self.categorical_data,
            num_embeddings=5,
            embedding_dim=3
        )
        self.assertEqual(embedded.shape[-1], 3)
        
        # Test feature selection
        selected = engineer.select_features(self.numeric_data, self.labels)
        self.assertLessEqual(selected.shape[1], 100)
        
if __name__ == '__main__':
    unittest.main() 

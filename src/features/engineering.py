"""Feature engineering module"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Optional, Tuple

class FeatureEngineer:
    """Feature engineering utilities"""
    def __init__(self,
                 polynomial_degree: int = 2,
                 use_scaling: bool = True,
                 max_features: Optional[int] = None):
        """Initialize engineer
        Args:
            polynomial_degree: Degree for polynomial feature generation
            use_scaling: Whether to scale features
            max_features: Maximum number of features to select
        """
        self.polynomial_degree = polynomial_degree
        self.use_scaling = use_scaling
        self.max_features = max_features
        
        # Initialize components
        self.poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        self.scaler = StandardScaler() if use_scaling else None
        self.selector = SelectKBest(score_func=f_classif, k=max_features) if max_features else None
        
    def generate_polynomial_features(self, data: np.ndarray) -> np.ndarray:
        """Generate polynomial features
        Args:
            data: Input feature array
        Returns:
            Array with polynomial features
        """
        return self.poly.fit_transform(data)
        
    def scale_features(self, data: np.ndarray) -> np.ndarray:
        """Scale features to zero mean and unit variance
        Args:
            data: Input feature array
        Returns:
            Scaled features
        """
        if self.use_scaling:
            return self.scaler.fit_transform(data)
        return data
        
    def create_embeddings(self,
                         data: np.ndarray,
                         num_embeddings: int,
                         embedding_dim: int) -> torch.Tensor:
        """Create embeddings for categorical features
        Args:
            data: Input categorical data
            num_embeddings: Number of unique categories
            embedding_dim: Embedding dimension
        Returns:
            Embedded features
        """
        # Create embedding layer
        embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Convert to tensor
        data_tensor = torch.from_numpy(data).long()
        
        # Get embeddings
        embedded = embedding(data_tensor)
        
        return embedded
        
    def select_features(self,
                       data: np.ndarray,
                       labels: np.ndarray) -> np.ndarray:
        """Select most important features
        Args:
            data: Input feature array
            labels: Target labels
        Returns:
            Selected features
        """
        if self.selector:
            return self.selector.fit_transform(data, labels)
        return data
        
    def extract_domain_features(self,
                              data: np.ndarray,
                              feature_type: str = 'statistical') -> np.ndarray:
        """Extract domain-specific features
        Args:
            data: Input data array
            feature_type: Type of features to extract ('statistical', 'temporal', etc.)
        Returns:
            Extracted features
        """
        if feature_type == 'statistical':
            # Extract statistical features
            features = np.concatenate([
                np.mean(data, axis=1, keepdims=True),
                np.std(data, axis=1, keepdims=True),
                np.min(data, axis=1, keepdims=True),
                np.max(data, axis=1, keepdims=True),
                np.median(data, axis=1, keepdims=True)
            ], axis=1)
            
        elif feature_type == 'temporal':
            # Extract temporal features (assuming time series data)
            features = np.concatenate([
                np.gradient(data, axis=1),  # First derivative
                np.gradient(np.gradient(data, axis=1), axis=1),  # Second derivative
                rolling_mean(data, window=3),  # Rolling statistics
                rolling_std(data, window=3)
            ], axis=1)
            
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
            
        return features
        
    def process_features(self,
                        data: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        generate_poly: bool = True,
                        scale: bool = True,
                        select: bool = True) -> np.ndarray:
        """Process features with configured transformations
        Args:
            data: Input feature array
            labels: Optional target labels for feature selection
            generate_poly: Whether to generate polynomial features
            scale: Whether to scale features
            select: Whether to perform feature selection
        Returns:
            Processed features
        """
        processed = data.copy()
        
        if generate_poly:
            processed = self.generate_polynomial_features(processed)
            
        if scale:
            processed = self.scale_features(processed)
            
        if select and labels is not None:
            processed = self.select_features(processed, labels)
            
        return processed
        
def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling mean
    Args:
        data: Input array
        window: Window size
    Returns:
        Rolling mean array
    """
    return np.array([np.convolve(row, np.ones(window)/window, mode='same')
                    for row in data])
                    
def rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling standard deviation
    Args:
        data: Input array
        window: Window size
    Returns:
        Rolling std array
    """
    return np.array([rolling_window_std(row, window) for row in data])
    
def rolling_window_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Helper for calculating rolling std
    Args:
        arr: Input array
        window: Window size
    Returns:
        Rolling std array
    """
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    return np.pad(np.std(windows, axis=1),
                 (window//2, (window-1)//2),
                 mode='edge') 

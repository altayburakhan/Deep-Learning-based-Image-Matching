"""Main test script for the Deep Learning-based Image Matching project"""
import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.training_config import TrainingConfig
from src.training.tuning import ModelTuner
from src.visualization.model_analysis import ModelVisualizer
from src.deployment.model_export import ModelExporter, ModelServer, ModelMonitor

class SimpleTestModel(nn.Module):
    """Simple CNN for testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 112 * 112, 10)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TestImageMatching(unittest.TestCase):
    """Test suite for the image matching project"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Set device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        cls.model = SimpleTestModel().to(cls.device)
        
        # Create dummy data
        cls.batch_size = 4
        cls.input_shape = (cls.batch_size, 3, 224, 224)
        cls.dummy_input = torch.randn(*cls.input_shape).to(cls.device)
        cls.dummy_target = torch.randint(0, 10, (cls.batch_size,)).to(cls.device)
        
        # Create config
        cls.config = TrainingConfig()
        
        # Create directories
        cls.test_dir = Path('test_outputs')
        cls.test_dir.mkdir(exist_ok=True)
        
    def test_model_forward(self):
        """Test model forward pass"""
        output = self.model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 10))
        
    def test_training_config(self):
        """Test training configuration"""
        self.assertEqual(self.config.batch_size, 32)
        self.assertEqual(self.config.num_epochs, 100)
        self.assertIsNotNone(self.config.optimizer)
        
    def test_model_tuning(self):
        """Test model tuning"""
        tuner = ModelTuner(
            base_config=self.config,
            train_fn=lambda x: self.model,
            eval_fn=lambda x: 0.8,
            n_trials=2
        )
        best_params = tuner.tune()
        self.assertIsNotNone(best_params)
        
    def test_visualization(self):
        """Test visualization tools"""
        visualizer = ModelVisualizer(
            model=self.model,
            device=self.device,
            class_names=[f'class_{i}' for i in range(10)]
        )
        
        # Test training curves
        metrics = {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.8, 0.85, 0.9]
        }
        visualizer.plot_training_curves(
            metrics,
            save_path=str(self.test_dir / 'training_curves.png')
        )
        self.assertTrue((self.test_dir / 'training_curves.png').exists())
        
        # Test confusion matrix
        y_true = np.random.randint(0, 10, 100)
        y_pred = np.random.randint(0, 10, 100)
        visualizer.plot_confusion_matrix(
            y_true,
            y_pred,
            save_path=str(self.test_dir / 'confusion_matrix.png')
        )
        self.assertTrue((self.test_dir / 'confusion_matrix.png').exists())
        
    def test_model_export(self):
        """Test model export"""
        exporter = ModelExporter(
            model=self.model,
            input_shape=self.input_shape,
            device=self.device,
            export_dir=str(self.test_dir / 'exported_models')
        )
        
        # Test ONNX export
        onnx_path = exporter.export_onnx()
        self.assertTrue(Path(onnx_path).exists())
        
        # Test TorchScript export
        script_path = exporter.export_torchscript()
        self.assertTrue(Path(script_path).exists())
        
    def test_model_server(self):
        """Test model server"""
        def preprocess(x):
            return torch.randn(*self.input_shape)
            
        def postprocess(x):
            return x.argmax(dim=1).tolist()
            
        server = ModelServer(
            model=self.model,
            device=self.device,
            preprocess_fn=preprocess,
            postprocess_fn=postprocess
        )
        self.assertIsNotNone(server.app)
        
    def test_model_monitor(self):
        """Test model monitoring"""
        monitor = ModelMonitor(
            model=self.model,
            metrics_dir=str(self.test_dir / 'metrics')
        )
        
        # Test logging
        monitor.log_prediction(
            input_data=self.dummy_input,
            output=self.model(self.dummy_input),
            latency=0.1
        )
        
        # Test drift analysis
        drift_results = monitor.analyze_drift(
            reference_data=torch.randn(100),
            current_data=torch.randn(100)
        )
        self.assertIn('drift_detected', drift_results)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main() 

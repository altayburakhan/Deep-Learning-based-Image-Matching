"""Model export and deployment module"""
import torch
import torch.nn as nn
import onnx
import tensorflow as tf
import coremltools as ct
from typing import Dict, Optional, Union, List, Tuple
import time
from pathlib import Path
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from prometheus_client import Counter, Histogram, start_http_server
import mlflow.pytorch
from torch.jit import trace

class ModelExporter:
    """Model export and conversion utilities"""
    def __init__(self, 
                 model: nn.Module,
                 input_shape: Tuple[int, ...],
                 device: torch.device,
                 export_dir: str = 'exported_models'):
        """Initialize exporter
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch_size, channels, height, width)
            device: Computation device
            export_dir: Directory to save exported models
        """
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
    def export_onnx(self,
                   opset_version: int = 11,
                   dynamic_axes: Optional[Dict] = None) -> str:
        """Export model to ONNX format
        Args:
            opset_version: ONNX opset version
            dynamic_axes: Optional dynamic axes specification
        Returns:
            Path to exported model
        """
        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
                          
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Export path
        export_path = str(self.export_dir / 'model.onnx')
        
        # Export model
        torch.onnx.export(self.model,
                         dummy_input,
                         export_path,
                         opset_version=opset_version,
                         input_names=['input'],
                         output_names=['output'],
                         dynamic_axes=dynamic_axes)
                         
        # Verify exported model
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"Model exported to ONNX: {export_path}")
        return export_path
        
    def export_tensorflow(self) -> str:
        """Export model to TensorFlow SavedModel format
        Returns:
            Path to exported model
        """
        # First export to ONNX
        onnx_path = self.export_onnx()
        
        # Convert ONNX to TensorFlow
        import tf2onnx
        export_path = str(self.export_dir / 'tensorflow_model')
        
        model_proto, _ = tf2onnx.convert.from_path(
            onnx_path,
            output_path=export_path,
            opset=11
        )
        
        print(f"Model exported to TensorFlow: {export_path}")
        return export_path
        
    def export_torchscript(self,
                          method: str = 'trace') -> str:
        """Export model to TorchScript format
        Args:
            method: 'trace' or 'script'
        Returns:
            Path to exported model
        """
        self.model.eval()
        
        if method == 'trace':
            # Create dummy input
            dummy_input = torch.randn(self.input_shape).to(self.device)
            
            # Trace model
            traced_model = trace(self.model, dummy_input)
            
        elif method == 'script':
            # Script model
            scripted_model = torch.jit.script(self.model)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        # Export path
        export_path = str(self.export_dir / 'model.pt')
        
        # Save model
        if method == 'trace':
            traced_model.save(export_path)
        else:
            scripted_model.save(export_path)
            
        print(f"Model exported to TorchScript: {export_path}")
        return export_path
        
    def export_coreml(self,
                     compute_units: str = 'ALL') -> str:
        """Export model to CoreML format
        Args:
            compute_units: 'ALL', 'CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE'
        Returns:
            Path to exported model
        """
        # First export to ONNX
        onnx_path = self.export_onnx()
        
        # Convert ONNX to CoreML
        model = ct.converters.onnx.convert(
            model=onnx_path,
            compute_units=compute_units,
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Export path
        export_path = str(self.export_dir / 'model.mlmodel')
        
        # Save model
        model.save(export_path)
        
        print(f"Model exported to CoreML: {export_path}")
        return export_path
        
    def quantize_model(self,
                      backend: str = 'qnnpack',
                      dtype: str = 'qint8') -> nn.Module:
        """Quantize model for reduced size and faster inference
        Args:
            backend: Quantization backend
            dtype: Quantization data type
        Returns:
            Quantized model
        """
        self.model.eval()
        
        # Configure quantization
        if backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
        elif backend == 'fbgemm':
            torch.backends.quantized.engine = 'fbgemm'
        else:
            raise ValueError(f"Unsupported backend: {backend}")
            
        # Fuse modules
        model_fused = torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'relu']]
        )
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model_fused)
        
        # Calibrate with dummy data
        dummy_input = torch.randn(self.input_shape)
        model_prepared(dummy_input)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)
        
        # Export path
        export_path = str(self.export_dir / 'model_quantized.pt')
        
        # Save model
        torch.save(model_quantized.state_dict(), export_path)
        
        print(f"Model quantized and saved: {export_path}")
        return model_quantized

class ModelServer:
    """Model serving utilities"""
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 preprocess_fn: callable,
                 postprocess_fn: callable):
        """Initialize server
        Args:
            model: PyTorch model
            device: Computation device
            preprocess_fn: Input preprocessing function
            postprocess_fn: Output postprocessing function
        """
        self.model = model
        self.device = device
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Model API")
        
        # Initialize metrics
        self.request_counter = Counter('model_requests_total', 'Total requests')
        self.latency_histogram = Histogram('model_latency_seconds', 'Request latency')
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # Register routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        @self.app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            # Increment request counter
            self.request_counter.inc()
            
            # Record latency
            start_time = time.time()
            
            try:
                # Read and preprocess input
                contents = await file.read()
                input_tensor = self.preprocess_fn(contents)
                input_tensor = input_tensor.to(self.device)
                
                # Model inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                    
                # Postprocess output
                result = self.postprocess_fn(output)
                
                # Record latency
                latency = time.time() - start_time
                self.latency_histogram.observe(latency)
                
                return {"prediction": result, "latency": latency}
                
            except Exception as e:
                return {"error": str(e)}
                
    def run(self, host: str = '0.0.0.0', port: int = 8080):
        """Run the server
        Args:
            host: Server host
            port: Server port
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

class ModelMonitor:
    """Model monitoring utilities"""
    def __init__(self,
                 model: nn.Module,
                 metrics_dir: str = 'metrics'):
        """Initialize monitor
        Args:
            model: PyTorch model
            metrics_dir: Directory to save metrics
        """
        self.model = model
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'predictions': [],
            'latencies': [],
            'input_distributions': [],
            'output_distributions': []
        }
        
    def log_prediction(self,
                      input_data: torch.Tensor,
                      output: torch.Tensor,
                      latency: float):
        """Log prediction metrics
        Args:
            input_data: Model input
            output: Model output
            latency: Inference latency
        """
        # Store metrics - detach tensors before converting to numpy
        self.metrics['predictions'].append(output.detach().cpu().numpy())
        self.metrics['latencies'].append(latency)
        
        # Compute distributions
        input_dist = {
            'mean': input_data.mean().item(),
            'std': input_data.std().item(),
            'min': input_data.min().item(),
            'max': input_data.max().item()
        }
        self.metrics['input_distributions'].append(input_dist)
        
        output_dist = {
            'mean': output.detach().mean().item(),
            'std': output.detach().std().item(),
            'min': output.detach().min().item(),
            'max': output.detach().max().item()
        }
        self.metrics['output_distributions'].append(output_dist)
        
        # Save metrics periodically
        if len(self.metrics['predictions']) % 100 == 0:
            self.save_metrics()
            
    def save_metrics(self):
        """Save monitoring metrics"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = self.metrics_dir / f'metrics_{timestamp}.json'
        
        metrics_summary = {
            'latency_stats': {
                'mean': np.mean(self.metrics['latencies']),
                'std': np.std(self.metrics['latencies']),
                'p95': np.percentile(self.metrics['latencies'], 95),
                'p99': np.percentile(self.metrics['latencies'], 99)
            },
            'input_stats': {
                'mean': np.mean([d['mean'] for d in self.metrics['input_distributions']]),
                'std': np.mean([d['std'] for d in self.metrics['input_distributions']])
            },
            'output_stats': {
                'mean': np.mean([d['mean'] for d in self.metrics['output_distributions']]),
                'std': np.mean([d['std'] for d in self.metrics['output_distributions']])
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
            
        print(f"Metrics saved to: {save_path}")
        
    def analyze_drift(self,
                     reference_data: torch.Tensor,
                     current_data: torch.Tensor,
                     threshold: float = 0.1) -> Dict:
        """Analyze data drift
        Args:
            reference_data: Reference distribution data
            current_data: Current distribution data
            threshold: Drift detection threshold
        Returns:
            Drift analysis results
        """
        from scipy.stats import ks_2samp
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(
            reference_data.cpu().numpy().flatten(),
            current_data.cpu().numpy().flatten()
        )
        
        # Check for drift
        drift_detected = p_value < threshold
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'threshold': threshold
        }
        
    def generate_report(self) -> Dict:
        """Generate monitoring report
        Returns:
            Report dictionary
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_predictions': len(self.metrics['predictions']),
            'latency_stats': {
                'mean': np.mean(self.metrics['latencies']),
                'std': np.std(self.metrics['latencies']),
                'p95': np.percentile(self.metrics['latencies'], 95),
                'p99': np.percentile(self.metrics['latencies'], 99)
            },
            'distribution_stats': {
                'input': {
                    'mean': np.mean([d['mean'] for d in self.metrics['input_distributions']]),
                    'std': np.mean([d['std'] for d in self.metrics['input_distributions']])
                },
                'output': {
                    'mean': np.mean([d['mean'] for d in self.metrics['output_distributions']]),
                    'std': np.mean([d['std'] for d in self.metrics['output_distributions']])
                }
            }
        }
        
        return report 

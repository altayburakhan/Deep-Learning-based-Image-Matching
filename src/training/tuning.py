"""Model tuning module"""
import optuna
from typing import Callable, Dict, Any, Optional
import logging
from src.config.training_config import TrainingConfig
import torch.nn as nn
import copy

class ModelTuner:
    """Hyperparameter tuning using Optuna"""
    def __init__(self,
                 base_config: TrainingConfig,
                 train_fn: Callable[[Dict[str, Any]], nn.Module],
                 eval_fn: Callable[[nn.Module], float],
                 n_trials: int = 100,
                 timeout: Optional[int] = None):
        """Initialize tuner
        Args:
            base_config: Base training configuration
            train_fn: Function that trains model with given params and returns model
            eval_fn: Function that evaluates model and returns metric
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
        """
        self.base_config = base_config
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.n_trials = n_trials
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization
        Args:
            trial: Optuna trial object
        Returns:
            Evaluation metric (lower is better)
        """
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'adamw', 'sgd']),
        }
        
        # Create a copy of the base config and update it
        trial_config = copy.deepcopy(self.base_config)
        trial_config.update(**params)
        
        try:
            # Train model with current parameters
            model = self.train_fn(trial_config)
            
            # Evaluate model
            metric = self.eval_fn(model)
            
            return metric
            
        except Exception as e:
            self.logger.warning(f"Trial failed with error: {str(e)}")
            return float('inf')  # Return worst possible score
            
    def tune(self) -> Dict[str, Any]:
        """Run hyperparameter tuning
        Returns:
            Dictionary of best parameters
        """
        self.logger.info("Starting hyperparameter tuning...")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Log results
        self.logger.info(f"Best trial:")
        self.logger.info(f"  Value: {study.best_trial.value:.4f}")
        self.logger.info(f"  Params: {study.best_trial.params}")
        
        return study.best_trial.params 

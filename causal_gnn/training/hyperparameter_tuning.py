"""Hyperparameter optimization using Optuna for UACT-GNN."""

import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from typing import Dict, List, Optional, Callable
import logging


class HyperparameterTuner:
    """
    Hyperparameter tuning using Optuna with advanced features.
    """

    def __init__(self, training_func: Callable, config_base: Dict,
                 metric_name: str = 'ndcg@10', direction: str = 'maximize'):
        """
        Initialize hyperparameter tuner.

        Args:
            training_func: Function that trains model and returns metrics
            config_base: Base configuration dictionary
            metric_name: Metric to optimize
            direction: 'maximize' or 'minimize'
        """
        self.training_func = training_func
        self.config_base = config_base
        self.metric_name = metric_name
        self.direction = direction

        self.logger = logging.getLogger("HyperparameterTuner")

    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to optimize
        """
        # Sample hyperparameters
        config = self.config_base.copy()

        # Model architecture hyperparameters
        config['embedding_dim'] = trial.suggest_categorical(
            'embedding_dim', [32, 64, 128, 256]
        )
        config['num_layers'] = trial.suggest_int('num_layers', 2, 5)
        config['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        config['time_steps'] = trial.suggest_categorical(
            'time_steps', [5, 10, 15, 20]
        )

        # Training hyperparameters
        config['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-4, 1e-2, log=True
        )
        config['batch_size'] = trial.suggest_categorical(
            'batch_size', [256, 512, 1024, 2048, 4096]
        )
        config['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-3, log=True
        )
        config['neg_samples'] = trial.suggest_int('neg_samples', 1, 5)

        # Causal parameters
        config['causal_strength'] = trial.suggest_float(
            'causal_strength', 0.1, 1.0
        )

        self.logger.info(f"Trial {trial.number}: Testing config {config}")

        try:
            # Train model and get metrics
            metrics = self.training_func(config, trial)

            # Extract target metric
            metric_value = metrics.get(self.metric_name, 0.0)

            # Report intermediate value for pruning
            trial.report(metric_value, step=trial.number)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return metric_value

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise

    def optimize(self, n_trials: int = 100, timeout: Optional[int] = None,
                 n_jobs: int = 1, study_name: Optional[str] = None) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            study_name: Name for the study

        Returns:
            Optuna study object with results
        """
        # Create study
        sampler = TPESampler(seed=self.config_base.get('seed', 42))
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name or 'uact_gnn_optimization'
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        # Log results
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best value: {study.best_value}")
        self.logger.info(f"Best params: {study.best_params}")

        return study

    def get_best_config(self, study: optuna.Study) -> Dict:
        """
        Get best configuration from study.

        Args:
            study: Completed Optuna study

        Returns:
            Best configuration dictionary
        """
        config = self.config_base.copy()
        config.update(study.best_params)
        return config


class MultiObjectiveTuner:
    """
    Multi-objective hyperparameter tuning for accuracy + diversity + fairness.
    """

    def __init__(self, training_func: Callable, config_base: Dict,
                 objectives: List[str], directions: List[str]):
        """
        Initialize multi-objective tuner.

        Args:
            training_func: Training function
            config_base: Base configuration
            objectives: List of objective metric names
            directions: List of directions ('maximize'/'minimize') for each objective
        """
        self.training_func = training_func
        self.config_base = config_base
        self.objectives = objectives
        self.directions = directions

        self.logger = logging.getLogger("MultiObjectiveTuner")

    def objective(self, trial: Trial) -> List[float]:
        """
        Multi-objective optimization function.

        Args:
            trial: Optuna trial

        Returns:
            List of objective values
        """
        # Sample hyperparameters
        config = self.config_base.copy()

        config['embedding_dim'] = trial.suggest_categorical(
            'embedding_dim', [32, 64, 128, 256]
        )
        config['num_layers'] = trial.suggest_int('num_layers', 2, 5)
        config['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        config['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-4, 1e-2, log=True
        )
        config['batch_size'] = trial.suggest_categorical(
            'batch_size', [512, 1024, 2048, 4096]
        )

        try:
            # Train and evaluate
            metrics = self.training_func(config, trial)

            # Extract all objectives
            objective_values = []
            for obj_name in self.objectives:
                value = metrics.get(obj_name, 0.0)
                objective_values.append(value)

            return objective_values

        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            raise

    def optimize(self, n_trials: int = 100, timeout: Optional[int] = None) -> optuna.Study:
        """
        Run multi-objective optimization.

        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Optuna study
        """
        study = optuna.create_study(
            directions=self.directions,
            sampler=TPESampler(seed=self.config_base.get('seed', 42))
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Log Pareto front
        self.logger.info(f"Number of Pareto optimal trials: {len(study.best_trials)}")
        for i, trial in enumerate(study.best_trials[:5]):  # Top 5
            self.logger.info(f"Pareto trial {i + 1}:")
            self.logger.info(f"  Params: {trial.params}")
            self.logger.info(f"  Values: {trial.values}")

        return study

    def get_pareto_configs(self, study: optuna.Study) -> List[Dict]:
        """
        Get all Pareto optimal configurations.

        Args:
            study: Completed study

        Returns:
            List of Pareto optimal configurations
        """
        configs = []
        for trial in study.best_trials:
            config = self.config_base.copy()
            config.update(trial.params)
            configs.append({
                'config': config,
                'metrics': {name: val for name, val in zip(self.objectives, trial.values)}
            })
        return configs


class NeuralArchitectureSearch:
    """
    Neural Architecture Search for optimal GNN architecture.
    """

    def __init__(self, training_func: Callable, config_base: Dict):
        """
        Initialize NAS.

        Args:
            training_func: Training function
            config_base: Base configuration
        """
        self.training_func = training_func
        self.config_base = config_base
        self.logger = logging.getLogger("NeuralArchitectureSearch")

    def objective(self, trial: Trial) -> float:
        """
        NAS objective function.

        Args:
            trial: Optuna trial

        Returns:
            Metric value
        """
        config = self.config_base.copy()

        # Architecture search space
        config['embedding_dim'] = trial.suggest_categorical(
            'embedding_dim', [32, 64, 128, 256, 512]
        )

        # Number of GNN layers
        num_layers = trial.suggest_int('num_layers', 1, 6)
        config['num_layers'] = num_layers

        # Layer types (can mix different types)
        layer_types = []
        for i in range(num_layers):
            layer_type = trial.suggest_categorical(
                f'layer_{i}_type',
                ['causal', 'temporal', 'gcn', 'sage', 'transformer']
            )
            layer_types.append(layer_type)

        config['layer_types'] = layer_types

        # Attention heads for transformer layers
        config['num_attention_heads'] = trial.suggest_categorical(
            'num_attention_heads', [2, 4, 8, 16]
        )

        # Residual connections
        config['use_residual'] = trial.suggest_categorical(
            'use_residual', [True, False]
        )

        # Layer normalization
        config['use_layer_norm'] = trial.suggest_categorical(
            'use_layer_norm', [True, False]
        )

        # Dropout strategy
        config['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)

        # Activation function
        config['activation'] = trial.suggest_categorical(
            'activation', ['relu', 'gelu', 'elu', 'leaky_relu']
        )

        self.logger.info(f"Testing architecture: {config}")

        try:
            metrics = self.training_func(config, trial)
            return metrics.get('ndcg@10', 0.0)
        except Exception as e:
            self.logger.error(f"Architecture failed: {str(e)}")
            raise

    def search(self, n_trials: int = 50) -> optuna.Study:
        """
        Run neural architecture search.

        Args:
            n_trials: Number of architectures to try

        Returns:
            Optuna study
        """
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config_base.get('seed', 42)),
            pruner=MedianPruner()
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True
        )

        self.logger.info(f"Best architecture found:")
        self.logger.info(f"  Params: {study.best_params}")
        self.logger.info(f"  NDCG@10: {study.best_value:.4f}")

        return study


def visualize_optimization_history(study: optuna.Study, save_path: Optional[str] = None):
    """
    Visualize optimization history.

    Args:
        study: Completed Optuna study
        save_path: Optional path to save figure
    """
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_contour
        )
        import plotly

        # Optimization history
        fig1 = plot_optimization_history(study)
        if save_path:
            fig1.write_html(f"{save_path}_history.html")
        else:
            fig1.show()

        # Parameter importances
        fig2 = plot_param_importances(study)
        if save_path:
            fig2.write_html(f"{save_path}_importance.html")
        else:
            fig2.show()

        # Parallel coordinate plot
        fig3 = plot_parallel_coordinate(study)
        if save_path:
            fig3.write_html(f"{save_path}_parallel.html")
        else:
            fig3.show()

    except ImportError:
        logging.warning("Plotly not available for visualization")


def run_automated_tuning(training_func: Callable, base_config: Dict,
                        n_trials: int = 100, study_name: str = 'uact_gnn_tuning'):
    """
    Convenience function to run automated hyperparameter tuning.

    Args:
        training_func: Training function that takes config and returns metrics
        base_config: Base configuration dictionary
        n_trials: Number of optimization trials
        study_name: Name for the study

    Returns:
        Best configuration and study object
    """
    tuner = HyperparameterTuner(
        training_func=training_func,
        config_base=base_config,
        metric_name='ndcg@10',
        direction='maximize'
    )

    study = tuner.optimize(n_trials=n_trials, study_name=study_name)
    best_config = tuner.get_best_config(study)

    # Visualize results
    visualize_optimization_history(study, save_path=f'./output/{study_name}')

    return best_config, study

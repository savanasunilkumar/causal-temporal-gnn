"""Logging utilities for the UACT-GNN system."""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Weights & Biases logging will be disabled.")
    print("Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not installed. TensorBoard logging will be disabled.")
    print("Install with: pip install tensorboard")


def setup_logging(log_dir: str, log_file: str = 'training.log', level: int = logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_file: Name of the log file
        level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('UACT-GNN')
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


def get_logger(name: str = 'UACT-GNN') -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Unified experiment logger supporting multiple backends."""
    
    def __init__(self, config, project_name: str = 'uact-gnn', experiment_name: Optional[str] = None):
        """
        Initialize the experiment logger.
        
        Args:
            config: Configuration object
            project_name: Name of the project
            experiment_name: Name of the experiment/run
        """
        self.config = config
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = config.use_tensorboard and TENSORBOARD_AVAILABLE
        
        self.wandb_run = None
        self.tb_writer = None
        
        self._initialize_loggers()
    
    def _initialize_loggers(self):
        """Initialize the logging backends."""
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
            )
            print(f"Weights & Biases logging initialized: {self.wandb_run.url}")
        
        if self.use_tensorboard:
            log_dir = Path(self.config.log_dir) / (self.experiment_name or 'default')
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logging initialized: {log_dir}")
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step number
        """
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        if self.use_tensorboard and self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
    
    def log_hyperparameters(self, params: dict):
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if self.use_wandb:
            wandb.config.update(params)
        
        if self.use_tensorboard and self.tb_writer is not None:
            # TensorBoard's add_hparams requires metrics
            # We'll add a dummy metric for now
            self.tb_writer.add_hparams(params, {'dummy': 0})
    
    def log_model(self, model, model_name: str = 'model'):
        """
        Log model architecture and weights.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
        """
        if self.use_wandb:
            wandb.watch(model, log='all', log_freq=100)
    
    def log_text(self, key: str, text: str, step: Optional[int] = None):
        """
        Log text data.
        
        Args:
            key: Key/name for the text
            text: Text content
            step: Global step number
        """
        if self.use_wandb:
            wandb.log({key: text}, step=step)
        
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_text(key, text, step)
    
    def log_figure(self, key: str, figure, step: Optional[int] = None):
        """
        Log a matplotlib figure.
        
        Args:
            key: Key/name for the figure
            figure: Matplotlib figure object
            step: Global step number
        """
        if self.use_wandb:
            wandb.log({key: wandb.Image(figure)}, step=step)
        
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_figure(key, figure, step)
    
    def finish(self):
        """Clean up and close all loggers."""
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
        
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()


"""Model checkpointing utilities."""

import os
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any


class ModelCheckpointer:
    """Handles saving and loading model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, keep_best_k: int = 3):
        """
        Initialize the checkpointer.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_k: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_k = keep_best_k
        self.checkpoints = []  # List of (metric_value, checkpoint_path) tuples
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Any] = None,
        is_best: bool = False,
        metric_value: Optional[float] = None
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer state
            epoch: Current epoch number
            metrics: Dictionary of metrics
            config: Configuration object
            is_best: Whether this is the best model so far
            metric_value: Value of the metric being tracked (for keeping best k)
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config.to_dict() if hasattr(config, 'to_dict') else config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Track checkpoints for keeping best k
        if metric_value is not None:
            self.checkpoints.append((metric_value, str(checkpoint_path)))
            self._prune_checkpoints()
        
        # Save metadata
        metadata_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'metrics': metrics,
                'is_best': is_best
            }, f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)
    
    def _prune_checkpoints(self):
        """Keep only the best k checkpoints."""
        if len(self.checkpoints) <= self.keep_best_k:
            return
        
        # Sort by metric value (descending - higher is better)
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove checkpoints beyond best k
        to_remove = self.checkpoints[self.keep_best_k:]
        self.checkpoints = self.checkpoints[:self.keep_best_k]
        
        for _, checkpoint_path in to_remove:
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    # Also remove metadata
                    metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    print(f"Removed checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Error removing checkpoint {checkpoint_path}: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load the model on
            
        Returns:
            Dictionary containing epoch, metrics, and config
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', None)
        }
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load the model on
            
        Returns:
            Dictionary containing epoch, metrics, and config, or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        if not best_path.exists():
            print("No best checkpoint found")
            return None
        
        return self.load_checkpoint(str(best_path), model, optimizer, device)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoints[-1])
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return [str(cp) for cp in checkpoints]


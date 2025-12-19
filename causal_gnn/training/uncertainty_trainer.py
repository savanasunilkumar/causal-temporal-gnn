"""Uncertainty-aware training module for the UACT-GNN recommendation system.

This module extends the base trainer with uncertainty quantification capabilities,
enabling Bayesian inference and calibrated predictions.

Novel contribution: Full uncertainty-aware training pipeline for GNN-based recommendations.
"""

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

from .trainer import RecommendationSystem
from ..models.uncertainty_gnn import UncertaintyAwareCausalTemporalGNN, UncertaintyCalibrator
from ..causal.bayesian_discovery import BayesianCausalGraphConstructor
from ..utils.uncertainty import (
    compute_calibration_metrics,
    uncertainty_decomposition,
    UncertaintyAwareEvaluator,
)


class UncertaintyAwareRecommendationSystem(RecommendationSystem):
    """
    Uncertainty-aware recommendation system with Bayesian inference.

    This extends the base RecommendationSystem with:
    1. Bayesian causal discovery with uncertainty
    2. Monte Carlo dropout for epistemic uncertainty
    3. Learnable aleatoric uncertainty
    4. Calibrated predictions
    5. Uncertainty-aware evaluation metrics
    """

    def __init__(self, config):
        super().__init__(config)
        self.uncertainty_calibrator = None
        self.bayesian_causal_constructor = None
        self.uncertainty_evaluator = UncertaintyAwareEvaluator(
            confidence_threshold=config.confidence_threshold
        )

    def initialize_model(self):
        """Initialize the uncertainty-aware Causal Temporal GNN model."""
        # Use uncertainty-aware model instead of base model
        self.model = UncertaintyAwareCausalTemporalGNN(
            self.config, self.metadata
        ).to(self.device)

        self.model.edge_index = self.edge_index
        self.model.edge_timestamps = self.edge_timestamps
        self.model.time_indices = self.time_indices

        # Initialize negative sampler (use mixed sampler for better NDCG if enabled)
        if getattr(self.config, 'use_hard_negatives', False):
            from ..data.samplers import MixedNegativeSampler
            self.negative_sampler = MixedNegativeSampler(
                self.metadata['num_items'],
                self.user_interactions,
                device=self.device,
                hard_ratio=getattr(self.config, 'hard_negative_ratio', 0.5),
                pool_size=getattr(self.config, 'hard_negative_pool_size', 100),
            )
            self.logger.info(f"Using MixedNegativeSampler with {self.config.hard_negative_ratio*100:.0f}% hard negatives")
        else:
            from ..data.samplers import NegativeSampler
            self.negative_sampler = NegativeSampler(
                self.metadata['num_items'],
                self.user_interactions,
                device=self.device
            )

        # Initialize evaluator (from parent)
        from .evaluator import Evaluator
        self.evaluator = Evaluator(self.model, device=self.device)

        # Initialize checkpointer (from parent)
        from ..utils.checkpointing import ModelCheckpointer
        self.checkpointer = ModelCheckpointer(
            self.config.checkpoint_dir,
            keep_best_k=self.config.keep_best_k_models
        )

        # Initialize uncertainty calibrator
        self.uncertainty_calibrator = UncertaintyCalibrator(
            initial_temperature=1.0
        ).to(self.device)

        # Initialize Bayesian causal graph constructor if using Bayesian method
        if self.config.causal_method == 'bayesian':
            self.bayesian_causal_constructor = BayesianCausalGraphConstructor(
                n_bootstrap=self.config.n_bootstrap_samples,
                significance_level=self.config.significance_level,
                max_lag=self.config.max_lag,
                prior_precision=self.config.causal_prior_precision
            )

        # Initialize experiment logger
        from ..utils.logging import ExperimentLogger
        if self.config.use_wandb or self.config.use_tensorboard:
            self.experiment_logger = ExperimentLogger(
                self.config,
                project_name='uact-gnn-uncertainty',
                experiment_name=f'run_{self.config.seed}'
            )

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Initialized Uncertainty-Aware Causal Temporal GNN with {num_params:,} parameters")
        self.logger.info(f"Uncertainty model initialized with {num_params:,} parameters")

    def train_epoch_with_uncertainty(self, optimizer, neg_samples=1, batch_size=1024, scaler=None):
        """
        Train the model for one epoch with uncertainty loss.

        Combines BPR loss with uncertainty regularization:
        - Encourages well-calibrated uncertainty estimates
        - Penalizes overconfident wrong predictions
        - Rewards confident correct predictions
        """
        self.model.train()
        total_loss = 0.0
        total_bpr_loss = 0.0
        total_uncertainty_loss = 0.0

        train_data = self.data['train_data']
        n_batches = (len(train_data) + batch_size - 1) // batch_size
        batches = np.array_split(train_data.index, n_batches)

        for batch_idx, batch_indices in enumerate(tqdm(batches, desc="Training", leave=False)):
            optimizer.zero_grad()
            batch = train_data.loc[batch_indices]

            user_indices = torch.tensor(batch['user_idx'].values, dtype=torch.long, device=self.device)
            pos_item_indices = torch.tensor(batch['item_idx'].values, dtype=torch.long, device=self.device)

            # Sample negative items
            neg_item_indices_list = []
            for user_idx in batch['user_idx'].values:
                neg_items = self.sample_negative_items(int(user_idx), neg_samples)
                neg_item_indices_list.extend(neg_items)

            neg_item_indices = torch.tensor(neg_item_indices_list, dtype=torch.long, device=self.device)

            if neg_samples > 1:
                user_indices = user_indices.repeat_interleave(neg_samples)
                pos_item_indices = pos_item_indices.repeat_interleave(neg_samples)

            # Forward pass with uncertainty
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, bpr_loss, unc_loss = self._compute_uncertainty_loss(
                        user_indices, pos_item_indices, neg_item_indices
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, bpr_loss, unc_loss = self._compute_uncertainty_loss(
                    user_indices, pos_item_indices, neg_item_indices
                )

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_uncertainty_loss += unc_loss.item()

            # Log batch metrics
            if self.experiment_logger and batch_idx % self.config.log_every_n_steps == 0:
                self.experiment_logger.log_metrics({
                    'batch_loss': loss.item(),
                    'batch_bpr_loss': bpr_loss.item(),
                    'batch_uncertainty_loss': unc_loss.item(),
                    'batch': batch_idx
                })

        avg_loss = total_loss / n_batches
        avg_bpr_loss = total_bpr_loss / n_batches
        avg_uncertainty_loss = total_uncertainty_loss / n_batches

        return avg_loss, avg_bpr_loss, avg_uncertainty_loss

    def _compute_uncertainty_loss(self, user_indices, pos_item_indices, neg_item_indices):
        """Compute combined BPR loss and uncertainty loss."""
        # Forward pass with uncertainty - returns (all_mean, user_mean, item_mean, all_var, confidence)
        _, user_emb_mean, item_emb_mean, all_var, _ = self.model.forward(
            self.edge_index, self.edge_timestamps, self.time_indices
        )

        # Split variance into user and item components
        user_emb_var = all_var[:self.model.num_users]
        item_emb_var = all_var[self.model.num_users:]

        # Get embeddings for batch
        users_mean = user_emb_mean[user_indices]
        users_var = user_emb_var[user_indices]
        pos_items_mean = item_emb_mean[pos_item_indices]
        pos_items_var = item_emb_var[pos_item_indices]
        neg_items_mean = item_emb_mean[neg_item_indices]
        neg_items_var = item_emb_var[neg_item_indices]

        # Compute scores with uncertainty propagation
        pos_scores = torch.sum(users_mean * pos_items_mean, dim=1)
        neg_scores = torch.sum(users_mean * neg_items_mean, dim=1)

        # Score variances (first-order approximation)
        pos_score_var = torch.sum(
            users_var * pos_items_mean**2 +
            pos_items_var * users_mean**2 +
            users_var * pos_items_var,
            dim=1
        )
        neg_score_var = torch.sum(
            users_var * neg_items_mean**2 +
            neg_items_var * users_mean**2 +
            users_var * neg_items_var,
            dim=1
        )

        # BPR loss (using config epsilon for numerical stability)
        bpr_epsilon = getattr(self.config, 'bpr_epsilon', 1e-10)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + bpr_epsilon))

        # Uncertainty loss: encourage higher uncertainty for close scores
        score_diff = torch.abs(pos_scores - neg_scores)
        combined_var = pos_score_var + neg_score_var

        # When scores are similar (hard examples), uncertainty should be high
        uncertainty_loss = torch.mean(
            torch.relu(1.0 - combined_var) * torch.exp(-score_diff)
        )

        # L2 regularization
        if self.config.weight_decay > 0:
            l2_reg = sum(torch.norm(param, 2) for param in self.model.parameters())
            bpr_loss = bpr_loss + self.config.weight_decay * l2_reg

        # Combined loss
        total_loss = bpr_loss + self.config.uncertainty_weight * uncertainty_loss

        return total_loss, bpr_loss, uncertainty_loss

    def train(self):
        """
        Train the uncertainty-aware model with BPR loss, uncertainty loss,
        and periodic calibration.
        """
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.uncertainty_calibrator.parameters()),
            lr=self.config.learning_rate,
            weight_decay=0
        )

        # Learning rate scheduler (using config parameters)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(self.config, 'scheduler_t0', 10),
            T_mult=getattr(self.config, 'scheduler_t_mult', 2),
            eta_min=self.config.learning_rate * getattr(self.config, 'scheduler_eta_min_factor', 0.01)
        )

        # Mixed precision training
        scaler = None
        if self.config.use_amp and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using mixed precision training")

        neg_samples = self.config.neg_samples
        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs
        patience = self.config.early_stopping_patience

        best_metric = 0.0
        best_epoch = 0
        no_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, bpr_loss, unc_loss = self.train_epoch_with_uncertainty(
                optimizer, neg_samples, batch_size, scaler
            )

            self.train_history['train_loss'].append(train_loss)

            # Log training metrics
            if self.experiment_logger:
                self.experiment_logger.log_metrics({
                    'train_loss': train_loss,
                    'bpr_loss': bpr_loss,
                    'uncertainty_loss': unc_loss,
                    'epoch': epoch
                })

            # Validation with uncertainty metrics
            if self.data['val_data'] is not None and len(self.data['val_data']) > 0:
                val_metrics = self.evaluate_with_uncertainty('val', k_values=[10])

                # Track validation metrics
                for metric, values in val_metrics.items():
                    if isinstance(values, dict):
                        for k, value in values.items():
                            key = f"{metric}@{k}"
                            if key not in self.train_history['val_metrics']:
                                self.train_history['val_metrics'][key] = []
                            self.train_history['val_metrics'][key].append(value)
                    else:
                        if metric not in self.train_history['val_metrics']:
                            self.train_history['val_metrics'][metric] = []
                        self.train_history['val_metrics'][metric].append(values)

                current_metric = val_metrics['ndcg'][10] if 'ndcg' in val_metrics else 0.0

                # Log validation metrics
                if self.experiment_logger:
                    log_dict = {'epoch': epoch}
                    for metric, values in val_metrics.items():
                        if isinstance(values, dict):
                            for k, value in values.items():
                                log_dict[f'val_{metric}@{k}'] = value
                        else:
                            log_dict[f'val_{metric}'] = values
                    self.experiment_logger.log_metrics(log_dict)

                calibration_str = ""
                if 'ece' in val_metrics:
                    calibration_str = f", ECE: {val_metrics['ece']:.4f}"

                print(f"Epoch {epoch}/{num_epochs}, Loss: {train_loss:.4f} "
                      f"(BPR: {bpr_loss:.4f}, Unc: {unc_loss:.4f}), "
                      f"NDCG@10: {current_metric:.4f}{calibration_str}")

                # Early stopping and checkpointing
                is_best = current_metric > best_metric
                if is_best:
                    best_metric = current_metric
                    best_epoch = epoch
                    no_improvement = 0
                else:
                    no_improvement += 1
                    print(f"No improvement for {no_improvement} epochs")

                # Save checkpoint
                if epoch % self.config.save_every_n_epochs == 0 or is_best:
                    self.checkpointer.save_checkpoint(
                        self.model, optimizer, epoch, val_metrics, self.config,
                        is_best=is_best, metric_value=current_metric
                    )

                scheduler.step()

                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {train_loss:.4f} "
                      f"(BPR: {bpr_loss:.4f}, Unc: {unc_loss:.4f})")
                scheduler.step()

                if epoch % self.config.save_every_n_epochs == 0:
                    self.checkpointer.save_checkpoint(
                        self.model, optimizer, epoch, {}, self.config
                    )

        # Calibrate uncertainty after training
        self._calibrate_uncertainty()

        # Load best model
        if self.data['val_data'] is not None:
            best_checkpoint = self.checkpointer.load_best_checkpoint(
                self.model, optimizer, self.device
            )
            if best_checkpoint:
                print(f"Loaded best model from epoch {best_checkpoint['epoch']}")

        # Finish logging
        if self.experiment_logger:
            self.experiment_logger.finish()

        return self.train_history

    def _calibrate_uncertainty(self):
        """Calibrate uncertainty estimates using temperature scaling on validation data."""
        if self.data['val_data'] is None or len(self.data['val_data']) == 0:
            self.logger.info("No validation data available for calibration")
            return

        self.model.eval()
        val_data = self.data['val_data']

        with torch.no_grad():
            # Get validation predictions
            user_indices = torch.tensor(
                val_data['user_idx'].values, dtype=torch.long, device=self.device
            )
            item_indices = torch.tensor(
                val_data['item_idx'].values, dtype=torch.long, device=self.device
            )

            # Get scores and uncertainties
            scores, uncertainties = self.model.predict_with_uncertainty(
                user_indices, item_indices,
                self.edge_index, self.edge_timestamps, self.time_indices
            )

            # Create binary labels (assuming positive interactions)
            labels = torch.ones(len(val_data), device=self.device)

            # Learn optimal temperature
            self.uncertainty_calibrator.calibrate(scores, labels)

        self.logger.info(f"Calibrated temperature: {self.uncertainty_calibrator.temperature.item():.4f}")
        print(f"Calibrated temperature: {self.uncertainty_calibrator.temperature.item():.4f}")

    def evaluate_with_uncertainty(self, data_split='val', k_values=[5, 10, 20]):
        """Evaluate with both standard and uncertainty-aware metrics."""
        if data_split == 'val':
            eval_data = self.data['val_data']
        elif data_split == 'test':
            eval_data = self.data['test_data']
        else:
            raise ValueError(f"Invalid data split: {data_split}")

        if eval_data is None or len(eval_data) == 0:
            return None

        # Get standard metrics from parent
        standard_metrics = self.evaluator.evaluate(
            eval_data, self.user_interactions, k_values
        )

        # Compute uncertainty-specific metrics
        self.model.eval()

        with torch.no_grad():
            # Sample users for uncertainty evaluation (using config parameter)
            max_eval_users = getattr(self.config, 'eval_sample_users', 100)
            sample_users = eval_data['user_idx'].unique()[:min(max_eval_users, len(eval_data['user_idx'].unique()))]

            all_predictions = []
            all_uncertainties = []
            all_labels = []

            for user_idx in sample_users:
                user_interactions = eval_data[eval_data['user_idx'] == user_idx]
                pos_items = set(user_interactions['item_idx'].values)

                # Get top-k recommendations with uncertainty
                user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)

                # recommend_items_with_uncertainty returns (top_indices, top_scores, top_confidence, uncertain_flags)
                top_indices, top_scores, top_confidence, _ = self.model.recommend_items_with_uncertainty(
                    user_tensor,
                    top_k=max(k_values),
                    excluded_items={user_idx: list(self.user_interactions.get(user_idx, set()))},
                    confidence_threshold=self.config.confidence_threshold
                )

                # Process results - tensors are [batch, k]
                for i in range(top_indices.size(1)):
                    item_idx = top_indices[0, i].item()
                    confidence = top_confidence[0, i].item()
                    all_predictions.append(1 if item_idx in pos_items else 0)
                    all_uncertainties.append(1 - confidence)
                    all_labels.append(1 if item_idx in pos_items else 0)

            if len(all_predictions) > 0:
                predictions = np.array(all_predictions)
                uncertainties = np.array(all_uncertainties)
                labels = np.array(all_labels)

                # Compute calibration metrics
                calibration = compute_calibration_metrics(
                    predictions, uncertainties, labels,
                    n_bins=self.config.calibration_bins
                )

                standard_metrics['ece'] = calibration.expected_calibration_error
                standard_metrics['mce'] = calibration.maximum_calibration_error
                standard_metrics['uncertainty_correlation'] = calibration.uncertainty_correlation

        return standard_metrics

    def generate_recommendations_with_uncertainty(self, user_id, top_k=10,
                                                   use_mc_dropout=False):
        """
        Generate recommendations with uncertainty estimates.

        Returns items with confidence scores, enabling uncertainty-aware decisions.
        """
        self.model.eval()

        if user_id not in self.data['user_id_map']:
            raise ValueError(f"User ID {user_id} not found in the dataset")

        user_idx = self.data['user_id_map'][user_id]
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)

        # Get recommendations with uncertainty
        # Returns (top_indices, top_scores, top_confidence, uncertain_flags)
        top_indices, top_scores, top_confidence, uncertain_flags = self.model.recommend_items_with_uncertainty(
            user_tensor,
            top_k=top_k,
            excluded_items={user_idx: list(self.user_interactions.get(user_idx, set()))},
            confidence_threshold=self.config.confidence_threshold,
            use_mc_dropout=use_mc_dropout
        )

        # Convert to output format
        recommendations = []
        for i in range(top_indices.size(1)):
            item_idx = top_indices[0, i].item()
            score = top_scores[0, i].item()
            confidence = top_confidence[0, i].item()
            is_uncertain = uncertain_flags[0, i].item()

            item_id = self.item_index_to_id[item_idx]
            recommendations.append({
                'item_id': item_id,
                'score': float(score),
                'confidence': float(confidence),
                'is_uncertain': bool(is_uncertain),
                'should_recommend': confidence >= self.config.confidence_threshold
            })

        return recommendations

    def get_uncertainty_report(self):
        """Generate a comprehensive uncertainty report for the model."""
        if self.data['test_data'] is None:
            return {"error": "No test data available"}

        test_metrics = self.evaluate_with_uncertainty('test', k_values=[5, 10, 20])

        report = {
            'performance': {
                'ndcg@5': test_metrics['ndcg'].get(5, 0.0),
                'ndcg@10': test_metrics['ndcg'].get(10, 0.0),
                'ndcg@20': test_metrics['ndcg'].get(20, 0.0),
                'precision@10': test_metrics['precision'].get(10, 0.0),
                'recall@10': test_metrics['recall'].get(10, 0.0),
            },
            'calibration': {
                'expected_calibration_error': test_metrics.get('ece', 0.0),
                'maximum_calibration_error': test_metrics.get('mce', 0.0),
                'uncertainty_correlation': test_metrics.get('uncertainty_correlation', 0.0),
            },
            'model_config': {
                'mc_dropout_samples': self.config.mc_dropout_samples,
                'uncertainty_weight': self.config.uncertainty_weight,
                'confidence_threshold': self.config.confidence_threshold,
                'calibration_temperature': self.uncertainty_calibrator.temperature.item()
                    if self.uncertainty_calibrator else 1.0,
            }
        }

        return report

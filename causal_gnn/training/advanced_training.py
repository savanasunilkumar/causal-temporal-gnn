"""Advanced training techniques: knowledge distillation, curriculum learning, meta-learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class KnowledgeDistillation:
    """
    Knowledge distillation to compress large models into smaller ones.
    """

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        """
        Initialize knowledge distillation.

        Args:
            teacher_model: Large pre-trained teacher model
            student_model: Smaller student model to train
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss vs. task loss
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

        # Set teacher to eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits, true_labels, task_loss_fn):
        """
        Compute knowledge distillation loss.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            true_labels: Ground truth labels
            task_loss_fn: Task-specific loss function

        Returns:
            Combined distillation loss
        """
        # Soften probability distributions
        soft_student = F.softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence loss
        distill_loss = F.kl_div(
            soft_student.log(),
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Task loss
        task_loss = task_loss_fn(student_logits, true_labels)

        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss

        return total_loss, distill_loss, task_loss

    def train_step(self, batch, optimizer, task_loss_fn):
        """
        Single training step with knowledge distillation.

        Args:
            batch: Input batch
            optimizer: Optimizer
            task_loss_fn: Task loss function

        Returns:
            Loss values
        """
        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher(*batch['inputs'])

        # Get student predictions
        student_output = self.student(*batch['inputs'])

        # Compute distillation loss
        total_loss, distill_loss, task_loss = self.distillation_loss(
            student_output,
            teacher_output,
            batch['labels'],
            task_loss_fn
        )

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'distillation_loss': distill_loss.item(),
            'task_loss': task_loss.item()
        }


class CurriculumLearning:
    """
    Curriculum learning: train from easy to hard examples.
    """

    def __init__(self, dataset, difficulty_metric='confidence', num_stages=5):
        """
        Initialize curriculum learning.

        Args:
            dataset: Training dataset
            difficulty_metric: How to measure difficulty ('confidence', 'loss', 'manual')
            num_stages: Number of curriculum stages
        """
        self.dataset = dataset
        self.difficulty_metric = difficulty_metric
        self.num_stages = num_stages
        self.difficulties = None
        self.stage = 0

    def compute_difficulties(self, model, dataloader):
        """
        Compute difficulty scores for all examples.

        Args:
            model: Model to use for scoring
            dataloader: DataLoader for the dataset

        Returns:
            Array of difficulty scores
        """
        model.eval()
        difficulties = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(*batch['inputs'])
                labels = batch['labels']

                if self.difficulty_metric == 'confidence':
                    # Higher confidence = easier
                    probs = F.softmax(outputs, dim=-1)
                    confidences = torch.max(probs, dim=-1)[0]
                    difficulty = 1.0 - confidences  # Invert so higher = harder

                elif self.difficulty_metric == 'loss':
                    # Higher loss = harder
                    loss = F.cross_entropy(outputs, labels, reduction='none')
                    difficulty = loss

                difficulties.extend(difficulty.cpu().numpy())

        self.difficulties = np.array(difficulties)
        return self.difficulties

    def get_current_stage_indices(self):
        """
        Get indices of examples for current curriculum stage.

        Returns:
            Indices of examples to include
        """
        if self.difficulties is None:
            raise ValueError("Must call compute_difficulties first")

        # Percentage of data to include at current stage
        pct = (self.stage + 1) / self.num_stages

        # Sort by difficulty and take easiest pct%
        sorted_indices = np.argsort(self.difficulties)
        num_examples = int(len(sorted_indices) * pct)

        return sorted_indices[:num_examples]

    def advance_stage(self):
        """Move to next curriculum stage."""
        self.stage = min(self.stage + 1, self.num_stages - 1)

    def get_curriculum_dataloader(self, batch_size=32):
        """
        Get dataloader for current curriculum stage.

        Args:
            batch_size: Batch size

        Returns:
            DataLoader with current stage examples
        """
        indices = self.get_current_stage_indices()
        subset = torch.utils.data.Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)


class SelfPacedLearning:
    """
    Self-paced learning: automatically determine example weighting.
    """

    def __init__(self, lambda_init=0.1, lambda_max=10.0, growth_rate=1.5):
        """
        Initialize self-paced learning.

        Args:
            lambda_init: Initial pace parameter
            lambda_max: Maximum pace parameter
            growth_rate: Growth rate of lambda
        """
        self.lambda_val = lambda_init
        self.lambda_max = lambda_max
        self.growth_rate = growth_rate

    def compute_weights(self, losses):
        """
        Compute example weights based on losses.

        Args:
            losses: Per-example losses

        Returns:
            Per-example weights
        """
        # Weight examples inversely by loss (easier examples get higher weight)
        weights = (losses < self.lambda_val).float()
        return weights

    def weighted_loss(self, losses):
        """
        Compute weighted loss.

        Args:
            losses: Per-example losses

        Returns:
            Weighted loss
        """
        weights = self.compute_weights(losses)
        return (weights * losses).mean()

    def update_pace(self):
        """Increase pace parameter (include harder examples)."""
        self.lambda_val = min(self.lambda_val * self.growth_rate, self.lambda_max)


class MultiTaskLearning:
    """
    Multi-task learning for joint optimization of related tasks.
    """

    def __init__(self, model, task_names: List[str], task_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-task learning.

        Args:
            model: Multi-task model
            task_names: List of task names
            task_weights: Optional weights for each task
        """
        self.model = model
        self.task_names = task_names

        if task_weights is None:
            # Equal weights
            self.task_weights = {name: 1.0 for name in task_names}
        else:
            self.task_weights = task_weights

        # Learnable task weights (uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def compute_multi_task_loss(self, outputs: Dict[str, torch.Tensor],
                                targets: Dict[str, torch.Tensor],
                                loss_fns: Dict[str, callable]):
        """
        Compute weighted multi-task loss.

        Args:
            outputs: Dictionary of task outputs
            targets: Dictionary of task targets
            loss_fns: Dictionary of task loss functions

        Returns:
            Combined multi-task loss
        """
        total_loss = 0.0
        task_losses = {}

        for i, task_name in enumerate(self.task_names):
            # Task-specific loss
            task_loss = loss_fns[task_name](outputs[task_name], targets[task_name])

            # Uncertainty weighting (Kendall et al., 2018)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * task_loss + self.log_vars[i]

            total_loss += weighted_loss
            task_losses[task_name] = task_loss.item()

        return total_loss, task_losses


class MetaLearning:
    """
    Model-Agnostic Meta-Learning (MAML) for fast adaptation.
    """

    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        """
        Initialize MAML.

        Args:
            model: Model to meta-learn
            inner_lr: Learning rate for inner loop
            outer_lr: Learning rate for outer loop
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def inner_loop(self, support_data, support_labels, loss_fn):
        """
        Inner loop: adapt to support set.

        Args:
            support_data: Support set data
            support_labels: Support set labels
            loss_fn: Loss function

        Returns:
            Adapted model parameters
        """
        # Clone model parameters
        fast_weights = [p.clone() for p in self.model.parameters()]

        # Compute loss on support set
        outputs = self.model(support_data)
        loss = loss_fn(outputs, support_labels)

        # Compute gradients
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        # Update fast weights
        fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]

        return fast_weights

    def outer_loop(self, tasks, loss_fn):
        """
        Outer loop: update meta-parameters across tasks.

        Args:
            tasks: List of (support, query) task batches
            loss_fn: Loss function

        Returns:
            Meta-loss
        """
        meta_loss = 0.0

        for support_data, support_labels, query_data, query_labels in tasks:
            # Inner loop adaptation
            fast_weights = self.inner_loop(support_data, support_labels, loss_fn)

            # Evaluate on query set with adapted weights
            # (In practice, you'd need to forward with fast_weights)
            outputs = self.model(query_data)  # Simplified
            task_loss = loss_fn(outputs, query_labels)

            meta_loss += task_loss

        # Average across tasks
        meta_loss = meta_loss / len(tasks)

        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


class AdversarialTraining:
    """
    Adversarial training for robustness.
    """

    def __init__(self, model, epsilon=0.1, alpha=0.01, num_steps=10):
        """
        Initialize adversarial training.

        Args:
            model: Model to train
            epsilon: Maximum perturbation
            alpha: Step size
            num_steps: Number of adversarial steps
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate_adversarial_examples(self, data, labels, loss_fn):
        """
        Generate adversarial examples using PGD.

        Args:
            data: Input data
            labels: True labels
            loss_fn: Loss function

        Returns:
            Adversarial examples
        """
        adv_data = data.clone().detach()
        adv_data.requires_grad = True

        for _ in range(self.num_steps):
            outputs = self.model(adv_data)
            loss = loss_fn(outputs, labels)

            # Compute gradients
            loss.backward()

            # PGD step
            with torch.no_grad():
                adv_data = adv_data + self.alpha * adv_data.grad.sign()

                # Project to epsilon ball
                perturbation = torch.clamp(adv_data - data, -self.epsilon, self.epsilon)
                adv_data = data + perturbation

            adv_data.requires_grad = True

        return adv_data.detach()

    def train_step(self, data, labels, optimizer, loss_fn):
        """
        Training step with adversarial examples.

        Args:
            data: Input data
            labels: Labels
            optimizer: Optimizer
            loss_fn: Loss function

        Returns:
            Loss value
        """
        # Generate adversarial examples
        adv_data = self.generate_adversarial_examples(data, labels, loss_fn)

        # Train on both clean and adversarial
        clean_outputs = self.model(data)
        adv_outputs = self.model(adv_data)

        clean_loss = loss_fn(clean_outputs, labels)
        adv_loss = loss_fn(adv_outputs, labels)

        total_loss = 0.5 * (clean_loss + adv_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item()


class MixupAugmentation:
    """
    Mixup augmentation for recommendations.
    """

    def __init__(self, alpha=0.2):
        """
        Initialize Mixup.

        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def mixup(self, x1, x2, y1, y2):
        """
        Apply mixup to two examples.

        Args:
            x1, x2: Input features
            y1, y2: Labels

        Returns:
            Mixed inputs and labels
        """
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix inputs
        mixed_x = lam * x1 + (1 - lam) * x2

        # Mix labels
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y, lam

    def apply_to_batch(self, batch_x, batch_y):
        """
        Apply mixup to a batch.

        Args:
            batch_x: Batch of inputs
            batch_y: Batch of labels

        Returns:
            Mixed batch
        """
        indices = torch.randperm(batch_x.size(0))

        mixed_x, mixed_y, lam = self.mixup(
            batch_x,
            batch_x[indices],
            batch_y,
            batch_y[indices]
        )

        return mixed_x, mixed_y, lam


def create_advanced_trainer(model, training_technique='standard', **kwargs):
    """
    Factory function to create advanced trainer.

    Args:
        model: Model to train
        training_technique: Technique to use
        **kwargs: Additional arguments

    Returns:
        Trainer instance
    """
    if training_technique == 'distillation':
        return KnowledgeDistillation(
            teacher_model=kwargs['teacher_model'],
            student_model=model,
            **kwargs.get('distill_kwargs', {})
        )
    elif training_technique == 'curriculum':
        return CurriculumLearning(
            dataset=kwargs['dataset'],
            **kwargs.get('curriculum_kwargs', {})
        )
    elif training_technique == 'self_paced':
        return SelfPacedLearning(**kwargs.get('self_paced_kwargs', {}))
    elif training_technique == 'adversarial':
        return AdversarialTraining(model, **kwargs.get('adversarial_kwargs', {}))
    elif training_technique == 'meta':
        return MetaLearning(model, **kwargs.get('meta_kwargs', {}))
    else:
        raise ValueError(f"Unknown training technique: {training_technique}")

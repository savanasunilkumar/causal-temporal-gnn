"""PyTorch datasets for recommendation systems."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd


class RecommendationDataset(Dataset):
    """
    PyTorch Dataset for user-item interactions.

    Supports both pointwise and pairwise (BPR) training.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_col: str = 'user_idx',
        item_col: str = 'item_idx',
        rating_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        num_items: Optional[int] = None,
        user_interactions: Optional[Dict[int, Set[int]]] = None,
        negative_sampling: bool = True,
        num_negatives: int = 1,
    ):
        """
        Initialize the dataset.

        Args:
            interactions: DataFrame with user-item interactions
            user_col: Name of user column
            item_col: Name of item column
            rating_col: Name of rating column (optional)
            timestamp_col: Name of timestamp column (optional)
            num_items: Total number of items (for negative sampling)
            user_interactions: Dict mapping user_idx to set of positive item indices
            negative_sampling: Whether to use negative sampling
            num_negatives: Number of negative samples per positive
        """
        self.interactions = interactions.reset_index(drop=True)
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.num_items = num_items or interactions[item_col].max() + 1
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives

        # Build user interactions dict if not provided
        if user_interactions is not None:
            self.user_interactions = user_interactions
        else:
            self.user_interactions = {}
            for _, row in interactions.iterrows():
                user_idx = int(row[user_col])
                item_idx = int(row[item_col])
                if user_idx not in self.user_interactions:
                    self.user_interactions[user_idx] = set()
                self.user_interactions[user_idx].add(item_idx)

        # Pre-compute arrays for faster access
        self.users = self.interactions[user_col].values.astype(np.int64)
        self.items = self.interactions[item_col].values.astype(np.int64)

        if rating_col and rating_col in interactions.columns:
            self.ratings = self.interactions[rating_col].values.astype(np.float32)
        else:
            self.ratings = np.ones(len(self.interactions), dtype=np.float32)

        if timestamp_col and timestamp_col in interactions.columns:
            self.timestamps = self.interactions[timestamp_col].values.astype(np.int64)
        else:
            self.timestamps = np.arange(len(self.interactions), dtype=np.int64)

    def __len__(self) -> int:
        """Return number of interactions."""
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with user, positive item, (optional) negative items, rating, timestamp
        """
        user_idx = self.users[idx]
        pos_item_idx = self.items[idx]
        rating = self.ratings[idx]
        timestamp = self.timestamps[idx]

        sample = {
            'user': torch.tensor(user_idx, dtype=torch.long),
            'pos_item': torch.tensor(pos_item_idx, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float),
            'timestamp': torch.tensor(timestamp, dtype=torch.long),
        }

        # Add negative samples if enabled
        if self.negative_sampling:
            neg_items = self._sample_negatives(user_idx, self.num_negatives)
            sample['neg_items'] = torch.tensor(neg_items, dtype=torch.long)

        return sample

    def _sample_negatives(self, user_idx: int, num_negatives: int) -> List[int]:
        """Sample negative items for a user."""
        positive_items = self.user_interactions.get(user_idx, set())
        neg_items = []
        max_attempts = num_negatives * 10  # Prevent infinite loops
        attempts = 0

        while len(neg_items) < num_negatives and attempts < max_attempts:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        # If we couldn't find enough negatives, pad with random items
        while len(neg_items) < num_negatives:
            neg_items.append(np.random.randint(0, self.num_items))

        return neg_items


class PairwiseDataset(Dataset):
    """
    Dataset for pairwise (BPR) training with pre-sampled negatives.

    More efficient than on-the-fly sampling for large datasets.
    """

    def __init__(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        neg_items: np.ndarray,
    ):
        """
        Initialize with pre-sampled triplets.

        Args:
            users: Array of user indices
            pos_items: Array of positive item indices
            neg_items: Array of negative item indices
        """
        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.pos_items[idx], dtype=torch.long),
            torch.tensor(self.neg_items[idx], dtype=torch.long),
        )


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    user_col: str = 'user_idx',
    item_col: str = 'item_idx',
    rating_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    num_items: Optional[int] = None,
    batch_size: int = 1024,
    num_workers: int = 4,
    negative_sampling: bool = True,
    num_negatives: int = 1,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for training, validation, and test sets.

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame (optional)
        test_data: Test DataFrame (optional)
        user_col: Name of user column
        item_col: Name of item column
        rating_col: Name of rating column (optional)
        timestamp_col: Name of timestamp column (optional)
        num_items: Total number of items
        batch_size: Batch size
        num_workers: Number of data loading workers
        negative_sampling: Whether to use negative sampling
        num_negatives: Number of negative samples per positive
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Build user interactions from training data
    user_interactions = {}
    for _, row in train_data.iterrows():
        user_idx = int(row[user_col])
        item_idx = int(row[item_col])
        if user_idx not in user_interactions:
            user_interactions[user_idx] = set()
        user_interactions[user_idx].add(item_idx)

    # Determine num_items
    if num_items is None:
        all_items = set(train_data[item_col].unique())
        if val_data is not None:
            all_items.update(val_data[item_col].unique())
        if test_data is not None:
            all_items.update(test_data[item_col].unique())
        num_items = max(all_items) + 1

    # Create training dataset and loader
    train_dataset = RecommendationDataset(
        interactions=train_data,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        timestamp_col=timestamp_col,
        num_items=num_items,
        user_interactions=user_interactions,
        negative_sampling=negative_sampling,
        num_negatives=num_negatives,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Create validation loader
    val_loader = None
    if val_data is not None and len(val_data) > 0:
        val_dataset = RecommendationDataset(
            interactions=val_data,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            timestamp_col=timestamp_col,
            num_items=num_items,
            user_interactions=user_interactions,
            negative_sampling=False,  # No negative sampling for validation
            num_negatives=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Create test loader
    test_loader = None
    if test_data is not None and len(test_data) > 0:
        test_dataset = RecommendationDataset(
            interactions=test_data,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            timestamp_col=timestamp_col,
            num_items=num_items,
            user_interactions=user_interactions,
            negative_sampling=False,  # No negative sampling for test
            num_negatives=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    result = {}
    for key in batch[0].keys():
        if key == 'neg_items':
            # Stack negative items into a 2D tensor
            result[key] = torch.stack([sample[key] for sample in batch])
        else:
            result[key] = torch.stack([sample[key] for sample in batch])
    return result

"""Negative sampling strategies for recommendation systems."""

import numpy as np
import torch
from typing import Dict, Set, List, Optional, Union
from collections import defaultdict


class NegativeSampler:
    """
    Efficient negative sampler for recommendation systems.

    Supports uniform sampling and popularity-based sampling strategies.
    """

    def __init__(
        self,
        num_items: int,
        user_interactions: Dict[int, Set[int]],
        device: Union[str, torch.device] = 'cpu',
        strategy: str = 'uniform',
        item_popularity: Optional[np.ndarray] = None,
    ):
        """
        Initialize the negative sampler.

        Args:
            num_items: Total number of items in the catalog
            user_interactions: Dictionary mapping user_idx to set of positive item indices
            device: Device for tensor operations
            strategy: Sampling strategy ('uniform' or 'popularity')
            item_popularity: Item popularity counts (required for popularity strategy)
        """
        self.num_items = num_items
        self.user_interactions = user_interactions
        self.device = torch.device(device) if isinstance(device, str) else device
        self.strategy = strategy

        # Pre-compute item popularity distribution for popularity-based sampling
        if strategy == 'popularity':
            if item_popularity is not None:
                self.item_probs = item_popularity / item_popularity.sum()
            else:
                # Compute from user_interactions
                item_counts = np.zeros(num_items)
                for items in user_interactions.values():
                    for item in items:
                        if item < num_items:
                            item_counts[item] += 1
                # Add smoothing to avoid zero probabilities
                item_counts += 1
                self.item_probs = item_counts / item_counts.sum()
        else:
            self.item_probs = None

        # Cache for vectorized sampling
        self._all_items = np.arange(num_items)

    def sample(self, user_idx: int, num_negatives: int = 1) -> List[int]:
        """
        Sample negative items for a single user.

        Args:
            user_idx: User index
            num_negatives: Number of negative samples to generate

        Returns:
            List of negative item indices
        """
        positive_items = self.user_interactions.get(user_idx, set())
        neg_items = []
        max_attempts = num_negatives * 20  # Increased for safety
        attempts = 0

        while len(neg_items) < num_negatives and attempts < max_attempts:
            if self.strategy == 'popularity' and self.item_probs is not None:
                neg_item = np.random.choice(self.num_items, p=self.item_probs)
            else:
                neg_item = np.random.randint(0, self.num_items)

            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        # If we still don't have enough, fill with random items (may include positives)
        while len(neg_items) < num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in neg_items:  # At least avoid duplicates
                neg_items.append(neg_item)

        return neg_items

    def sample_batch(
        self,
        user_indices: Union[List[int], np.ndarray, torch.Tensor],
        num_negatives: int = 1,
        return_tensor: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample negative items for a batch of users.

        Args:
            user_indices: Array of user indices
            num_negatives: Number of negative samples per user
            return_tensor: Whether to return a PyTorch tensor

        Returns:
            Array of shape (batch_size, num_negatives) with negative item indices
        """
        if isinstance(user_indices, torch.Tensor):
            user_indices = user_indices.cpu().numpy()

        batch_size = len(user_indices)
        neg_items = np.zeros((batch_size, num_negatives), dtype=np.int64)

        for i, user_idx in enumerate(user_indices):
            neg_items[i] = self.sample(int(user_idx), num_negatives)

        if return_tensor:
            return torch.tensor(neg_items, dtype=torch.long, device=self.device)
        return neg_items

    def sample_vectorized(
        self,
        user_indices: torch.Tensor,
        num_negatives: int = 1,
    ) -> torch.Tensor:
        """
        Vectorized negative sampling (faster for large batches on GPU).

        Note: This may occasionally sample positive items as negatives,
        but the probability is very low for sparse datasets.

        Args:
            user_indices: Tensor of user indices
            num_negatives: Number of negative samples per user

        Returns:
            Tensor of shape (batch_size, num_negatives) with negative item indices
        """
        batch_size = user_indices.size(0)

        if self.strategy == 'popularity' and self.item_probs is not None:
            # Popularity-based sampling
            probs = torch.tensor(self.item_probs, device=self.device)
            neg_items = torch.multinomial(
                probs.expand(batch_size, -1),
                num_negatives,
                replacement=False
            )
        else:
            # Uniform sampling
            neg_items = torch.randint(
                0, self.num_items,
                (batch_size, num_negatives),
                device=self.device
            )

        return neg_items


class HardNegativeSampler(NegativeSampler):
    """
    Hard negative sampler that samples items similar to positives.

    Useful for improving model discrimination.
    """

    def __init__(
        self,
        num_items: int,
        user_interactions: Dict[int, Set[int]],
        item_embeddings: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = 'cpu',
        num_candidates: int = 100,
        temperature: float = 1.0,
    ):
        """
        Initialize hard negative sampler.

        Args:
            num_items: Total number of items
            user_interactions: Dictionary mapping user_idx to set of positive items
            item_embeddings: Pre-computed item embeddings for similarity
            device: Device for tensor operations
            num_candidates: Number of candidates to consider for hard negatives
            temperature: Temperature for softmax sampling (lower = harder negatives)
        """
        super().__init__(num_items, user_interactions, device, strategy='uniform')
        self.item_embeddings = item_embeddings
        self.num_candidates = num_candidates
        self.temperature = temperature

        # Pre-compute item-item similarities if embeddings provided
        if item_embeddings is not None:
            self._compute_similarities()

    def _compute_similarities(self):
        """Compute item-item cosine similarities."""
        if self.item_embeddings is None:
            self.item_similarities = None
            return

        # Normalize embeddings
        embeddings = self.item_embeddings
        norms = embeddings.norm(dim=1, keepdim=True)
        normalized = embeddings / (norms + 1e-8)

        # Compute cosine similarity (may be memory intensive for large catalogs)
        if self.num_items <= 10000:
            self.item_similarities = torch.mm(normalized, normalized.t())
        else:
            # For large catalogs, we'll compute on-the-fly
            self.item_similarities = None
            self.normalized_embeddings = normalized

    def sample_hard_negatives(
        self,
        user_idx: int,
        pos_item_idx: int,
        num_negatives: int = 1,
    ) -> List[int]:
        """
        Sample hard negatives similar to a positive item.

        Args:
            user_idx: User index
            pos_item_idx: Positive item index
            num_negatives: Number of hard negatives to sample

        Returns:
            List of hard negative item indices
        """
        positive_items = self.user_interactions.get(user_idx, set())

        if self.item_embeddings is None:
            # Fall back to uniform sampling
            return self.sample(user_idx, num_negatives)

        # Get similarities to positive item
        if self.item_similarities is not None:
            similarities = self.item_similarities[pos_item_idx]
        else:
            pos_emb = self.normalized_embeddings[pos_item_idx:pos_item_idx+1]
            similarities = torch.mm(pos_emb, self.normalized_embeddings.t()).squeeze()

        # Mask positive items
        mask = torch.zeros(self.num_items, device=self.device)
        for item in positive_items:
            if item < self.num_items:
                mask[item] = float('-inf')
        similarities = similarities + mask

        # Sample from top candidates
        top_k = min(self.num_candidates, self.num_items)
        _, top_indices = torch.topk(similarities, top_k)

        # Sample with temperature
        top_similarities = similarities[top_indices]
        probs = torch.softmax(top_similarities / self.temperature, dim=0)

        sampled_indices = torch.multinomial(probs, min(num_negatives, top_k), replacement=False)
        neg_items = top_indices[sampled_indices].cpu().tolist()

        # Pad if needed
        while len(neg_items) < num_negatives:
            neg_items.extend(self.sample(user_idx, num_negatives - len(neg_items)))

        return neg_items[:num_negatives]


class DynamicNegativeSampler:
    """
    Dynamic negative sampler that updates item popularity during training.

    Useful for online learning scenarios.
    """

    def __init__(
        self,
        num_items: int,
        device: Union[str, torch.device] = 'cpu',
        decay: float = 0.99,
    ):
        """
        Initialize dynamic sampler.

        Args:
            num_items: Total number of items
            device: Device for tensor operations
            decay: Decay factor for popularity scores
        """
        self.num_items = num_items
        self.device = torch.device(device) if isinstance(device, str) else device
        self.decay = decay

        # Initialize popularity scores uniformly
        self.popularity = np.ones(num_items)
        self.user_history = defaultdict(set)

    def update(self, user_idx: int, item_idx: int):
        """Update sampler with a new interaction."""
        self.user_history[user_idx].add(item_idx)
        self.popularity[item_idx] += 1
        # Apply decay to all items
        self.popularity *= self.decay

    def sample(self, user_idx: int, num_negatives: int = 1) -> List[int]:
        """Sample negative items avoiding user's history."""
        positive_items = self.user_history.get(user_idx, set())
        neg_items = []
        max_attempts = num_negatives * 20

        # Normalize popularity for sampling
        probs = self.popularity / self.popularity.sum()

        attempts = 0
        while len(neg_items) < num_negatives and attempts < max_attempts:
            neg_item = np.random.choice(self.num_items, p=probs)
            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        # Fallback to uniform if needed
        while len(neg_items) < num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in neg_items:
                neg_items.append(neg_item)

        return neg_items

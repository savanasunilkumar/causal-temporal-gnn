"""Evaluation utilities for recommendation models."""

import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


class Evaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the evaluator.
        
        Args:
            model: The recommendation model
            device: Device to use for evaluation
        """
        self.model = model
        self.device = torch.device(device)
    
    def evaluate(self, eval_data, user_interactions, k_values=[5, 10, 20], batch_size=1024):
        """
        Evaluate the model on given data.
        
        Args:
            eval_data: DataFrame with evaluation data
            user_interactions: Dictionary mapping user_idx to set of training items
            k_values: List of k values for top-k metrics
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        metrics = {
            'precision': {k: 0.0 for k in k_values},
            'recall': {k: 0.0 for k in k_values},
            'ndcg': {k: 0.0 for k in k_values},
            'hit_ratio': {k: 0.0 for k in k_values}
        }
        
        # Group test items by user
        user_test_items = defaultdict(list)
        for _, row in eval_data.iterrows():
            user_test_items[row['user_idx']].append(row['item_idx'])
        
        test_users = list(user_test_items.keys())
        n_batches = (len(test_users) + batch_size - 1) // batch_size
        user_batches = np.array_split(test_users, n_batches)
        
        with torch.no_grad():
            for user_batch in tqdm(user_batches, desc="Evaluating", leave=False):
                user_indices = torch.tensor(user_batch, dtype=torch.long, device=self.device)
                
                # Exclude training items from recommendations
                excluded_items = {}
                for user_idx in user_batch:
                    excluded_items[user_idx] = list(user_interactions[user_idx])
                
                # Get top-k recommendations
                top_indices, top_scores = self.model.recommend_items(
                    user_indices, top_k=max(k_values), excluded_items=excluded_items
                )
                
                top_indices = top_indices.cpu().numpy()
                
                # Compute metrics for each user
                for i, user_idx in enumerate(user_batch):
                    true_items = user_test_items[user_idx]
                    recommended_items = top_indices[i]
                    
                    for k in k_values:
                        top_k_items = recommended_items[:k]
                        
                        # Precision@k
                        precision = len(set(top_k_items) & set(true_items)) / k
                        metrics['precision'][k] += precision
                        
                        # Recall@k
                        recall = len(set(top_k_items) & set(true_items)) / len(true_items) if true_items else 0
                        metrics['recall'][k] += recall
                        
                        # Hit Ratio@k
                        hit_ratio = 1.0 if len(set(top_k_items) & set(true_items)) > 0 else 0.0
                        metrics['hit_ratio'][k] += hit_ratio
                        
                        # NDCG@k
                        dcg = 0.0
                        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
                        
                        for j, item in enumerate(top_k_items):
                            if item in true_items:
                                dcg += 1.0 / np.log2(j + 2)
                        
                        ndcg = dcg / idcg if idcg > 0 else 0
                        metrics['ndcg'][k] += ndcg
        
        # Average metrics over all users
        num_users = len(test_users)
        for metric in metrics:
            for k in k_values:
                metrics[metric][k] /= num_users
        
        return metrics
    
    @staticmethod
    def compute_diversity(recommendations, item_features=None):
        """
        Compute diversity of recommendations.
        
        Args:
            recommendations: List of recommended item lists
            item_features: Optional item feature matrix for computing similarity
            
        Returns:
            Diversity score
        """
        # Simple diversity: average number of unique items
        all_items = set()
        for rec_list in recommendations:
            all_items.update(rec_list)
        
        diversity = len(all_items) / (len(recommendations) * len(recommendations[0]))
        return diversity
    
    @staticmethod
    def compute_coverage(recommendations, num_items):
        """
        Compute catalog coverage of recommendations.
        
        Args:
            recommendations: List of recommended item lists
            num_items: Total number of items in catalog
            
        Returns:
            Coverage score (fraction of catalog recommended)
        """
        all_items = set()
        for rec_list in recommendations:
            all_items.update(rec_list)
        
        coverage = len(all_items) / num_items
        return coverage


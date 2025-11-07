"""Advanced evaluation metrics for recommendation systems."""

import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from scipy.stats import entropy


class DiversityMetrics:
    """
    Compute diversity metrics for recommendation lists.
    """

    @staticmethod
    def intra_list_diversity(recommendations, item_embeddings):
        """
        Compute intra-list diversity (average pairwise dissimilarity).

        Args:
            recommendations: List of recommended item indices for each user [[items_u1], [items_u2], ...]
            item_embeddings: Item embedding matrix [num_items, embedding_dim]

        Returns:
            Average intra-list diversity score
        """
        diversities = []

        for rec_list in recommendations:
            if len(rec_list) < 2:
                continue

            pairwise_dists = []
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    item_i_emb = item_embeddings[rec_list[i]].cpu().numpy()
                    item_j_emb = item_embeddings[rec_list[j]].cpu().numpy()

                    # Cosine distance
                    dist = cosine(item_i_emb, item_j_emb)
                    pairwise_dists.append(dist)

            if pairwise_dists:
                diversities.append(np.mean(pairwise_dists))

        return np.mean(diversities) if diversities else 0.0

    @staticmethod
    def category_diversity(recommendations, item_categories):
        """
        Compute diversity based on item categories.

        Args:
            recommendations: List of recommended item indices
            item_categories: Dict mapping item_idx to category

        Returns:
            Average number of unique categories per recommendation list
        """
        category_counts = []

        for rec_list in recommendations:
            categories = set()
            for item_idx in rec_list:
                if item_idx in item_categories:
                    categories.add(item_categories[item_idx])

            category_counts.append(len(categories))

        return np.mean(category_counts) if category_counts else 0.0

    @staticmethod
    def coverage(recommendations, num_items):
        """
        Compute catalog coverage (percentage of items recommended).

        Args:
            recommendations: List of recommended item indices
            num_items: Total number of items

        Returns:
            Coverage percentage [0, 1]
        """
        recommended_items = set()
        for rec_list in recommendations:
            recommended_items.update(rec_list)

        return len(recommended_items) / num_items

    @staticmethod
    def gini_coefficient(item_counts):
        """
        Compute Gini coefficient for item popularity distribution.
        Lower values indicate more equal distribution (better diversity).

        Args:
            item_counts: Array of recommendation counts per item

        Returns:
            Gini coefficient [0, 1]
        """
        sorted_counts = np.sort(item_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n


class NoveltyMetrics:
    """
    Compute novelty and serendipity metrics.
    """

    @staticmethod
    def novelty_score(recommendations, item_popularity):
        """
        Compute novelty score based on item popularity.
        Less popular items are more novel.

        Args:
            recommendations: List of recommended item indices
            item_popularity: Dict mapping item_idx to popularity score

        Returns:
            Average novelty score
        """
        novelty_scores = []

        for rec_list in recommendations:
            for item_idx in rec_list:
                if item_idx in item_popularity:
                    # Novelty = -log(popularity)
                    pop = item_popularity[item_idx]
                    novelty = -np.log2(pop + 1e-10)
                    novelty_scores.append(novelty)

        return np.mean(novelty_scores) if novelty_scores else 0.0

    @staticmethod
    def serendipity_score(recommendations, user_history, item_embeddings, threshold=0.7):
        """
        Compute serendipity: relevant but unexpected recommendations.

        Args:
            recommendations: Dict {user_idx: [recommended_items]}
            user_history: Dict {user_idx: [historical_items]}
            item_embeddings: Item embedding matrix
            threshold: Similarity threshold for "unexpected"

        Returns:
            Average serendipity score
        """
        serendipity_scores = []

        for user_idx, rec_list in recommendations.items():
            if user_idx not in user_history or not user_history[user_idx]:
                continue

            history = user_history[user_idx]

            for rec_item in rec_list:
                # Check if recommendation is dissimilar from history
                is_unexpected = True
                rec_emb = item_embeddings[rec_item].cpu().numpy()

                for hist_item in history:
                    hist_emb = item_embeddings[hist_item].cpu().numpy()
                    similarity = 1 - cosine(rec_emb, hist_emb)

                    if similarity > threshold:
                        is_unexpected = False
                        break

                if is_unexpected:
                    serendipity_scores.append(1.0)
                else:
                    serendipity_scores.append(0.0)

        return np.mean(serendipity_scores) if serendipity_scores else 0.0

    @staticmethod
    def unexpectedness(recommendations, expected_items, k=10):
        """
        Measure unexpectedness compared to expected/popular items.

        Args:
            recommendations: List of recommended item indices
            expected_items: Set of expected/popular items
            k: Top-k to consider

        Returns:
            Unexpectedness score
        """
        unexpected_count = 0
        total_count = 0

        for rec_list in recommendations:
            for item_idx in rec_list[:k]:
                total_count += 1
                if item_idx not in expected_items:
                    unexpected_count += 1

        return unexpected_count / total_count if total_count > 0 else 0.0


class FairnessMetrics:
    """
    Compute fairness metrics for recommendations.
    """

    @staticmethod
    def user_fairness(metrics_by_user):
        """
        Measure fairness across users using standard deviation of metrics.

        Args:
            metrics_by_user: Dict {user_idx: metric_value}

        Returns:
            Fairness score (lower std = more fair)
        """
        values = list(metrics_by_user.values())
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'gini': DiversityMetrics.gini_coefficient(np.array(values))
        }

    @staticmethod
    def provider_fairness(recommendations, num_items):
        """
        Measure fairness in exposure across items (providers).

        Args:
            recommendations: List of recommended item indices
            num_items: Total number of items

        Returns:
            Provider fairness metrics
        """
        item_exposure = np.zeros(num_items)

        for rec_list in recommendations:
            for item_idx in rec_list:
                item_exposure[item_idx] += 1

        # Normalize
        item_exposure = item_exposure / (len(recommendations) + 1e-10)

        return {
            'exposure_mean': np.mean(item_exposure),
            'exposure_std': np.std(item_exposure),
            'exposure_gini': DiversityMetrics.gini_coefficient(item_exposure),
            'items_never_recommended': np.sum(item_exposure == 0)
        }

    @staticmethod
    def demographic_parity(recommendations_by_group):
        """
        Measure demographic parity across user groups.

        Args:
            recommendations_by_group: Dict {group_id: [metrics]}

        Returns:
            Demographic parity score
        """
        group_means = {}
        for group_id, metrics in recommendations_by_group.items():
            group_means[group_id] = np.mean(metrics)

        # Compute max difference
        mean_values = list(group_means.values())
        max_diff = np.max(mean_values) - np.min(mean_values)

        return {
            'group_means': group_means,
            'max_difference': max_diff,
            'std_across_groups': np.std(mean_values)
        }


class TemporalMetrics:
    """
    Metrics for temporal consistency and stability.
    """

    @staticmethod
    def recommendation_stability(recs_t1, recs_t2, k=10):
        """
        Measure stability of recommendations over time.

        Args:
            recs_t1: Recommendations at time t1 {user_idx: [items]}
            recs_t2: Recommendations at time t2 {user_idx: [items]}
            k: Top-k to consider

        Returns:
            Average Jaccard similarity between consecutive recommendations
        """
        stabilities = []

        for user_idx in recs_t1:
            if user_idx not in recs_t2:
                continue

            set1 = set(recs_t1[user_idx][:k])
            set2 = set(recs_t2[user_idx][:k])

            if len(set1) > 0 or len(set2) > 0:
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                stabilities.append(jaccard)

        return np.mean(stabilities) if stabilities else 0.0

    @staticmethod
    def temporal_diversity(recommendations_over_time):
        """
        Measure diversity of recommendations over time for each user.

        Args:
            recommendations_over_time: List of recommendation dicts over time
                                      [{user_idx: [items]}, {user_idx: [items]}, ...]

        Returns:
            Average temporal diversity
        """
        user_diversities = defaultdict(list)

        # Collect all recommendations per user
        for recs in recommendations_over_time:
            for user_idx, items in recs.items():
                user_diversities[user_idx].extend(items)

        # Compute unique item ratio per user
        diversity_scores = []
        for user_idx, all_items in user_diversities.items():
            unique_ratio = len(set(all_items)) / len(all_items) if all_items else 0
            diversity_scores.append(unique_ratio)

        return np.mean(diversity_scores) if diversity_scores else 0.0


class BeyondAccuracyEvaluator:
    """
    Comprehensive evaluator for beyond-accuracy metrics.
    """

    def __init__(self, model, item_embeddings, device='cpu'):
        self.model = model
        self.item_embeddings = item_embeddings
        self.device = device

    def evaluate_all(self, recommendations, ground_truth, item_popularity,
                     user_history=None, item_categories=None):
        """
        Compute all beyond-accuracy metrics.

        Args:
            recommendations: Dict {user_idx: [recommended_items]}
            ground_truth: Dict {user_idx: [relevant_items]}
            item_popularity: Dict {item_idx: popularity_score}
            user_history: Dict {user_idx: [historical_items]}
            item_categories: Dict {item_idx: category}

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Convert to list format for some metrics
        rec_lists = [recommendations[u] for u in recommendations]
        num_items = self.item_embeddings.size(0)

        # Diversity metrics
        metrics['intra_list_diversity'] = DiversityMetrics.intra_list_diversity(
            rec_lists, self.item_embeddings
        )
        metrics['coverage'] = DiversityMetrics.coverage(rec_lists, num_items)

        if item_categories:
            metrics['category_diversity'] = DiversityMetrics.category_diversity(
                rec_lists, item_categories
            )

        # Compute item exposure for Gini
        item_counts = np.zeros(num_items)
        for rec_list in rec_lists:
            for item_idx in rec_list:
                item_counts[item_idx] += 1
        metrics['gini_coefficient'] = DiversityMetrics.gini_coefficient(item_counts)

        # Novelty metrics
        metrics['novelty'] = NoveltyMetrics.novelty_score(rec_lists, item_popularity)

        if user_history:
            metrics['serendipity'] = NoveltyMetrics.serendipity_score(
                recommendations, user_history, self.item_embeddings
            )

        # Get most popular items for unexpectedness
        popular_items = set(sorted(item_popularity.keys(),
                                   key=lambda x: item_popularity[x],
                                   reverse=True)[:100])
        metrics['unexpectedness'] = NoveltyMetrics.unexpectedness(
            rec_lists, popular_items
        )

        # Fairness metrics
        # User fairness (based on NDCG per user)
        user_ndcg = {}
        for user_idx in recommendations:
            if user_idx in ground_truth:
                rec_list = recommendations[user_idx]
                true_items = set(ground_truth[user_idx])

                # Compute NDCG@10
                dcg = 0.0
                idcg = 0.0
                for i, item in enumerate(rec_list[:10], 1):
                    if item in true_items:
                        dcg += 1.0 / np.log2(i + 1)
                for i in range(1, min(len(true_items), 10) + 1):
                    idcg += 1.0 / np.log2(i + 1)

                user_ndcg[user_idx] = dcg / idcg if idcg > 0 else 0.0

        metrics['user_fairness'] = FairnessMetrics.user_fairness(user_ndcg)

        # Provider fairness
        metrics['provider_fairness'] = FairnessMetrics.provider_fairness(
            rec_lists, num_items
        )

        return metrics

    def print_metrics(self, metrics):
        """
        Pretty print metrics.

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 80)
        print("BEYOND-ACCURACY METRICS")
        print("=" * 80)

        print("\n📊 DIVERSITY METRICS:")
        print(f"  • Intra-List Diversity:  {metrics.get('intra_list_diversity', 0):.4f}")
        print(f"  • Catalog Coverage:      {metrics.get('coverage', 0):.4f}")
        print(f"  • Category Diversity:    {metrics.get('category_diversity', 0):.4f}")
        print(f"  • Gini Coefficient:      {metrics.get('gini_coefficient', 0):.4f}")

        print("\n✨ NOVELTY & SERENDIPITY:")
        print(f"  • Novelty Score:         {metrics.get('novelty', 0):.4f}")
        print(f"  • Serendipity Score:     {metrics.get('serendipity', 0):.4f}")
        print(f"  • Unexpectedness:        {metrics.get('unexpectedness', 0):.4f}")

        print("\n⚖️  FAIRNESS METRICS:")
        if 'user_fairness' in metrics:
            uf = metrics['user_fairness']
            print(f"  • User Fairness (mean):  {uf['mean']:.4f}")
            print(f"  • User Fairness (std):   {uf['std']:.4f}")
            print(f"  • User Gini:             {uf['gini']:.4f}")

        if 'provider_fairness' in metrics:
            pf = metrics['provider_fairness']
            print(f"  • Provider Exposure (mean): {pf['exposure_mean']:.4f}")
            print(f"  • Provider Exposure (std):  {pf['exposure_std']:.4f}")
            print(f"  • Provider Gini:            {pf['exposure_gini']:.4f}")
            print(f"  • Items Never Recommended:  {pf['items_never_recommended']}")

        print("=" * 80 + "\n")


def compute_all_metrics(model, recommendations, ground_truth, item_embeddings,
                        item_popularity, user_history=None, item_categories=None):
    """
    Convenience function to compute all advanced metrics.

    Args:
        model: Recommendation model
        recommendations: Dict {user_idx: [recommended_items]}
        ground_truth: Dict {user_idx: [relevant_items]}
        item_embeddings: Item embedding matrix
        item_popularity: Dict {item_idx: popularity_score}
        user_history: Optional user history
        item_categories: Optional item categories

    Returns:
        Dictionary of all metrics
    """
    evaluator = BeyondAccuracyEvaluator(model, item_embeddings)
    metrics = evaluator.evaluate_all(
        recommendations, ground_truth, item_popularity,
        user_history, item_categories
    )
    return metrics

"""Explainability and interpretability tools for UACT-GNN recommendations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class AttentionVisualizer:
    """
    Visualize attention weights from Graph Transformer and Temporal Attention layers.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.attention_weights = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def get_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attention_weights'):
                    self.attention_weights[name] = module.attention_weights.detach().cpu()
            return hook

        # Register hooks for all attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'transformer' in name.lower():
                module.register_forward_hook(get_attention_hook(name))

    def extract_attention(self, user_idx, item_idx):
        """
        Extract attention weights for a user-item pair.

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Dictionary of attention weights by layer
        """
        self.attention_weights = {}
        self.model.eval()

        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long, device=self.device)
            _ = self.model.predict(user_tensor, item_tensor)

        return self.attention_weights

    def visualize_attention_heatmap(self, attention_weights, save_path=None):
        """
        Visualize attention weights as heatmap.

        Args:
            attention_weights: Attention weight tensor
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights.numpy(), cmap='viridis', annot=False)
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def get_top_k_attended_nodes(self, user_idx, k=10):
        """
        Get top-k nodes that the user attends to most.

        Args:
            user_idx: User index
            k: Number of top nodes to return

        Returns:
            List of (node_idx, attention_score) tuples
        """
        attention = self.extract_attention(user_idx, None)

        if not attention:
            return []

        # Average attention across all layers
        avg_attention = torch.mean(torch.stack([v for v in attention.values()]), dim=0)

        # Get top-k
        topk_values, topk_indices = torch.topk(avg_attention.view(-1), k=k)

        return list(zip(topk_indices.tolist(), topk_values.tolist()))


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for recommendations using gradient-based methods.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def compute_integrated_gradients(self, user_idx, item_idx, baseline=None, steps=50):
        """
        Compute Integrated Gradients for feature importance.

        Args:
            user_idx: User index
            item_idx: Item index
            baseline: Baseline input (default: zeros)
            steps: Number of integration steps

        Returns:
            Feature importance scores
        """
        self.model.eval()

        # Get user and item embeddings
        with torch.no_grad():
            _, user_emb, item_emb = self.model.forward(
                self.model.edge_index,
                self.model.edge_timestamps,
                self.model.time_indices
            )

        user_embedding = user_emb[user_idx].detach()
        item_embedding = item_emb[item_idx].detach()

        if baseline is None:
            baseline_user = torch.zeros_like(user_embedding)
            baseline_item = torch.zeros_like(item_embedding)
        else:
            baseline_user, baseline_item = baseline

        # Create interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)

        gradients_user = []
        gradients_item = []

        for alpha in alphas:
            # Interpolate
            interp_user = baseline_user + alpha * (user_embedding - baseline_user)
            interp_item = baseline_item + alpha * (item_embedding - baseline_item)

            interp_user.requires_grad = True
            interp_item.requires_grad = True

            # Compute score
            score = torch.sum(interp_user * interp_item)

            # Compute gradients
            score.backward()

            gradients_user.append(interp_user.grad.detach())
            gradients_item.append(interp_item.grad.detach())

        # Average gradients
        avg_grad_user = torch.mean(torch.stack(gradients_user), dim=0)
        avg_grad_item = torch.mean(torch.stack(gradients_item), dim=0)

        # Compute integrated gradients
        ig_user = (user_embedding - baseline_user) * avg_grad_user
        ig_item = (item_embedding - baseline_item) * avg_grad_item

        return {
            'user_importance': ig_user.cpu().numpy(),
            'item_importance': ig_item.cpu().numpy()
        }

    def compute_gradient_saliency(self, user_idx, item_idx):
        """
        Compute gradient-based saliency for a user-item pair.

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Saliency maps for user and item
        """
        self.model.eval()

        # Enable gradients for embeddings
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long, device=self.device)

        # Get embeddings with gradients
        _, user_emb, item_emb = self.model.forward(
            self.model.edge_index,
            self.model.edge_timestamps,
            self.model.time_indices
        )

        user_embedding = user_emb[user_idx]
        item_embedding = item_emb[item_idx]

        user_embedding.requires_grad = True
        item_embedding.requires_grad = True

        # Compute score
        score = torch.sum(user_embedding * item_embedding)
        score.backward()

        # Get gradients
        user_saliency = torch.abs(user_embedding.grad).detach().cpu().numpy()
        item_saliency = torch.abs(item_embedding.grad).detach().cpu().numpy()

        return {
            'user_saliency': user_saliency,
            'item_saliency': item_saliency
        }


class CausalPathTracer:
    """
    Trace causal paths in the recommendation graph to explain why items are recommended.
    """

    def __init__(self, model, edge_index, edge_timestamps, device='cpu'):
        self.model = model
        self.edge_index = edge_index.cpu()
        self.edge_timestamps = edge_timestamps.cpu()
        self.device = device
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency list for efficient path finding."""
        self.adjacency = {}

        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            timestamp = self.edge_timestamps[i].item()

            if src not in self.adjacency:
                self.adjacency[src] = []
            self.adjacency[src].append((dst, timestamp))

    def find_causal_paths(self, user_idx, item_idx, max_length=5, max_paths=10):
        """
        Find causal paths from user to item through the graph.

        Args:
            user_idx: User index
            item_idx: Item index (offset by num_users in graph)
            max_length: Maximum path length
            max_paths: Maximum number of paths to return

        Returns:
            List of causal paths
        """
        paths = []
        visited = set()

        def dfs(current, target, path, timestamp):
            if len(paths) >= max_paths:
                return

            if current == target:
                paths.append(list(path))
                return

            if len(path) >= max_length:
                return

            if current in self.adjacency:
                for neighbor, edge_time in self.adjacency[current]:
                    # Ensure causal ordering (later timestamps only)
                    if neighbor not in visited and edge_time >= timestamp:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, target, path, edge_time)
                        path.pop()
                        visited.remove(neighbor)

        visited.add(user_idx)
        dfs(user_idx, item_idx, [user_idx], 0)

        return paths

    def compute_path_importance(self, paths, user_idx, item_idx):
        """
        Compute importance scores for causal paths.

        Args:
            paths: List of paths
            user_idx: User index
            item_idx: Item index

        Returns:
            List of (path, importance_score) tuples
        """
        self.model.eval()

        path_scores = []

        with torch.no_grad():
            # Get embeddings
            _, user_emb, item_emb = self.model.forward(
                self.model.edge_index,
                self.model.edge_timestamps,
                self.model.time_indices
            )

            for path in paths:
                # Compute path embedding as average of intermediate nodes
                path_embeddings = []
                for node in path[1:-1]:  # Exclude start and end
                    if node < user_emb.size(0):
                        path_embeddings.append(user_emb[node])
                    else:
                        path_embeddings.append(item_emb[node - user_emb.size(0)])

                if path_embeddings:
                    avg_path_emb = torch.mean(torch.stack(path_embeddings), dim=0)

                    # Score based on alignment with user and item
                    user_alignment = F.cosine_similarity(
                        user_emb[user_idx].unsqueeze(0),
                        avg_path_emb.unsqueeze(0)
                    )
                    item_alignment = F.cosine_similarity(
                        item_emb[item_idx].unsqueeze(0),
                        avg_path_emb.unsqueeze(0)
                    )

                    score = (user_alignment + item_alignment).item() / 2
                else:
                    score = 0.0

                path_scores.append((path, score))

        # Sort by importance
        path_scores.sort(key=lambda x: x[1], reverse=True)

        return path_scores


class CounterfactualExplainer:
    """
    Generate counterfactual explanations: "If X were different, the recommendation would change."
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def generate_counterfactual(self, user_idx, item_idx, num_perturbations=10):
        """
        Generate counterfactual explanations by perturbing embeddings.

        Args:
            user_idx: User index
            item_idx: Item index
            num_perturbations: Number of counterfactuals to generate

        Returns:
            List of counterfactual explanations
        """
        self.model.eval()

        with torch.no_grad():
            _, user_emb, item_emb = self.model.forward(
                self.model.edge_index,
                self.model.edge_timestamps,
                self.model.time_indices
            )

        user_embedding = user_emb[user_idx].clone()
        item_embedding = item_emb[item_idx].clone()

        # Original score
        original_score = torch.sum(user_embedding * item_embedding).item()

        counterfactuals = []

        for _ in range(num_perturbations):
            # Perturb user embedding
            perturbation = torch.randn_like(user_embedding) * 0.1
            perturbed_user = user_embedding + perturbation

            # Compute new score
            new_score = torch.sum(perturbed_user * item_embedding).item()

            # Check if recommendation would change
            if (new_score > 0) != (original_score > 0):
                counterfactuals.append({
                    'perturbation': perturbation.cpu().numpy(),
                    'original_score': original_score,
                    'new_score': new_score,
                    'change': 'positive' if new_score > 0 else 'negative'
                })

        return counterfactuals

    def explain_feature_contribution(self, user_idx, item_idx, top_k=10):
        """
        Explain which features contribute most to the recommendation.

        Args:
            user_idx: User index
            item_idx: Item index
            top_k: Number of top features to return

        Returns:
            Top contributing features
        """
        self.model.eval()

        with torch.no_grad():
            _, user_emb, item_emb = self.model.forward(
                self.model.edge_index,
                self.model.edge_timestamps,
                self.model.time_indices
            )

        user_embedding = user_emb[user_idx]
        item_embedding = item_emb[item_idx]

        # Compute element-wise contribution
        contributions = user_embedding * item_embedding

        # Get top-k
        topk_values, topk_indices = torch.topk(torch.abs(contributions), k=top_k)

        explanations = []
        for idx, val in zip(topk_indices.tolist(), topk_values.tolist()):
            explanations.append({
                'dimension': idx,
                'contribution': contributions[idx].item(),
                'magnitude': val
            })

        return explanations


class ExplanationGenerator:
    """
    Generate human-readable explanations for recommendations.
    """

    def __init__(self, model, user_id_map, item_id_map, device='cpu'):
        self.model = model
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.device = device

        self.attention_viz = AttentionVisualizer(model, device)
        self.feature_analyzer = FeatureImportanceAnalyzer(model, device)
        self.counterfactual = CounterfactualExplainer(model, device)

    def explain_recommendation(self, user_id, item_id, top_k_features=5):
        """
        Generate comprehensive explanation for a recommendation.

        Args:
            user_id: Original user ID
            item_id: Original item ID
            top_k_features: Number of top features to explain

        Returns:
            Dictionary with explanation components
        """
        user_idx = self.user_id_map.get(user_id)
        item_idx = self.item_id_map.get(item_id)

        if user_idx is None or item_idx is None:
            return {"error": "User or item not found"}

        # Get prediction score
        with torch.no_grad():
            score = self.model.predict(
                torch.tensor([user_idx], device=self.device),
                torch.tensor([item_idx], device=self.device)
            ).item()

        # Feature importance
        importance = self.feature_analyzer.compute_integrated_gradients(user_idx, item_idx)

        # Feature contributions
        contributions = self.counterfactual.explain_feature_contribution(user_idx, item_idx, top_k_features)

        # Build explanation
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'score': score,
            'recommendation': 'positive' if score > 0 else 'negative',
            'top_contributing_features': contributions,
            'feature_importance': {
                'user': importance['user_importance'][:top_k_features].tolist(),
                'item': importance['item_importance'][:top_k_features].tolist()
            }
        }

        return explanation

    def generate_text_explanation(self, explanation):
        """
        Generate human-readable text explanation.

        Args:
            explanation: Explanation dictionary

        Returns:
            Text explanation string
        """
        score = explanation['score']
        rec_type = explanation['recommendation']

        text = f"Recommendation for User {explanation['user_id']} and Item {explanation['item_id']}:\n"
        text += f"Score: {score:.4f} ({rec_type})\n\n"

        text += "Top Contributing Factors:\n"
        for i, contrib in enumerate(explanation['top_contributing_features'], 1):
            text += f"{i}. Feature dimension {contrib['dimension']}: "
            text += f"{'positive' if contrib['contribution'] > 0 else 'negative'} "
            text += f"contribution of {abs(contrib['contribution']):.4f}\n"

        return text

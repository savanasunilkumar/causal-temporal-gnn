"""Causal discovery techniques for recommendation systems."""

import numpy as np
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# For causal discovery
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    print("Warning: causal-learn not installed. Causal discovery will be disabled.")
    print("Install with: pip install causal-learn")
    CAUSAL_LEARN_AVAILABLE = False


class CausalGraphConstructor:
    """Implements causal discovery techniques for recommendation systems."""
    
    def __init__(self, config):
        self.config = config
        self.causal_method = config.causal_method
        self.significance_level = config.significance_level
        self.max_lag = config.max_lag
        self.min_causal_strength = config.min_causal_strength
        
    def compute_granger_causality(self, time_series, max_lag=None):
        """
        Compute Granger causality between time series.
        
        Args:
            time_series: Dictionary of time series data {node_id: [values]}
            max_lag: Maximum time lag to consider
            
        Returns:
            Causal adjacency matrix
        """
        if not CAUSAL_LEARN_AVAILABLE:
            print("Warning: causal-learn not available. Skipping Granger causality.")
            return np.zeros((len(time_series), len(time_series)))

        if max_lag is None:
            max_lag = self.max_lag
            
        nodes = list(time_series.keys())
        n_nodes = len(nodes)
        causal_matrix = np.zeros((n_nodes, n_nodes))
        
        # Standardize time series
        scaler = StandardScaler()
        standardized_series = {}
        for node in nodes:
            if len(time_series[node]) > max_lag:
                standardized_series[node] = scaler.fit_transform(
                    np.array(time_series[node]).reshape(-1, 1)
                ).flatten()
        
        # Compute Granger causality for each pair
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i == j or source not in standardized_series or target not in standardized_series:
                    continue
                
                source_series = standardized_series[source]
                target_series = standardized_series[target]
                
                # Skip if not enough data points
                if len(source_series) <= max_lag or len(target_series) <= max_lag:
                    continue
                
                # Prepare lagged data
                X = []
                y = []
                
                for t in range(max_lag, len(target_series)):
                    # Target value at time t
                    y.append(target_series[t])
                    
                    # Lagged values of target and source
                    row = []
                    for lag in range(1, max_lag + 1):
                        row.append(target_series[t - lag])  # Target's own past
                        row.append(source_series[t - lag])  # Source's past
                    X.append(row)
                
                X = np.array(X)
                y = np.array(y)
                
                if len(X) == 0:
                    continue
                
                # Model 1: Using only target's past
                X_target_only = X[:, ::2]  # Every other column (target's past)
                if X_target_only.shape[1] > 0:
                    model_target_only = LinearRegression().fit(X_target_only, y)
                    mse_target_only = np.mean((model_target_only.predict(X_target_only) - y) ** 2)
                else:
                    mse_target_only = np.mean((y - np.mean(y))**2)

                # Model 2: Using both target's and source's past
                model_both = LinearRegression().fit(X, y)
                mse_both = np.mean((model_both.predict(X) - y) ** 2)
                
                # F-statistic for Granger causality
                if mse_target_only > 1e-6:  # Avoid division by zero
                    f_stat = ((mse_target_only - mse_both) / max_lag) / (mse_both / (len(y) - 2 * max_lag))
                    
                    # Convert to p-value (simplified F-distribution approximation)
                    p_value = 1 / (1 + f_stat) if f_stat > 0 else 1.0
                    
                    if p_value < self.significance_level:
                        causal_strength = min(1.0, f_stat / 10)  # Normalize to [0, 1]
                        if causal_strength > self.min_causal_strength:
                            causal_matrix[i, j] = causal_strength
        
        return causal_matrix
    
    def compute_pc_algorithm(self, data):
        """
        Compute causal graph using the PC algorithm.
        
        Args:
            data: Node features matrix [n_nodes, n_features]
            
        Returns:
            Causal adjacency matrix
        """
        if not CAUSAL_LEARN_AVAILABLE:
            print("Warning: causal-learn not available. Skipping PC algorithm.")
            return np.zeros((data.shape[0], data.shape[0]))

        try:
            # Apply PC algorithm from causal-learn
            cg = pc(data, alpha=self.significance_level, indep_test="fisherz")
            
            # Convert to adjacency matrix
            n_nodes = data.shape[0]
            causal_matrix = np.zeros((n_nodes, n_nodes))
            
            # Extract edges from the causal graph
            G = cg.G
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if G.has_edge(i, j):
                        # Estimate edge strength
                        correlation, _ = pearsonr(data[i], data[j])
                        if not np.isnan(correlation):
                            causal_matrix[i, j] = abs(correlation)
            
            return causal_matrix
        except Exception as e:
            print(f"Error in PC algorithm: {e}. Falling back to correlation-based causality.")
            # Fallback to simple correlation
            n_nodes = data.shape[0]
            causal_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        correlation, _ = pearsonr(data[i], data[j])
                        if not np.isnan(correlation) and abs(correlation) > self.min_causal_strength:
                            causal_matrix[i, j] = abs(correlation)
            return causal_matrix

    def compute_hybrid_causal_graph(self, interaction_data, node_features, edge_timestamps):
        """
        Compute a hybrid causal graph combining temporal and feature-based causality.
        
        Args:
            interaction_data: User-item interaction data
            node_features: Node feature matrix
            edge_timestamps: Edge timestamps
            
        Returns:
            Causal edge index and edge weights
        """
        # 1. Extract time series for each node
        time_series = self._extract_time_series(interaction_data, edge_timestamps)
        
        # 2. Compute Granger causality
        granger_matrix = self.compute_granger_causality(time_series)
        
        # 3. Compute PC algorithm causality
        pc_matrix = self.compute_pc_algorithm(node_features)
        
        # 4. Combine both methods
        combined_matrix = 0.6 * granger_matrix + 0.4 * pc_matrix
        
        # 5. Apply threshold to get final causal edges
        causal_edges = []
        edge_weights = []
        
        for i in range(combined_matrix.shape[0]):
            for j in range(combined_matrix.shape[1]):
                if i != j and combined_matrix[i, j] > self.min_causal_strength:
                    causal_edges.append((i, j))
                    edge_weights.append(combined_matrix[i, j])
        
        return causal_edges, edge_weights
    
    def _extract_time_series(self, interaction_data, edge_timestamps):
        """Extract time series data for each node from interactions."""
        node_time_series = {}
        time_step_interactions = defaultdict(list)
        
        # Normalize timestamps to time steps
        min_time = edge_timestamps.min()
        max_time = edge_timestamps.max()
        time_range = max_time - min_time
        if time_range == 0:
            time_range = 1
        
        time_step_size = time_range / self.config.time_steps
        
        # Group interactions by time step
        for i, timestamp in enumerate(edge_timestamps):
            time_step = int((timestamp - min_time) / time_step_size)
            time_step = min(time_step, self.config.time_steps - 1)
            
            user_idx = interaction_data[0, i]
            item_idx = interaction_data[1, i]
            
            time_step_interactions[time_step].append((user_idx, item_idx))
        
        # Create time series for each node
        n_nodes = interaction_data.max() + 1
        for node_idx in range(n_nodes):
            time_series = []
            
            for time_step in sorted(time_step_interactions.keys()):
                count = sum(
                    1 for user, item in time_step_interactions[time_step]
                    if user == node_idx or item == node_idx
                )
                time_series.append(count)
            
            node_time_series[node_idx] = time_series
        
        return node_time_series


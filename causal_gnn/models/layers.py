"""Custom PyTorch Geometric layers for UACT-GNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


class CausalGNNLayer(MessagePassing):
    """
    Causal Graph Neural Network Layer with temporal edge weights.
    
    Implements message passing on causal graphs where edges represent
    causal relationships with associated strength/weights.
    """
    
    def __init__(self, in_channels, out_channels, dropout=0.1, use_edge_weight=True):
        super().__init__(aggr='add')  # "Add" aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_edge_weight = use_edge_weight
        
        # Linear transformation for source nodes
        self.lin_src = nn.Linear(in_channels, out_channels)
        
        # Linear transformation for messages
        self.lin_msg = nn.Linear(in_channels, out_channels)
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.lin_src.reset_parameters()
        self.lin_msg.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform node features
        x = self.lin_src(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        # Add bias
        out = out + self.bias
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out
    
    def message(self, x_j, edge_weight):
        """
        Construct messages from neighbors.
        
        Args:
            x_j: Neighbor node features [num_edges, in_channels]
            edge_weight: Edge weights [num_edges]
            
        Returns:
            Messages [num_edges, out_channels]
        """
        # Transform messages
        msg = self.lin_msg(x_j)
        
        # Apply edge weights if provided
        if self.use_edge_weight and edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        
        return msg
    
    def update(self, aggr_out):
        """Update node features after aggregation."""
        return aggr_out


class TemporalAttentionLayer(MessagePassing):
    """
    Temporal attention layer for time-aware message passing.
    
    Uses attention mechanism to weight messages based on temporal proximity.
    """
    
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Ensure out_channels is divisible by heads
        assert out_channels % heads == 0
        self.head_dim = out_channels // heads
        
        # Linear transformations for Q, K, V
        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)
        
        # Output projection
        self.lin_out = nn.Linear(out_channels, out_channels)
        
        # Temporal encoding
        self.temporal_encoding = nn.Linear(1, heads)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_out.reset_parameters()
        self.temporal_encoding.reset_parameters()
    
    def forward(self, x, edge_index, edge_time=None):
        """
        Forward pass with temporal attention.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_time: Edge timestamps [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Compute Q, K, V
        query = self.lin_query(x).view(-1, self.heads, self.head_dim)
        key = self.lin_key(x).view(-1, self.heads, self.head_dim)
        value = self.lin_value(x).view(-1, self.heads, self.head_dim)
        
        # Propagate with attention
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_time=edge_time
        )
        
        # Reshape and project output
        out = out.view(-1, self.out_channels)
        out = self.lin_out(out)
        
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out
    
    def message(self, query_i, key_j, value_j, edge_time, index, ptr, size_i):
        """
        Compute attention-weighted messages.
        
        Args:
            query_i: Query from target nodes [num_edges, heads, head_dim]
            key_j: Key from source nodes [num_edges, heads, head_dim]
            value_j: Value from source nodes [num_edges, heads, head_dim]
            edge_time: Edge timestamps [num_edges]
            index: Target node indices
            ptr: CSR row pointers
            size_i: Number of target nodes
            
        Returns:
            Attention-weighted messages [num_edges, heads, head_dim]
        """
        # Compute attention scores
        attn = (query_i * key_j).sum(dim=-1)  # [num_edges, heads]
        
        # Add temporal encoding if timestamps provided
        if edge_time is not None:
            temporal_weights = self.temporal_encoding(edge_time.view(-1, 1))  # [num_edges, heads]
            attn = attn + temporal_weights
        
        # Apply softmax
        attn = softmax(attn, index, ptr, size_i)
        
        # Apply dropout to attention weights
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Weight values by attention
        return value_j * attn.unsqueeze(-1)


class SparseGCNLayer(MessagePassing):
    """
    Sparse Graph Convolutional Layer optimized for large graphs.
    
    Uses sparse operations and normalization for efficient computation.
    """
    
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 add_self_loops=True, normalize=True, bias=True):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._cached_edge_index = None
        self._cached_adj_t = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        self._cached_edge_index = None
        self._cached_adj_t = None
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with sparse operations.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform features
        x = self.lin(x)
        
        # Cache normalized edge_index if requested
        if self.cached and self._cached_edge_index is not None:
            edge_index, edge_weight = self._cached_edge_index
        else:
            # Add self-loops
            if self.add_self_loops:
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight, num_nodes=x.size(0)
                )
            
            # Normalize
            if self.normalize:
                row, col = edge_index
                deg = degree(col, x.size(0), dtype=x.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                
                if edge_weight is None:
                    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                else:
                    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
            
            if self.cached:
                self._cached_edge_index = (edge_index, edge_weight)
        
        # Propagate
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_j, edge_weight):
        """Apply edge weights to messages."""
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GraphSAGELayer(MessagePassing):
    """
    GraphSAGE layer with sampling support for large graphs.
    
    Efficient for neighbor sampling and mini-batch training.
    """
    
    def __init__(self, in_channels, out_channels, normalize=True, 
                 bias=True, aggr='mean'):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
    
    def forward(self, x, edge_index, size=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels] or tuple for bipartite
            edge_index: Edge indices [2, num_edges]
            size: Graph size (for bipartite graphs)
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        if isinstance(x, torch.Tensor):
            x = (x, x)
        
        # Propagate and aggregate neighbor features
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)
        
        # Add transformed self features
        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_r(x_r)
        
        # Normalize
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def message(self, x_j):
        """Identity message function."""
        return x_j


class GraphTransformerLayer(MessagePassing):
    """
    Graph Transformer Layer with full attention mechanism.

    More expressive than standard GNNs, uses attention over all graph connections.
    """

    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1,
                 edge_dim=None, bias=True):
        super().__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        assert out_channels % heads == 0
        self.head_dim = out_channels // heads

        # Multi-head attention components
        self.lin_query = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, out_channels, bias=bias)

        # Edge feature encoding (if provided)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads, bias=False)
        else:
            self.lin_edge = None

        # Output projection with residual
        self.lin_out = nn.Linear(out_channels, out_channels, bias=bias)
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=bias)

        # Layer normalization
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_out.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        for layer in self.ffn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with Transformer-style attention.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Compute Q, K, V
        query = self.lin_query(x).view(-1, self.heads, self.head_dim)
        key = self.lin_key(x).view(-1, self.heads, self.head_dim)
        value = self.lin_value(x).view(-1, self.heads, self.head_dim)

        # Message passing with attention
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr
        )

        # Reshape and project
        out = out.view(-1, self.out_channels)
        out = self.lin_out(out)

        # Residual connection and layer norm
        out = out + self.lin_skip(x)
        out = self.norm1(out)

        # Feed-forward network with residual
        out_ffn = self.ffn(out)
        out = out + out_ffn
        out = self.norm2(out)

        return out

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        """
        Compute attention-weighted messages.
        """
        # Compute attention scores (scaled dot-product)
        attn = (query_i * key_j).sum(dim=-1) / (self.head_dim ** 0.5)

        # Add edge features to attention if provided
        if edge_attr is not None and self.lin_edge is not None:
            edge_attn = self.lin_edge(edge_attr)
            attn = attn + edge_attn

        # Apply softmax
        attn = softmax(attn, index, ptr, size_i)

        # Apply dropout
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Weight values by attention
        return value_j * attn.unsqueeze(-1)


class HeteroGNNLayer(nn.Module):
    """
    Heterogeneous GNN Layer for bipartite user-item graphs.

    Handles different node types (users vs items) with separate transformations.
    """

    def __init__(self, in_channels_user, in_channels_item, out_channels,
                 dropout=0.1, aggr='mean'):
        super().__init__()

        self.in_channels_user = in_channels_user
        self.in_channels_item = in_channels_item
        self.out_channels = out_channels
        self.dropout = dropout

        # Separate transformations for user and item nodes
        self.user_transform = nn.Linear(in_channels_user, out_channels)
        self.item_transform = nn.Linear(in_channels_item, out_channels)

        # Message transformations for different edge types
        self.user_to_item_msg = nn.Linear(in_channels_user, out_channels)
        self.item_to_user_msg = nn.Linear(in_channels_item, out_channels)

        # Attention weights for message aggregation
        self.attn_user = nn.Linear(out_channels * 2, 1)
        self.attn_item = nn.Linear(out_channels * 2, 1)

        self.aggr = aggr

    def forward(self, user_features, item_features, edge_index):
        """
        Forward pass for heterogeneous graph.

        Args:
            user_features: User node features [num_users, in_channels_user]
            item_features: Item node features [num_items, in_channels_item]
            edge_index: Edge indices [2, num_edges] (user indices in row 0)

        Returns:
            Updated user and item features
        """
        # Transform node features
        user_emb = self.user_transform(user_features)
        item_emb = self.item_transform(item_features)

        # User -> Item messages
        user_indices = edge_index[0]
        item_indices = edge_index[1]

        user_msgs = self.user_to_item_msg(user_features[user_indices])
        item_msgs_agg = torch.zeros(item_features.size(0), self.out_channels,
                                     device=item_features.device)
        item_msgs_agg = item_msgs_agg.index_add_(0, item_indices, user_msgs)

        # Item -> User messages
        item_msgs = self.item_to_user_msg(item_features[item_indices])
        user_msgs_agg = torch.zeros(user_features.size(0), self.out_channels,
                                     device=user_features.device)
        user_msgs_agg = user_msgs_agg.index_add_(0, user_indices, item_msgs)

        # Update with residual connections
        user_emb_new = user_emb + user_msgs_agg
        item_emb_new = item_emb + item_msgs_agg

        # Apply dropout
        if self.dropout > 0:
            user_emb_new = F.dropout(user_emb_new, p=self.dropout, training=self.training)
            item_emb_new = F.dropout(item_emb_new, p=self.dropout, training=self.training)

        return user_emb_new, item_emb_new


class ResidualGNNBlock(nn.Module):
    """
    Residual GNN Block with skip connections for deeper models.
    """

    def __init__(self, channels, layer_type='gcn', num_layers=2, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if layer_type == 'gcn':
                layer = SparseGCNLayer(channels, channels)
            elif layer_type == 'sage':
                layer = GraphSAGELayer(channels, channels)
            elif layer_type == 'causal':
                layer = CausalGNNLayer(channels, channels, dropout=dropout)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(channels))

        self.dropout = dropout
        self.activation = nn.GELU()

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with residual connections.
        """
        identity = x

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_weight)
            x = norm(x)
            x = self.activation(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Residual connection
        x = x + identity

        return x


def create_gnn_layer(layer_type, in_channels, out_channels, **kwargs):
    """
    Factory function to create GNN layers.

    Args:
        layer_type: Type of layer ('causal', 'temporal', 'gcn', 'sage', 'transformer', 'hetero')
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        **kwargs: Additional layer-specific arguments

    Returns:
        GNN layer instance
    """
    if layer_type == 'causal':
        return CausalGNNLayer(in_channels, out_channels, **kwargs)
    elif layer_type == 'temporal':
        return TemporalAttentionLayer(in_channels, out_channels, **kwargs)
    elif layer_type == 'gcn':
        return SparseGCNLayer(in_channels, out_channels, **kwargs)
    elif layer_type == 'sage':
        return GraphSAGELayer(in_channels, out_channels, **kwargs)
    elif layer_type == 'transformer':
        return GraphTransformerLayer(in_channels, out_channels, **kwargs)
    elif layer_type == 'residual':
        return ResidualGNNBlock(in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


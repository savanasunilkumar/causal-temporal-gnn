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


def create_gnn_layer(layer_type, in_channels, out_channels, **kwargs):
    """
    Factory function to create GNN layers.
    
    Args:
        layer_type: Type of layer ('causal', 'temporal', 'gcn', 'sage')
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
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


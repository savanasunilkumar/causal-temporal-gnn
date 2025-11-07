"""Session-based recommendation using RNNs and Transformers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class GRU4Rec(nn.Module):
    """
    GRU-based session recommendation (GRU4Rec).
    """

    def __init__(self, num_items, embedding_dim=100, hidden_dim=100,
                 num_layers=1, dropout=0.1):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Item embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # GRU layers
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_items)

        self.dropout = nn.Dropout(dropout)

    def forward(self, session_items, lengths=None):
        """
        Forward pass for session.

        Args:
            session_items: Item IDs in session [batch_size, seq_len]
            lengths: Actual lengths of sequences

        Returns:
            Predictions for next item
        """
        # Embed items
        embedded = self.item_embedding(session_items)  # [batch, seq_len, emb_dim]
        embedded = self.dropout(embedded)

        # Pack sequences if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )

        # GRU forward
        output, hidden = self.gru(embedded)

        # Unpack if needed
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Get last hidden state
        if lengths is not None:
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2))
            last_hidden = output.gather(1, idx).squeeze(1)
        else:
            last_hidden = output[:, -1, :]

        # Predict next item
        logits = self.output_layer(last_hidden)

        return logits

    def recommend(self, session_items, top_k=10):
        """
        Generate top-k recommendations for session.

        Args:
            session_items: Item IDs in session
            top_k: Number of recommendations

        Returns:
            Top-k item IDs and scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(session_items)
            scores = F.softmax(logits, dim=-1)
            top_scores, top_items = torch.topk(scores, k=top_k, dim=-1)

        return top_items, top_scores


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec).
    """

    def __init__(self, num_items, embedding_dim=64, num_blocks=2,
                 num_heads=2, dropout=0.1, max_len=200):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # Item and positional embeddings
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)

        # Transformer blocks
        self.attention_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, num_items)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, session_items):
        """
        Forward pass.

        Args:
            session_items: [batch_size, seq_len]

        Returns:
            Logits for next item
        """
        batch_size, seq_len = session_items.size()

        # Create attention mask (causal)
        attention_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=session_items.device),
            diagonal=1
        ).bool()

        # Item embeddings
        item_emb = self.item_embedding(session_items)

        # Positional embeddings
        positions = torch.arange(seq_len, device=session_items.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        # Combine embeddings
        seq_emb = item_emb + pos_emb
        seq_emb = self.dropout(seq_emb)

        # Apply transformer blocks
        for block in self.attention_blocks:
            seq_emb = block(seq_emb, attention_mask)

        # Output (use last position)
        seq_emb = self.layer_norm(seq_emb)
        logits = self.output_layer(seq_emb[:, -1, :])

        return logits


class TransformerBlock(nn.Module):
    """Single transformer block for SASRec."""

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, attention_mask=None):
        """
        Forward pass.

        Args:
            x: [batch, seq_len, dim]
            attention_mask: Causal mask

        Returns:
            Transformed sequence
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.layer_norm1(x + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        return x


class NARM(nn.Module):
    """
    Neural Attentive Recommendation Machine (NARM).
    Combines RNN with attention mechanism.
    """

    def __init__(self, num_items, embedding_dim=100, hidden_dim=100, dropout=0.1):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Item embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # GRU for global encoding
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # Attention
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.attention_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # Output
        self.output_layer = nn.Linear(hidden_dim, num_items)

        self.dropout = nn.Dropout(dropout)

    def forward(self, session_items):
        """
        Forward pass with attention.

        Args:
            session_items: [batch_size, seq_len]

        Returns:
            Logits for next item
        """
        # Embed items
        embedded = self.item_embedding(session_items)
        embedded = self.dropout(embedded)

        # GRU encoding
        gru_out, hidden = self.gru(embedded)

        # Attention over sequence
        # Query: last hidden state
        query = hidden[-1].unsqueeze(1)  # [batch, 1, hidden]

        # Keys: all hidden states
        keys = self.attention(gru_out)  # [batch, seq_len, hidden]

        # Attention scores
        scores = torch.bmm(query, keys.transpose(1, 2))  # [batch, 1, seq_len]
        attention_weights = F.softmax(scores, dim=-1)

        # Context vector
        context = torch.bmm(attention_weights, gru_out)  # [batch, 1, hidden]
        context = context.squeeze(1)

        # Combine context with last hidden
        combined = torch.cat([context, hidden[-1]], dim=-1)
        combined = self.attention_combine(combined)
        combined = torch.tanh(combined)

        # Output
        logits = self.output_layer(combined)

        return logits


class SessionGraph(nn.Module):
    """
    Session-based recommendation with graph neural networks.
    """

    def __init__(self, num_items, embedding_dim=100, hidden_dim=100, num_layers=1):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Item embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output
        self.output_layer = nn.Linear(hidden_dim * 2, num_items)

    def forward(self, session_items, adjacency_matrix):
        """
        Forward pass with session graph.

        Args:
            session_items: Item IDs in session
            adjacency_matrix: Adjacency matrix of session graph

        Returns:
            Predictions
        """
        # Embed items
        embedded = self.item_embedding(session_items)

        # Graph convolutions
        for conv in self.conv_layers:
            # Message passing
            messages = torch.matmul(adjacency_matrix, embedded)
            embedded = F.relu(conv(messages))

        # Global representation (mean pooling)
        global_rep = torch.mean(embedded, dim=1)

        # Local representation (last item)
        local_rep = embedded[:, -1, :]

        # Combine
        combined = torch.cat([global_rep, local_rep], dim=-1)

        # Output
        logits = self.output_layer(combined)

        return logits


def create_session_model(model_type='gru4rec', num_items=1000, **kwargs):
    """
    Factory function to create session-based models.

    Args:
        model_type: Type of model ('gru4rec', 'sasrec', 'narm', 'graph')
        num_items: Number of items
        **kwargs: Additional model arguments

    Returns:
        Session model instance
    """
    if model_type == 'gru4rec':
        return GRU4Rec(num_items, **kwargs)
    elif model_type == 'sasrec':
        return SASRec(num_items, **kwargs)
    elif model_type == 'narm':
        return NARM(num_items, **kwargs)
    elif model_type == 'graph':
        return SessionGraph(num_items, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

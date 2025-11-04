"""LightGCN baseline implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import scipy.sparse as sp


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
    
    Reference:
    He et al. "LightGCN: Simplifying and Powering Graph Convolution Network 
    for Recommendation" SIGIR 2020
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, learning_rate=0.001, reg_lambda=1e-4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Graph will be set during fit
        self.Graph = None
        
    def _create_adj_mat(self, train_data):
        """Create normalized adjacency matrix."""
        num_nodes = self.num_users + self.num_items
        
        # Create user-item interaction matrix
        user_np = train_data['user_idx'].values
        item_np = train_data['item_idx'].values + self.num_users  # Offset for bipartite graph
        
        # Create adjacency matrix
        ratings = np.ones_like(user_np, dtype=np.float32)
        
        # User-Item interactions
        adj_mat = sp.csr_matrix(
            (ratings, (user_np, item_np)),
            shape=(num_nodes, num_nodes)
        )
        
        # Make symmetric (bipartite graph)
        adj_mat = adj_mat + adj_mat.T
        
        # Normalize
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        return norm_adj
    
    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert scipy sparse matrix to torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def compute_graph_embeddings(self):
        """Compute embeddings using graph convolution."""
        # Get embeddings and move to graph device for sparse operations
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])
        
        # Store original device
        orig_device = all_embeddings.device
        graph_device = self.Graph.device
        
        # Move to graph device if needed
        if orig_device != graph_device:
            all_embeddings = all_embeddings.to(graph_device)
        
        embeddings_list = [all_embeddings]
        
        for layer in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.Graph, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Average across layers
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        
        # Move back to original device if needed
        if orig_device != graph_device:
            final_embeddings = final_embeddings.to(orig_device)
        
        users_emb, items_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        return users_emb, items_emb
    
    def forward(self, user_idx, item_idx):
        """Compute scores for user-item pairs."""
        users_emb, items_emb = self.compute_graph_embeddings()
        
        user_emb = users_emb[user_idx]
        item_emb = items_emb[item_idx]
        
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores
    
    def bpr_loss(self, user_idx, pos_item_idx, neg_item_idx):
        """Compute BPR loss."""
        pos_scores = self.forward(user_idx, pos_item_idx)
        neg_scores = self.forward(user_idx, neg_item_idx)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        
        # Regularization
        user_emb = self.user_embedding(user_idx)
        pos_item_emb = self.item_embedding(pos_item_idx)
        neg_item_emb = self.item_embedding(neg_item_idx)
        
        reg_loss = self.reg_lambda * (
            user_emb.norm(2).pow(2) + 
            pos_item_emb.norm(2).pow(2) + 
            neg_item_emb.norm(2).pow(2)
        ) / user_idx.size(0)
        
        return loss + reg_loss
    
    def fit(self, train_data, num_epochs=20, batch_size=256, device='cpu'):
        """Train the LightGCN model."""
        self.to(device)
        self.train()
        
        # Create graph
        print("Creating adjacency matrix...")
        adj_mat = self._create_adj_mat(train_data)
        # Keep graph on CPU for sparse operations (MPS doesn't support sparse matrix ops)
        graph_device = 'cpu' if device == 'mps' else device
        self.Graph = self._sparse_mx_to_torch_sparse_tensor(adj_mat).to(graph_device)
        
        # Create user-item interaction dict
        user_items = {}
        for _, row in train_data.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            if user_idx not in user_items:
                user_items[user_idx] = []
            user_items[user_idx].append(item_idx)
        
        all_items = set(range(self.num_items))
        
        print(f"Training LightGCN with {len(train_data)} interactions...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            users = list(user_items.keys())
            np.random.shuffle(users)
            
            pbar = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_start in pbar:
                batch_users = users[batch_start:batch_start + batch_size]
                
                user_idx_list = []
                pos_item_idx_list = []
                neg_item_idx_list = []
                
                for user in batch_users:
                    pos_items = user_items[user]
                    pos_item = np.random.choice(pos_items)
                    
                    neg_items = list(all_items - set(pos_items))
                    neg_item = np.random.choice(neg_items)
                    
                    user_idx_list.append(user)
                    pos_item_idx_list.append(pos_item)
                    neg_item_idx_list.append(neg_item)
                
                user_idx = torch.LongTensor(user_idx_list).to(device)
                pos_item_idx = torch.LongTensor(pos_item_idx_list).to(device)
                neg_item_idx = torch.LongTensor(neg_item_idx_list).to(device)
                
                self.optimizer.zero_grad()
                loss = self.bpr_loss(user_idx, pos_item_idx, neg_item_idx)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, user_idx, top_k=10, exclude_items=None, device='cpu'):
        """Predict top-k items for a user."""
        self.eval()
        with torch.no_grad():
            users_emb, items_emb = self.compute_graph_embeddings()
            
            user_emb = users_emb[user_idx]
            
            # Ensure computation happens on same device as embeddings
            scores = (user_emb @ items_emb.T).cpu().numpy()
            
            if exclude_items:
                scores[list(exclude_items)] = -np.inf
            
            top_items = np.argsort(scores)[::-1][:top_k]
            return [(int(item), float(scores[item])) for item in top_items]
    
    def predict_batch(self, user_indices, top_k=10, exclude_items_dict=None, device='cpu'):
        """Predict for a batch of users."""
        results = {}
        for user_idx in user_indices:
            exclude = exclude_items_dict.get(user_idx, None) if exclude_items_dict else None
            results[user_idx] = self.predict(user_idx, top_k=top_k, exclude_items=exclude, device=device)
        return results


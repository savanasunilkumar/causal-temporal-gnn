"""NCF (Neural Collaborative Filtering) baseline implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class NCF(nn.Module):
    """
    Neural Collaborative Filtering.
    
    Reference:
    He et al. "Neural Collaborative Filtering" WWW 2017
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128, 64, 32], learning_rate=0.001):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def forward(self, user_idx, item_idx):
        """Forward pass."""
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # Pass through MLP
        output = self.mlp(x).squeeze()
        return output
    
    def fit(self, train_data, num_epochs=20, batch_size=256, device='cpu'):
        """
        Train the NCF model.
        
        Args:
            train_data: DataFrame with 'user_idx' and 'item_idx' columns
            num_epochs: Number of training epochs
            batch_size: Batch size
            device: Device to train on
        """
        self.to(device)
        self.train()
        
        # Create user-item interaction dict
        user_items = {}
        for _, row in train_data.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            if user_idx not in user_items:
                user_items[user_idx] = []
            user_items[user_idx].append(item_idx)
        
        # All items set for negative sampling
        all_items = set(range(self.num_items))
        
        print(f"Training NCF with {len(train_data)} interactions...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle users
            users = list(user_items.keys())
            np.random.shuffle(users)
            
            pbar = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_start in pbar:
                batch_users = users[batch_start:batch_start + batch_size]
                
                user_idx_list = []
                item_idx_list = []
                label_list = []
                
                for user in batch_users:
                    pos_items = user_items[user]
                    
                    # Positive sample
                    pos_item = np.random.choice(pos_items)
                    user_idx_list.append(user)
                    item_idx_list.append(pos_item)
                    label_list.append(1.0)
                    
                    # Negative sample
                    neg_items = list(all_items - set(pos_items))
                    neg_item = np.random.choice(neg_items)
                    user_idx_list.append(user)
                    item_idx_list.append(neg_item)
                    label_list.append(0.0)
                
                user_idx = torch.LongTensor(user_idx_list).to(device)
                item_idx = torch.LongTensor(item_idx_list).to(device)
                labels = torch.FloatTensor(label_list).to(device)
                
                self.optimizer.zero_grad()
                predictions = self.forward(user_idx, item_idx)
                loss = self.criterion(predictions, labels)
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
            user_tensor = torch.LongTensor([user_idx] * self.num_items).to(device)
            item_tensor = torch.LongTensor(list(range(self.num_items))).to(device)
            
            scores = self.forward(user_tensor, item_tensor).cpu().numpy()
            
            # Exclude items
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


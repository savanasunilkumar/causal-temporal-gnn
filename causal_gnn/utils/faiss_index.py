"""FAISS integration for fast approximate nearest neighbor search."""

import numpy as np
import torch
from typing import List, Tuple, Optional
import logging


try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")


class FAISSIndex:
    """
    FAISS-based index for fast similarity search in recommendation embeddings.
    """

    def __init__(self, embedding_dim: int, index_type='IVF', nlist=100,
                 use_gpu=False, gpu_id=0):
        """
        Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ('Flat', 'IVF', 'HNSW', 'PQ')
            nlist: Number of clusters for IVF
            use_gpu: Whether to use GPU
            gpu_id: GPU device ID
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        self.index = None
        self.id_map = []  # Map FAISS index to original IDs

        self._build_index()

    def _build_index(self):
        """Build FAISS index based on type."""
        if self.index_type == 'Flat':
            # Exact search (slower but accurate)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product

        elif self.index_type == 'IVF':
            # Inverted file index (fast approximate search)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )

        elif self.index_type == 'HNSW':
            # Hierarchical Navigable Small World (very fast)
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT

        elif self.index_type == 'PQ':
            # Product Quantization (memory efficient)
            m = 8  # number of subquantizers
            bits = 8  # bits per subquantizer
            self.index = faiss.IndexPQ(self.embedding_dim, m, bits)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)

    def train(self, embeddings: np.ndarray):
        """
        Train the index (required for IVF, PQ).

        Args:
            embeddings: Training embeddings [num_samples, embedding_dim]
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Train index
        if hasattr(self.index, 'train'):
            print(f"Training FAISS index with {len(embeddings)} embeddings...")
            self.index.train(embeddings.astype('float32'))
            print("Training complete!")

    def add(self, embeddings: np.ndarray, ids: Optional[List[int]] = None):
        """
        Add embeddings to index.

        Args:
            embeddings: Embeddings to add [num_samples, embedding_dim]
            ids: Optional original IDs for the embeddings
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Normalize
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings.astype('float32'))

        # Store ID mapping
        if ids is None:
            ids = list(range(len(self.id_map), len(self.id_map) + len(embeddings)))
        self.id_map.extend(ids)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query embedding(s) [num_queries, embedding_dim]
            k: Number of neighbors to return

        Returns:
            distances: Similarity scores [num_queries, k]
            indices: Indices of neighbors [num_queries, k]
        """
        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()

        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query
        faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query.astype('float32'), k)

        # Map back to original IDs
        original_indices = np.array([[self.id_map[idx] for idx in row] for row in indices])

        return distances, original_indices

    def save(self, filepath: str):
        """
        Save index to disk.

        Args:
            filepath: Path to save index
        """
        # Move to CPU if on GPU
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, filepath)
        else:
            faiss.write_index(self.index, filepath)

        # Save ID mapping
        np.save(filepath + '.idmap.npy', np.array(self.id_map))

    def load(self, filepath: str):
        """
        Load index from disk.

        Args:
            filepath: Path to load index from
        """
        self.index = faiss.read_index(filepath)

        # Load ID mapping
        self.id_map = np.load(filepath + '.idmap.npy').tolist()

        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)


class RecommendationFAISS:
    """
    FAISS-based recommendation engine for fast item retrieval.
    """

    def __init__(self, item_embeddings: torch.Tensor, index_type='IVF',
                 use_gpu=False):
        """
        Initialize recommendation engine.

        Args:
            item_embeddings: Item embedding matrix [num_items, embedding_dim]
            index_type: FAISS index type
            use_gpu: Whether to use GPU
        """
        self.item_embeddings = item_embeddings
        self.num_items = item_embeddings.size(0)
        self.embedding_dim = item_embeddings.size(1)

        # Build FAISS index
        self.index = FAISSIndex(
            self.embedding_dim,
            index_type=index_type,
            use_gpu=use_gpu
        )

        # Train and add items
        item_emb_np = item_embeddings.cpu().numpy()
        self.index.train(item_emb_np)
        self.index.add(item_emb_np, ids=list(range(self.num_items)))

    def recommend_for_user(self, user_embedding: torch.Tensor, k: int = 10,
                          exclude_items: Optional[List[int]] = None) -> Tuple[List[int], List[float]]:
        """
        Get top-k recommendations for a user.

        Args:
            user_embedding: User embedding [embedding_dim]
            k: Number of recommendations
            exclude_items: Items to exclude

        Returns:
            item_ids: Recommended item IDs
            scores: Similarity scores
        """
        # Search
        scores, item_ids = self.index.search(user_embedding, k=k + len(exclude_items or []))

        # Flatten arrays
        scores = scores[0]
        item_ids = item_ids[0]

        # Filter excluded items
        if exclude_items:
            mask = ~np.isin(item_ids, exclude_items)
            item_ids = item_ids[mask][:k]
            scores = scores[mask][:k]
        else:
            item_ids = item_ids[:k]
            scores = scores[:k]

        return item_ids.tolist(), scores.tolist()

    def recommend_batch(self, user_embeddings: torch.Tensor, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recommendations for multiple users.

        Args:
            user_embeddings: User embeddings [batch_size, embedding_dim]
            k: Number of recommendations

        Returns:
            item_ids: [batch_size, k]
            scores: [batch_size, k]
        """
        scores, item_ids = self.index.search(user_embeddings, k=k)
        return item_ids, scores

    def find_similar_items(self, item_id: int, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Find similar items.

        Args:
            item_id: Query item ID
            k: Number of similar items

        Returns:
            similar_item_ids: Similar item IDs
            scores: Similarity scores
        """
        item_embedding = self.item_embeddings[item_id]
        scores, item_ids = self.index.search(item_embedding, k=k + 1)  # +1 to exclude self

        # Remove the item itself
        scores = scores[0][1:]  # Skip first (self)
        item_ids = item_ids[0][1:]

        return item_ids.tolist(), scores.tolist()

    def save(self, filepath: str):
        """Save FAISS index."""
        self.index.save(filepath)

    def load(self, filepath: str):
        """Load FAISS index."""
        self.index.load(filepath)


class HybridSearchEngine:
    """
    Hybrid search combining exact and approximate search.
    """

    def __init__(self, item_embeddings: torch.Tensor, use_gpu=False):
        """
        Initialize hybrid search.

        Args:
            item_embeddings: Item embeddings
            use_gpu: Use GPU acceleration
        """
        self.item_embeddings = item_embeddings

        # Exact search for small k
        self.exact_index = FAISSIndex(
            item_embeddings.size(1),
            index_type='Flat',
            use_gpu=use_gpu
        )

        # Approximate search for large k
        self.approx_index = FAISSIndex(
            item_embeddings.size(1),
            index_type='IVF',
            use_gpu=use_gpu
        )

        # Build indices
        item_emb_np = item_embeddings.cpu().numpy()
        self.approx_index.train(item_emb_np)

        self.exact_index.add(item_emb_np)
        self.approx_index.add(item_emb_np)

    def search(self, query: torch.Tensor, k: int = 10, use_exact=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with automatic selection of exact vs approximate.

        Args:
            query: Query embedding
            k: Number of results
            use_exact: Force exact (True) or approximate (False) search

        Returns:
            scores, item_ids
        """
        # Auto-select based on k
        if use_exact is None:
            use_exact = k <= 100

        if use_exact:
            return self.exact_index.search(query, k)
        else:
            return self.approx_index.search(query, k)


def benchmark_faiss_performance(item_embeddings: torch.Tensor, num_queries=1000):
    """
    Benchmark different FAISS index types.

    Args:
        item_embeddings: Item embeddings to index
        num_queries: Number of queries for benchmarking

    Returns:
        Benchmark results
    """
    import time

    results = {}

    index_types = ['Flat', 'IVF', 'HNSW']
    k = 10

    # Generate random queries
    queries = torch.randn(num_queries, item_embeddings.size(1))

    for index_type in index_types:
        print(f"\nBenchmarking {index_type}...")

        try:
            # Build index
            start = time.time()
            rec_engine = RecommendationFAISS(item_embeddings, index_type=index_type)
            build_time = time.time() - start

            # Query
            start = time.time()
            _, _ = rec_engine.recommend_batch(queries, k=k)
            query_time = time.time() - start

            results[index_type] = {
                'build_time': build_time,
                'query_time': query_time,
                'qps': num_queries / query_time
            }

            print(f"  Build time: {build_time:.2f}s")
            print(f"  Query time: {query_time:.2f}s")
            print(f"  QPS: {num_queries / query_time:.0f}")

        except Exception as e:
            print(f"  Error: {str(e)}")
            results[index_type] = {'error': str(e)}

    return results

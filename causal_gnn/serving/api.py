"""FastAPI serving layer for production deployment of UACT-GNN."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from datetime import datetime
import logging
import time
import asyncio
from functools import lru_cache

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Pydantic models for request/response validation
class User(BaseModel):
    """User model for API requests."""
    user_id: str
    features: Optional[Dict[str, Any]] = None


class Item(BaseModel):
    """Item model for API requests."""
    item_id: str
    features: Optional[Dict[str, Any]] = None


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    exclude_items: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    explain: bool = False


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: str
    recommendations: List[Dict[str, Any]]
    timestamp: str
    latency_ms: float
    model_version: str


class BatchRecommendationRequest(BaseModel):
    """Batch recommendation request."""
    user_ids: List[str]
    top_k: int = Field(default=10, ge=1, le=100)
    explain: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    num_parameters: int
    num_users: int
    num_items: int
    embedding_dim: int
    device: str


class RecommendationServer:
    """
    FastAPI-based recommendation server with caching and monitoring.
    """

    def __init__(self, model, config, user_id_map, item_id_map, device='cpu'):
        self.model = model
        self.config = config
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.device = device

        # Reverse mappings
        self.user_index_to_id = {v: k for k, v in user_id_map.items()}
        self.item_index_to_id = {v: k for k, v in item_id_map.items()}

        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_ttl = 3600  # 1 hour

        # Server start time
        self.start_time = time.time()

        # Model version
        self.model_version = "1.0.0"

        # Setup logging
        self.logger = logging.getLogger("RecommendationServer")
        self.logger.setLevel(logging.INFO)

        # Set model to eval mode
        self.model.eval()

        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.request_count = Counter(
                'recommendation_requests_total',
                'Total recommendation requests'
            )
            self.request_latency = Histogram(
                'recommendation_request_latency_seconds',
                'Recommendation request latency'
            )
            self.active_requests = Gauge(
                'recommendation_active_requests',
                'Number of active recommendation requests'
            )

    def get_or_compute_embeddings(self):
        """Get or compute user and item embeddings with caching."""
        cache_key = 'embeddings'

        if cache_key in self.embedding_cache:
            cache_entry = self.embedding_cache[cache_key]
            # Check if cache is still valid
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['user_emb'], cache_entry['item_emb']

        # Compute embeddings
        with torch.no_grad():
            _, user_emb, item_emb = self.model.forward(
                self.model.edge_index,
                self.model.edge_timestamps,
                self.model.time_indices
            )

        # Cache embeddings
        self.embedding_cache[cache_key] = {
            'user_emb': user_emb,
            'item_emb': item_emb,
            'timestamp': time.time()
        }

        return user_emb, item_emb

    async def get_recommendations(self, user_id: str, top_k: int = 10,
                                  exclude_items: List[str] = None):
        """
        Get recommendations for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_items: Items to exclude from recommendations

        Returns:
            List of recommended items with scores
        """
        start_time = time.time()

        if user_id not in self.user_id_map:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        user_idx = self.user_id_map[user_id]

        # Get embeddings
        user_emb, item_emb = self.get_or_compute_embeddings()

        # Compute scores
        user_embedding = user_emb[user_idx]
        scores = torch.matmul(user_embedding, item_emb.t())

        # Exclude items if specified
        if exclude_items:
            for item_id in exclude_items:
                if item_id in self.item_id_map:
                    item_idx = self.item_id_map[item_id]
                    scores[item_idx] = -float('inf')

        # Get top-k
        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.size(0)))

        # Convert to response format
        recommendations = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            item_id = self.item_index_to_id[idx]
            recommendations.append({
                'item_id': item_id,
                'score': float(score),
                'rank': len(recommendations) + 1
            })

        latency_ms = (time.time() - start_time) * 1000

        return {
            'recommendations': recommendations,
            'latency_ms': latency_ms
        }

    async def get_batch_recommendations(self, user_ids: List[str], top_k: int = 10):
        """
        Get recommendations for multiple users in batch.

        Args:
            user_ids: List of user IDs
            top_k: Number of recommendations per user

        Returns:
            Dictionary of recommendations per user
        """
        results = {}

        # Process in batch
        valid_user_indices = []
        valid_user_ids = []

        for user_id in user_ids:
            if user_id in self.user_id_map:
                valid_user_indices.append(self.user_id_map[user_id])
                valid_user_ids.append(user_id)

        if not valid_user_indices:
            return results

        # Get embeddings
        user_emb, item_emb = self.get_or_compute_embeddings()

        # Batch compute scores
        user_embeddings = user_emb[valid_user_indices]
        scores = torch.matmul(user_embeddings, item_emb.t())

        # Get top-k for each user
        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.size(1)), dim=1)

        # Convert to response format
        for i, user_id in enumerate(valid_user_ids):
            recommendations = []
            for idx, score in zip(top_indices[i].tolist(), top_scores[i].tolist()):
                item_id = self.item_index_to_id[idx]
                recommendations.append({
                    'item_id': item_id,
                    'score': float(score),
                    'rank': len(recommendations) + 1
                })

            results[user_id] = recommendations

        return results

    def get_model_info(self):
        """Get model information."""
        num_params = sum(p.numel() for p in self.model.parameters())

        return {
            'model_name': 'UACT-GNN',
            'version': self.model_version,
            'num_parameters': num_params,
            'num_users': len(self.user_id_map),
            'num_items': len(self.item_id_map),
            'embedding_dim': self.config.embedding_dim,
            'device': str(self.device)
        }

    def health_check(self):
        """Health check endpoint."""
        uptime = time.time() - self.start_time

        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'uptime_seconds': uptime
        }


# Create FastAPI app
def create_app(model, config, user_id_map, item_id_map, device='cpu'):
    """
    Create FastAPI application.

    Args:
        model: Trained recommendation model
        config: Model configuration
        user_id_map: Mapping from user IDs to indices
        item_id_map: Mapping from item IDs to indices
        device: Compute device

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="UACT-GNN Recommendation API",
        description="Production API for Universal Adaptive Causal Temporal GNN recommendations",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize server
    server = RecommendationServer(model, config, user_id_map, item_id_map, device)

    # Add Prometheus metrics endpoint (if available)
    if PROMETHEUS_AVAILABLE:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "UACT-GNN Recommendation API",
            "version": "1.0.0",
            "docs": "/docs"
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return server.health_check()

    @app.get("/model/info", response_model=ModelInfo)
    async def model_info():
        """Get model information."""
        return server.get_model_info()

    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest):
        """
        Get recommendations for a user.

        Args:
            request: Recommendation request

        Returns:
            Recommendation response with items and scores
        """
        try:
            if PROMETHEUS_AVAILABLE:
                server.request_count.inc()
                server.active_requests.inc()

            start_time = time.time()

            result = await server.get_recommendations(
                request.user_id,
                request.top_k,
                request.exclude_items
            )

            latency = time.time() - start_time

            if PROMETHEUS_AVAILABLE:
                server.request_latency.observe(latency)
                server.active_requests.dec()

            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=result['recommendations'],
                timestamp=datetime.now().isoformat(),
                latency_ms=result['latency_ms'],
                model_version=server.model_version
            )

        except HTTPException:
            raise
        except Exception as e:
            server.logger.error(f"Error in recommendation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/recommend/batch")
    async def recommend_batch(request: BatchRecommendationRequest):
        """
        Get recommendations for multiple users.

        Args:
            request: Batch recommendation request

        Returns:
            Dictionary of recommendations per user
        """
        try:
            start_time = time.time()

            results = await server.get_batch_recommendations(
                request.user_ids,
                request.top_k
            )

            latency = time.time() - start_time

            return {
                'results': results,
                'num_users': len(results),
                'latency_ms': latency * 1000,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            server.logger.error(f"Error in batch recommendation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/similar-items/{item_id}")
    async def similar_items(item_id: str, top_k: int = 10):
        """
        Get similar items based on embeddings.

        Args:
            item_id: Item ID
            top_k: Number of similar items

        Returns:
            List of similar items
        """
        try:
            if item_id not in server.item_id_map:
                raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

            item_idx = server.item_id_map[item_id]

            # Get embeddings
            _, item_emb = server.get_or_compute_embeddings()

            # Compute similarity
            item_embedding = item_emb[item_idx]
            similarities = torch.matmul(item_embedding, item_emb.t())

            # Exclude the item itself
            similarities[item_idx] = -float('inf')

            # Get top-k
            top_scores, top_indices = torch.topk(similarities, k=min(top_k, similarities.size(0)))

            # Convert to response
            similar_items = []
            for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
                similar_item_id = server.item_index_to_id[idx]
                similar_items.append({
                    'item_id': similar_item_id,
                    'similarity': float(score),
                    'rank': len(similar_items) + 1
                })

            return {
                'item_id': item_id,
                'similar_items': similar_items,
                'timestamp': datetime.now().isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            server.logger.error(f"Error in similar items: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear embedding cache."""
        server.embedding_cache.clear()
        return {"message": "Cache cleared successfully"}

    return app


def run_server(model, config, user_id_map, item_id_map, host="0.0.0.0", port=8000, device='cpu'):
    """
    Run the FastAPI server.

    Args:
        model: Trained recommendation model
        config: Model configuration
        user_id_map: User ID mapping
        item_id_map: Item ID mapping
        host: Host address
        port: Port number
        device: Compute device
    """
    import uvicorn

    app = create_app(model, config, user_id_map, item_id_map, device)

    uvicorn.run(app, host=host, port=port)

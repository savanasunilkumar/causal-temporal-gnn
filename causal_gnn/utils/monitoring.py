"""Prometheus metrics and monitoring for production deployment."""

import time
import functools
from typing import Dict, Any, Optional
import logging

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, push_to_gateway, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")


class MetricsCollector:
    """
    Collect and expose Prometheus metrics for recommendation system.
    """

    def __init__(self, namespace='uact_gnn', registry=None):
        """
        Initialize metrics collector.

        Args:
            namespace: Metrics namespace
            registry: Prometheus registry (None for default)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus-client not installed")

        self.namespace = namespace
        self.registry = registry or CollectorRegistry()

        self._init_metrics()

    def _init_metrics(self):
        """Initialize all metrics."""

        # Request metrics
        self.request_count = Counter(
            f'{self.namespace}_requests_total',
            'Total number of recommendation requests',
            ['endpoint', 'status'],
            registry=self.registry
        )

        self.request_latency = Histogram(
            f'{self.namespace}_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint'],
            registry=self.registry,
            buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0, 10.0)
        )

        self.active_requests = Gauge(
            f'{self.namespace}_active_requests',
            'Number of active requests',
            ['endpoint'],
            registry=self.registry
        )

        # Model metrics
        self.model_inference_time = Histogram(
            f'{self.namespace}_model_inference_seconds',
            'Model inference time',
            ['model_version'],
            registry=self.registry
        )

        self.embedding_cache_hits = Counter(
            f'{self.namespace}_cache_hits_total',
            'Number of embedding cache hits',
            ['cache_type'],
            registry=self.registry
        )

        self.embedding_cache_misses = Counter(
            f'{self.namespace}_cache_misses_total',
            'Number of embedding cache misses',
            ['cache_type'],
            registry=self.registry
        )

        # Recommendation metrics
        self.recommendations_served = Counter(
            f'{self.namespace}_recommendations_served_total',
            'Total recommendations served',
            ['user_type'],  # new, returning, cold_start
            registry=self.registry
        )

        self.recommendation_diversity = Histogram(
            f'{self.namespace}_recommendation_diversity',
            'Diversity score of recommendations',
            registry=self.registry
        )

        self.recommendation_novelty = Histogram(
            f'{self.namespace}_recommendation_novelty',
            'Novelty score of recommendations',
            registry=self.registry
        )

        # Data quality metrics
        self.data_quality_score = Gauge(
            f'{self.namespace}_data_quality_score',
            'Data quality score (0-1)',
            registry=self.registry
        )

        self.missing_features = Counter(
            f'{self.namespace}_missing_features_total',
            'Number of missing features encountered',
            ['feature_type'],
            registry=self.registry
        )

        # System metrics
        self.model_memory_mb = Gauge(
            f'{self.namespace}_model_memory_mb',
            'Model memory usage in MB',
            registry=self.registry
        )

        self.gpu_utilization = Gauge(
            f'{self.namespace}_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )

        # Business metrics
        self.items_recommended = Counter(
            f'{self.namespace}_items_recommended_total',
            'Total items recommended',
            ['item_category'],
            registry=self.registry
        )

        self.user_interactions = Counter(
            f'{self.namespace}_user_interactions_total',
            'Total user interactions',
            ['interaction_type'],  # click, purchase, etc.
            registry=self.registry
        )

        # Error metrics
        self.errors = Counter(
            f'{self.namespace}_errors_total',
            'Total errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )


class MonitoredRecommender:
    """
    Wrapper that adds monitoring to recommendation functions.
    """

    def __init__(self, recommender, metrics_collector: MetricsCollector):
        """
        Initialize monitored recommender.

        Args:
            recommender: Base recommender object
            metrics_collector: Metrics collector instance
        """
        self.recommender = recommender
        self.metrics = metrics_collector

    def recommend(self, user_id, top_k=10, **kwargs):
        """
        Monitored recommendation function.

        Args:
            user_id: User ID
            top_k: Number of recommendations
            **kwargs: Additional arguments

        Returns:
            Recommendations
        """
        # Track active requests
        self.metrics.active_requests.labels(endpoint='recommend').inc()

        start_time = time.time()

        try:
            # Get recommendations
            recommendations = self.recommender.recommend(user_id, top_k, **kwargs)

            # Track success
            self.metrics.request_count.labels(endpoint='recommend', status='success').inc()

            # Track recommendations served
            user_type = 'new' if self._is_new_user(user_id) else 'returning'
            self.metrics.recommendations_served.labels(user_type=user_type).inc()

            return recommendations

        except Exception as e:
            # Track error
            self.metrics.request_count.labels(endpoint='recommend', status='error').inc()
            self.metrics.errors.labels(
                error_type=type(e).__name__,
                endpoint='recommend'
            ).inc()
            raise

        finally:
            # Track latency
            latency = time.time() - start_time
            self.metrics.request_latency.labels(endpoint='recommend').observe(latency)

            # Decrement active requests
            self.metrics.active_requests.labels(endpoint='recommend').dec()

    def _is_new_user(self, user_id):
        """Check if user is new."""
        # Placeholder - implement based on your logic
        return False


def monitor_function(metrics_collector: MetricsCollector, endpoint_name: str):
    """
    Decorator to monitor function execution.

    Args:
        metrics_collector: Metrics collector
        endpoint_name: Name of endpoint

    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Track active requests
            metrics_collector.active_requests.labels(endpoint=endpoint_name).inc()

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Track success
                metrics_collector.request_count.labels(
                    endpoint=endpoint_name,
                    status='success'
                ).inc()

                return result

            except Exception as e:
                # Track error
                metrics_collector.request_count.labels(
                    endpoint=endpoint_name,
                    status='error'
                ).inc()
                metrics_collector.errors.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint_name
                ).inc()
                raise

            finally:
                # Track latency
                latency = time.time() - start_time
                metrics_collector.request_latency.labels(
                    endpoint=endpoint_name
                ).observe(latency)

                # Decrement active requests
                metrics_collector.active_requests.labels(endpoint=endpoint_name).dec()

        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Monitor model and system performance.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize performance monitor.

        Args:
            metrics_collector: Metrics collector
        """
        self.metrics = metrics_collector
        self.logger = logging.getLogger("PerformanceMonitor")

    def monitor_model_size(self, model):
        """
        Monitor model memory usage.

        Args:
            model: PyTorch model
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024

        self.metrics.model_memory_mb.set(size_mb)
        self.logger.info(f"Model size: {size_mb:.2f} MB")

    def monitor_gpu_usage(self):
        """Monitor GPU utilization."""
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    utilization = torch.cuda.utilization(i)
                    self.metrics.gpu_utilization.labels(gpu_id=str(i)).set(utilization)

        except Exception as e:
            self.logger.warning(f"Could not monitor GPU: {str(e)}")

    def monitor_cache_performance(self, cache_stats: Dict[str, Any]):
        """
        Monitor cache performance.

        Args:
            cache_stats: Dictionary with cache statistics
        """
        for cache_type, stats in cache_stats.items():
            if 'hits' in stats:
                self.metrics.embedding_cache_hits.labels(
                    cache_type=cache_type
                ).inc(stats['hits'])

            if 'misses' in stats:
                self.metrics.embedding_cache_misses.labels(
                    cache_type=cache_type
                ).inc(stats['misses'])


class MetricsExporter:
    """
    Export metrics to Prometheus.
    """

    def __init__(self, metrics_collector: MetricsCollector,
                 port=8000, pushgateway_url=None):
        """
        Initialize metrics exporter.

        Args:
            metrics_collector: Metrics collector
            port: Port for HTTP server
            pushgateway_url: Prometheus pushgateway URL
        """
        self.metrics = metrics_collector
        self.port = port
        self.pushgateway_url = pushgateway_url
        self.logger = logging.getLogger("MetricsExporter")

    def start_http_server(self):
        """Start HTTP server to expose metrics."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus not available")
            return

        start_http_server(self.port, registry=self.metrics.registry)
        self.logger.info(f"Metrics server started on port {self.port}")

    def push_to_gateway(self, job_name='uact_gnn'):
        """
        Push metrics to Prometheus Pushgateway.

        Args:
            job_name: Job name for metrics
        """
        if not PROMETHEUS_AVAILABLE or not self.pushgateway_url:
            self.logger.warning("Pushgateway not configured")
            return

        try:
            push_to_gateway(
                self.pushgateway_url,
                job=job_name,
                registry=self.metrics.registry
            )
            self.logger.info(f"Pushed metrics to {self.pushgateway_url}")

        except Exception as e:
            self.logger.error(f"Failed to push metrics: {str(e)}")


class A/BTestMonitor:
    """
    Monitor A/B test metrics.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize A/B test monitor.

        Args:
            metrics_collector: Metrics collector
        """
        self.metrics = metrics_collector

        # A/B test specific metrics
        self.variant_requests = Counter(
            f'{metrics_collector.namespace}_ab_requests_total',
            'Total requests per A/B test variant',
            ['experiment', 'variant'],
            registry=metrics_collector.registry
        )

        self.variant_conversions = Counter(
            f'{metrics_collector.namespace}_ab_conversions_total',
            'Total conversions per A/B test variant',
            ['experiment', 'variant'],
            registry=metrics_collector.registry
        )

    def track_request(self, experiment: str, variant: str):
        """
        Track A/B test request.

        Args:
            experiment: Experiment name
            variant: Variant name
        """
        self.variant_requests.labels(
            experiment=experiment,
            variant=variant
        ).inc()

    def track_conversion(self, experiment: str, variant: str):
        """
        Track A/B test conversion.

        Args:
            experiment: Experiment name
            variant: Variant name
        """
        self.variant_conversions.labels(
            experiment=experiment,
            variant=variant
        ).inc()


def create_monitoring_dashboard_config(namespace='uact_gnn'):
    """
    Generate Grafana dashboard configuration (JSON).

    Args:
        namespace: Metrics namespace

    Returns:
        Dashboard configuration dictionary
    """
    dashboard = {
        "dashboard": {
            "title": f"{namespace.upper()} Recommendation System",
            "panels": [
                {
                    "title": "Request Rate",
                    "targets": [{
                        "expr": f"rate({namespace}_requests_total[5m])"
                    }]
                },
                {
                    "title": "Request Latency (p95)",
                    "targets": [{
                        "expr": f"histogram_quantile(0.95, {namespace}_request_latency_seconds_bucket)"
                    }]
                },
                {
                    "title": "Active Requests",
                    "targets": [{
                        "expr": f"{namespace}_active_requests"
                    }]
                },
                {
                    "title": "Error Rate",
                    "targets": [{
                        "expr": f"rate({namespace}_errors_total[5m])"
                    }]
                },
                {
                    "title": "Cache Hit Rate",
                    "targets": [{
                        "expr": f"rate({namespace}_cache_hits_total[5m]) / (rate({namespace}_cache_hits_total[5m]) + rate({namespace}_cache_misses_total[5m]))"
                    }]
                },
                {
                    "title": "GPU Utilization",
                    "targets": [{
                        "expr": f"{namespace}_gpu_utilization_percent"
                    }]
                }
            ]
        }
    }

    return dashboard

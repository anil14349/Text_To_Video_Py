from prometheus_client import Counter, Gauge, Histogram, start_http_server
import psutil
import time
from typing import Dict, Any
import threading

class MetricsCollector:
    def __init__(self, port: int = 8000):
        # Initialize Prometheus metrics
        self.summary_requests = Counter(
            'resume_summary_requests_total',
            'Total number of resume summary requests'
        )
        self.summary_generation_time = Histogram(
            'summary_generation_seconds',
            'Time spent generating summaries',
            buckets=(1, 2, 5, 10, 30, 60, float("inf"))
        )
        self.audio_generation_time = Histogram(
            'audio_generation_seconds',
            'Time spent generating audio',
            buckets=(1, 2, 5, 10, 30, 60, float("inf"))
        )
        self.model_loading_time = Histogram(
            'model_loading_seconds',
            'Time spent loading models',
            buckets=(1, 2, 5, 10, 30, 60, float("inf"))
        )
        
        # System metrics
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
        self.gpu_usage = Gauge('gpu_usage_percent', 'GPU usage percentage')
        
        # Model metrics
        self.model_version = Gauge('model_version', 'Current model version')
        self.summary_quality = Gauge('summary_quality', 'Average summary quality score')
        
        # Start Prometheus HTTP server
        start_http_server(port)
        
        # Start system metrics collection
        self.start_system_metrics_collection()

    def start_system_metrics_collection(self):
        def collect_metrics():
            while True:
                # Collect CPU and memory metrics
                self.cpu_usage.set(psutil.cpu_percent())
                self.memory_usage.set(psutil.virtual_memory().percent)
                
                # Collect GPU metrics if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                        self.gpu_usage.set(gpu_usage * 100)
                except:
                    pass
                
                time.sleep(15)  # Update every 15 seconds

        # Start metrics collection in background
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()

    def record_summary_request(self):
        """Record a summary generation request."""
        self.summary_requests.inc()

    def record_summary_generation_time(self, duration: float):
        """Record time taken to generate a summary."""
        self.summary_generation_time.observe(duration)

    def record_audio_generation_time(self, duration: float):
        """Record time taken to generate audio."""
        self.audio_generation_time.observe(duration)

    def record_model_loading_time(self, duration: float):
        """Record time taken to load a model."""
        self.model_loading_time.observe(duration)

    def update_model_version(self, version: str):
        """Update the current model version."""
        try:
            version_num = float(version.replace('model_v', ''))
            self.model_version.set(version_num)
        except:
            pass

    def update_summary_quality(self, metrics: Dict[str, Any]):
        """Update the summary quality metrics."""
        if 'avg_bleu' in metrics:
            self.summary_quality.set(metrics['avg_bleu'])

class MonitoringDecorator:
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector

    def time_summary_generation(self, func):
        """Decorator to measure summary generation time."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.metrics.record_summary_generation_time(duration)
            return result
        return wrapper

    def time_audio_generation(self, func):
        """Decorator to measure audio generation time."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.metrics.record_audio_generation_time(duration)
            return result
        return wrapper

    def time_model_loading(self, func):
        """Decorator to measure model loading time."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self.metrics.record_model_loading_time(duration)
            return result
        return wrapper

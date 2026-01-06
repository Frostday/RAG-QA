"""
Metrics tracking for observability.

Tracks key performance metrics like latency, token usage, and request counts.
"""
import time
from typing import Dict, Optional, Any
from contextlib import contextmanager
from datetime import datetime, timezone
import json


class MetricsCollector:
    """
    Collects and tracks application metrics.
    
    Tracks:
    - Request latency
    - Token usage (if available)
    - Document processing metrics
    - Question answering metrics
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "documents_processed": 0,
            "questions_answered": 0,
            "total_tokens_used": 0,
            "total_latency_seconds": 0.0,
        }
    
    @contextmanager
    def track_request(self, request_type: str = "general"):
        """
        Context manager to track request metrics.
        
        Args:
            request_type: Type of request being tracked
        
        Yields:
            Dictionary to store request-specific metrics
        """
        start_time = time.time()
        request_metrics = {
            "request_type": request_type,
            "start_time": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "success": False,
            "latency_seconds": 0.0,
            "error": None,
        }
        
        self.metrics["requests_total"] += 1
        
        try:
            yield request_metrics
            request_metrics["success"] = True
            self.metrics["requests_success"] += 1
        except Exception as e:
            request_metrics["success"] = False
            request_metrics["error"] = str(e)
            self.metrics["requests_failed"] += 1
            raise
        finally:
            end_time = time.time()
            latency = end_time - start_time
            request_metrics["latency_seconds"] = round(latency, 3)
            request_metrics["end_time"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            self.metrics["total_latency_seconds"] += latency
    
    def record_document_processed(self, doc_type: str, chunk_count: int):
        """
        Record that a document was processed.
        
        Args:
            doc_type: Type of document (pdf, json)
            chunk_count: Number of chunks created
        """
        self.metrics["documents_processed"] += 1
        
        # Track by document type
        key = f"documents_processed_{doc_type}"
        self.metrics[key] = self.metrics.get(key, 0) + 1
        
        # Track chunk statistics
        self.metrics["total_chunks_created"] = self.metrics.get("total_chunks_created", 0) + chunk_count
    
    def record_question_answered(self, latency_seconds: float, token_count: Optional[int] = None):
        """
        Record that a question was answered.
        
        Args:
            latency_seconds: Time taken to answer
            token_count: Number of tokens used (if available)
        """
        self.metrics["questions_answered"] += 1
        
        if token_count:
            self.metrics["total_tokens_used"] += token_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary of current metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics["requests_total"] > 0:
            metrics["avg_latency_seconds"] = round(
                metrics["total_latency_seconds"] / metrics["requests_total"], 3
            )
            metrics["success_rate"] = round(
                metrics["requests_success"] / metrics["requests_total"], 3
            )
        
        if metrics["questions_answered"] > 0 and metrics["total_tokens_used"] > 0:
            metrics["avg_tokens_per_question"] = round(
                metrics["total_tokens_used"] / metrics["questions_answered"], 2
            )
        
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        return metrics
    
    def get_metrics_json(self) -> str:
        """
        Get metrics as JSON string.
        
        Returns:
            JSON-formatted metrics
        """
        return json.dumps(self.get_metrics(), indent=2)
    
    def reset(self):
        """Reset all metrics to initial values."""
        self.__init__()


# Global metrics collector instance
metrics_collector = MetricsCollector()


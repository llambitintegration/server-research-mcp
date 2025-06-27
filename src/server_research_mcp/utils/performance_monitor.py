"""
Performance Monitoring Utilities
================================

Production-ready performance monitoring for MCP server operations,
tool execution, and system metrics collection.

Features:
- Tool execution timing
- MCP server connection monitoring
- Memory and CPU usage tracking
- Metrics aggregation and reporting
- Integration with existing logging infrastructure
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationProfile:
    """Performance profile for a specific operation."""
    operation_name: str
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    error_count: int = 0
    total_calls: int = 0
    average_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_call: Optional[datetime] = None


class PerformanceMonitor:
    """Production performance monitoring system."""

    def __init__(self, enable_system_monitoring: bool = True, metrics_retention_hours: int = 24):
        """Initialize performance monitor.
        
        Args:
            enable_system_monitoring: Whether to collect system metrics
            metrics_retention_hours: How long to keep metrics in memory
        """
        self.enable_system_monitoring = enable_system_monitoring
        self.metrics_retention_hours = metrics_retention_hours
        
        # Performance data storage
        self.operation_profiles: Dict[str, OperationProfile] = {}
        self.system_metrics: deque = deque(maxlen=1000)  # Last 1000 system snapshots
        self.custom_metrics: deque = deque(maxlen=5000)  # Last 5000 custom metrics
        
        # Monitoring state
        self._active_operations: Dict[str, float] = {}  # operation_id -> start_time
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_enabled = False
        self._lock = threading.Lock()
        
        # System monitoring
        if self.enable_system_monitoring:
            self.start_system_monitoring()

    def start_system_monitoring(self, interval_seconds: int = 30):
        """Start background system monitoring.
        
        Args:
            interval_seconds: How often to collect system metrics
        """
        if self._monitoring_enabled:
            logger.warning("System monitoring already enabled")
            return
        
        self._monitoring_enabled = True
        
        def monitor_system():
            while self._monitoring_enabled:
                try:
                    self._collect_system_metrics()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    time.sleep(interval_seconds)
        
        self._monitoring_thread = threading.Thread(target=monitor_system, daemon=True)
        self._monitoring_thread.start()
        logger.info(f"Started system monitoring with {interval_seconds}s interval")

    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self._monitoring_enabled = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped system monitoring")

    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage for the current working directory
            disk = psutil.disk_usage('.')
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            system_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'process_memory_mb': process_memory.rss / (1024**2),
                'process_cpu_percent': process.cpu_percent()
            }
            
            with self._lock:
                self.system_metrics.append(system_snapshot)
                
            # Log warnings for high resource usage
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation.
        
        Args:
            operation_name: Name of the operation to time
            
        Returns:
            Operation ID for ending the operation
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        with self._lock:
            self._active_operations[operation_id] = start_time
            
        logger.debug(f"Started timing operation: {operation_name} (ID: {operation_id})")
        return operation_id

    def end_operation(self, operation_id: str, operation_name: str = None, success: bool = True, metadata: Dict[str, Any] = None) -> float:
        """End timing an operation.
        
        Args:
            operation_id: ID returned from start_operation
            operation_name: Name of the operation (extracted from ID if not provided)
            success: Whether the operation succeeded
            metadata: Additional metadata about the operation
            
        Returns:
            Duration in seconds
        """
        end_time = time.time()
        
        with self._lock:
            start_time = self._active_operations.pop(operation_id, None)
            
        if start_time is None:
            logger.warning(f"No start time found for operation ID: {operation_id}")
            return 0.0
        
        duration = end_time - start_time
        
        # Extract operation name from ID if not provided
        if operation_name is None:
            operation_name = operation_id.rsplit('_', 1)[0]
        
        # Update operation profile
        self._update_operation_profile(operation_name, duration, success, metadata or {})
        
        logger.debug(f"Completed operation: {operation_name} in {duration:.3f}s (success: {success})")
        return duration

    def _update_operation_profile(self, operation_name: str, duration: float, success: bool, metadata: Dict[str, Any]):
        """Update performance profile for an operation."""
        with self._lock:
            if operation_name not in self.operation_profiles:
                self.operation_profiles[operation_name] = OperationProfile(operation_name)
            
            profile = self.operation_profiles[operation_name]
            profile.execution_times.append(duration)
            profile.total_calls += 1
            profile.last_call = datetime.now()
            
            if success:
                profile.success_count += 1
            else:
                profile.error_count += 1
            
            # Update timing statistics
            profile.min_time = min(profile.min_time, duration)
            profile.max_time = max(profile.max_time, duration)
            profile.average_time = sum(profile.execution_times) / len(profile.execution_times)

    def record_metric(self, name: str, value: float, unit: str = "", metadata: Dict[str, Any] = None):
        """Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.custom_metrics.append(metric)
        
        logger.debug(f"Recorded metric: {name} = {value} {unit}")

    def get_operation_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance summary for operations.
        
        Args:
            operation_name: Specific operation name, or None for all operations
            
        Returns:
            Performance summary
        """
        with self._lock:
            if operation_name:
                if operation_name not in self.operation_profiles:
                    return {"error": f"No data for operation: {operation_name}"}
                
                profile = self.operation_profiles[operation_name]
                return {
                    "operation_name": profile.operation_name,
                    "total_calls": profile.total_calls,
                    "success_rate": profile.success_count / profile.total_calls if profile.total_calls > 0 else 0,
                    "average_time_ms": profile.average_time * 1000,
                    "min_time_ms": profile.min_time * 1000 if profile.min_time != float('inf') else 0,
                    "max_time_ms": profile.max_time * 1000,
                    "recent_calls": len(profile.execution_times),
                    "last_call": profile.last_call.isoformat() if profile.last_call else None
                }
            else:
                return {
                    "total_operations": len(self.operation_profiles),
                    "operations": {name: self.get_operation_summary(name) for name in self.operation_profiles.keys()}
                }

    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics."""
        if not self.system_metrics:
            return {"error": "No system metrics available"}
        
        with self._lock:
            recent_metrics = list(self.system_metrics)[-10:]  # Last 10 samples
        
        if not recent_metrics:
            return {"error": "No recent system metrics"}
        
        # Calculate averages for recent metrics
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_process_memory = sum(m['process_memory_mb'] for m in recent_metrics) / len(recent_metrics)
        
        latest = recent_metrics[-1]
        
        return {
            "latest_timestamp": latest['timestamp'],
            "current_cpu_percent": latest['cpu_percent'],
            "current_memory_percent": latest['memory_percent'],
            "current_memory_available_gb": latest['memory_available_gb'],
            "current_process_memory_mb": latest['process_memory_mb'],
            "recent_avg_cpu_percent": avg_cpu,
            "recent_avg_memory_percent": avg_memory,
            "recent_avg_process_memory_mb": avg_process_memory,
            "total_samples": len(self.system_metrics)
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_enabled": self._monitoring_enabled,
            "system_monitoring": self.enable_system_monitoring,
            "operations_summary": self.get_operation_summary(),
            "system_metrics": self.get_system_metrics_summary(),
            "custom_metrics_count": len(self.custom_metrics),
            "active_operations": len(self._active_operations)
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format.
        
        Args:
            format: Export format ("json", "csv")
            
        Returns:
            Formatted metrics string
        """
        if format.lower() == "json":
            return json.dumps(self.get_comprehensive_report(), indent=2)
        elif format.lower() == "csv":
            # Simple CSV export for operation summaries
            lines = ["operation_name,total_calls,success_rate,avg_time_ms,min_time_ms,max_time_ms"]
            
            with self._lock:
                for name, profile in self.operation_profiles.items():
                    summary = self.get_operation_summary(name)
                    if "error" not in summary:
                        lines.append(
                            f"{name},{summary['total_calls']},{summary['success_rate']:.3f},"
                            f"{summary['average_time_ms']:.3f},{summary['min_time_ms']:.3f},{summary['max_time_ms']:.3f}"
                        )
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def cleanup_old_metrics(self):
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        with self._lock:
            # Clean up custom metrics
            self.custom_metrics = deque(
                [m for m in self.custom_metrics if m.timestamp > cutoff_time],
                maxlen=self.custom_metrics.maxlen
            )
            
            # System metrics are cleaned up automatically by deque maxlen
            
        logger.debug(f"Cleaned up metrics older than {self.metrics_retention_hours} hours")


# Context manager for operation timing
class TimedOperation:
    """Context manager for automatic operation timing."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str, metadata: Dict[str, Any] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.operation_id = None
        self.success = True
    
    def __enter__(self):
        self.operation_id = self.monitor.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.success = exc_type is None
        if self.operation_id:
            duration = self.monitor.end_operation(
                self.operation_id, 
                self.operation_name, 
                self.success, 
                self.metadata
            )
            
            # Add exception info to metadata if there was an error
            if not self.success and exc_type:
                self.metadata['error_type'] = exc_type.__name__
                self.metadata['error_message'] = str(exc_val)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def time_operation(operation_name: str, metadata: Dict[str, Any] = None):
    """Decorator for timing function execution.
    
    Args:
        operation_name: Name of the operation
        metadata: Additional metadata to record
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with TimedOperation(monitor, operation_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def record_metric(name: str, value: float, unit: str = "", metadata: Dict[str, Any] = None):
    """Convenience function to record a metric using global monitor."""
    monitor = get_performance_monitor()
    monitor.record_metric(name, value, unit, metadata)


def get_performance_report() -> Dict[str, Any]:
    """Convenience function to get performance report from global monitor."""
    monitor = get_performance_monitor()
    return monitor.get_comprehensive_report()


# Example usage functions
def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    monitor = PerformanceMonitor()
    
    # Example 1: Manual timing
    op_id = monitor.start_operation("test_operation")
    time.sleep(0.1)  # Simulate work
    duration = monitor.end_operation(op_id, success=True)
    print(f"Manual timing: {duration:.3f}s")
    
    # Example 2: Context manager
    with TimedOperation(monitor, "context_operation"):
        time.sleep(0.05)  # Simulate work
    
    # Example 3: Decorator
    @time_operation("decorated_operation")
    def test_function():
        time.sleep(0.02)
        return "result"
    
    result = test_function()
    
    # Example 4: Custom metrics
    monitor.record_metric("queue_size", 42, "items")
    monitor.record_metric("success_rate", 0.95, "ratio")
    
    # Get report
    report = monitor.get_comprehensive_report()
    print(json.dumps(report, indent=2))
    
    monitor.stop_system_monitoring()


if __name__ == "__main__":
    demo_performance_monitoring() 
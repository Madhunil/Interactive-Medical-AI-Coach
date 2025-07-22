import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
from utils.logging import setup_logging, log_function_call

logger = setup_logging()

@dataclass
class LambdaMetrics:
    """Lambda function performance metrics"""
    function_name: str
    invocation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_invocation: Optional[datetime] = None
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None

class LambdaMonitor:
    """Monitor Lambda function performance and health"""
    
    def __init__(self, max_history_size: int = 1000):
        """Initialize Lambda monitor"""
        self.max_history_size = max_history_size
        self.metrics = defaultdict(lambda: LambdaMetrics(""))
        self.call_history = deque(maxlen=max_history_size)
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.warning_duration = 10.0  # seconds
        self.error_duration = 30.0    # seconds
        self.error_rate_threshold = 0.1  # 10% error rate
        
        logger.info("🔍 Lambda monitor initialized")
    
    @log_function_call
    def record_invocation(self, function_name: str, duration: float, 
                         success: bool, error: str = None) -> None:
        """Record a Lambda function invocation"""
        
        with self.lock:
            # Update metrics
            metrics = self.metrics[function_name]
            metrics.function_name = function_name
            metrics.invocation_count += 1
            metrics.last_invocation = datetime.now()
            
            if success:
                metrics.success_count += 1
                metrics.last_success = datetime.now()
            else:
                metrics.error_count += 1
                metrics.last_error = error or "Unknown error"
            
            # Update duration metrics
            metrics.total_duration += duration
            metrics.avg_duration = metrics.total_duration / metrics.invocation_count
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)
            
            # Add to call history
            call_record = {
                'timestamp': datetime.now().isoformat(),
                'function_name': function_name,
                'duration': duration,
                'success': success,
                'error': error
            }
            self.call_history.append(call_record)
            
            # Log performance alerts
            self._check_performance_alerts(function_name, duration, success, error)
    
    def _check_performance_alerts(self, function_name: str, duration: float, 
                                success: bool, error: str = None) -> None:
        """Check for performance alerts and log warnings"""
        
        # Duration alerts
        if duration > self.error_duration:
            logger.error(f"🚨 Lambda {function_name}: Very slow response ({duration:.2f}s)")
        elif duration > self.warning_duration:
            logger.warning(f"⚠️ Lambda {function_name}: Slow response ({duration:.2f}s)")
        
        # Error alerts
        if not success:
            logger.error(f"❌ Lambda {function_name}: Invocation failed - {error}")
            
            # Check error rate
            metrics = self.metrics[function_name]
            if metrics.invocation_count >= 10:  # Only check after 10+ calls
                error_rate = metrics.error_count / metrics.invocation_count
                if error_rate >= self.error_rate_threshold:
                    logger.error(f"🚨 Lambda {function_name}: High error rate ({error_rate:.1%})")
    
    @log_function_call
    def get_metrics(self, function_name: str = None) -> Dict[str, Any]:
        """Get metrics for a specific function or all functions"""
        
        with self.lock:
            if function_name:
                if function_name in self.metrics:
                    return self._metrics_to_dict(self.metrics[function_name])
                else:
                    return {}
            else:
                return {
                    name: self._metrics_to_dict(metrics) 
                    for name, metrics in self.metrics.items()
                }
    
    def _metrics_to_dict(self, metrics: LambdaMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        error_rate = (metrics.error_count / metrics.invocation_count 
                     if metrics.invocation_count > 0 else 0.0)
        
        return {
            'function_name': metrics.function_name,
            'invocation_count': metrics.invocation_count,
            'success_count': metrics.success_count,
            'error_count': metrics.error_count,
            'success_rate': (metrics.success_count / metrics.invocation_count 
                           if metrics.invocation_count > 0 else 0.0),
            'error_rate': error_rate,
            'avg_duration': metrics.avg_duration,
            'min_duration': metrics.min_duration if metrics.min_duration != float('inf') else 0.0,
            'max_duration': metrics.max_duration,
            'last_invocation': metrics.last_invocation.isoformat() if metrics.last_invocation else None,
            'last_success': metrics.last_success.isoformat() if metrics.last_success else None,
            'last_error': metrics.last_error,
            'health_status': self._get_health_status(metrics)
        }
    
    def _get_health_status(self, metrics: LambdaMetrics) -> str:
        """Determine health status of a Lambda function"""
        if metrics.invocation_count == 0:
            return "unknown"
        
        error_rate = metrics.error_count / metrics.invocation_count
        
        # Check recent activity (last 5 minutes)
        if metrics.last_invocation:
            time_since_last = datetime.now() - metrics.last_invocation
            if time_since_last > timedelta(minutes=5):
                recent_calls = [call for call in self.call_history 
                              if (datetime.now() - datetime.fromisoformat(call['timestamp'])) < timedelta(minutes=5)
                              and call['function_name'] == metrics.function_name]
                
                if not recent_calls:
                    return "idle"
        
        # Health based on error rate and performance
        if error_rate >= 0.5:  # 50% error rate
            return "critical"
        elif error_rate >= 0.2:  # 20% error rate
            return "degraded"
        elif metrics.avg_duration > self.warning_duration:
            return "slow"
        else:
            return "healthy"
    
    @log_function_call
    def get_recent_calls(self, function_name: str = None, 
                        minutes: int = 10) -> List[Dict[str, Any]]:
        """Get recent Lambda calls"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_calls = []
        for call in self.call_history:
            call_time = datetime.fromisoformat(call['timestamp'])
            if call_time >= cutoff_time:
                if function_name is None or call['function_name'] == function_name:
                    recent_calls.append(call)
        
        return sorted(recent_calls, key=lambda x: x['timestamp'], reverse=True)
    
    @log_function_call
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        
        with self.lock:
            total_calls = sum(m.invocation_count for m in self.metrics.values())
            total_errors = sum(m.error_count for m in self.metrics.values())
            total_duration = sum(m.total_duration for m in self.metrics.values())
            
            function_count = len(self.metrics)
            healthy_functions = sum(1 for m in self.metrics.values() 
                                  if self._get_health_status(m) == "healthy")
            
            return {
                'total_functions': function_count,
                'healthy_functions': healthy_functions,
                'unhealthy_functions': function_count - healthy_functions,
                'total_invocations': total_calls,
                'total_errors': total_errors,
                'overall_error_rate': total_errors / total_calls if total_calls > 0 else 0.0,
                'avg_response_time': total_duration / total_calls if total_calls > 0 else 0.0,
                'monitoring_since': min(m.last_invocation for m in self.metrics.values() 
                                      if m.last_invocation) if self.metrics else None
            }
    
    @log_function_call
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to JSON file"""
        
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_performance_summary(),
                'function_metrics': self.get_metrics(),
                'recent_calls': self.get_recent_calls(minutes=60)  # Last hour
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"📊 Lambda metrics exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export metrics: {e}")
            return False
    
    @log_function_call
    def reset_metrics(self, function_name: str = None) -> None:
        """Reset metrics for a function or all functions"""
        
        with self.lock:
            if function_name:
                if function_name in self.metrics:
                    del self.metrics[function_name]
                    logger.info(f"🔄 Reset metrics for {function_name}")
            else:
                self.metrics.clear()
                self.call_history.clear()
                logger.info("🔄 Reset all Lambda metrics")

# Global monitor instance
lambda_monitor = LambdaMonitor()

def monitor_lambda_call(function_name: str):
    """Decorator to monitor Lambda function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None
            
            try:
                result = func(*args, **kwargs)
                success = result.get('success', False) if isinstance(result, dict) else True
                if not success and isinstance(result, dict):
                    error = result.get('error', 'Unknown error')
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                lambda_monitor.record_invocation(function_name, duration, success, error)
        
        return wrapper
    return decorator
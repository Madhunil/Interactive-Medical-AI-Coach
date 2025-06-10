import os
import sys
from datetime import datetime
from loguru import logger

def setup_logging():
    """Configure comprehensive loguru logging"""
    
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Console logging (colored, formatted)
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Debug file logging (everything)
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="1 day",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # Error file logging (errors only)
    logger.add(
        "logs/errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # AWS operations logging (separate file)
    logger.add(
        "logs/aws_operations_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | AWS | {message}",
        rotation="1 day",
        retention="7 days",
        filter=lambda record: "aws" in record["extra"].get("category", ""),
        backtrace=True
    )
    
    # User interactions logging
    logger.add(
        "logs/user_interactions_{time:YYYY-MM-DD}.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | USER | {message}",
        rotation="1 day",
        retention="30 days",
        filter=lambda record: "user" in record["extra"].get("category", "")
    )
    
    # Performance logging
    logger.add(
        "logs/performance_{time:YYYY-MM-DD}.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {message}",
        rotation="1 day",
        retention="7 days",
        filter=lambda record: "performance" in record["extra"].get("category", "")
    )
    
    logger.info("🚀 Loguru logging system initialized")
    logger.debug(f"Logs directory: {os.path.abspath('logs')}")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Working directory: {os.getcwd()}")
    
    return logger

# Decorators for automatic function logging
def log_function_call(func):
    """Decorator to log function entry/exit with parameters and timing"""
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Log function entry
        logger.debug(f"🔵 ENTER {func_name}")
        logger.debug(f"   Args: {len(args)} positional args")
        logger.debug(f"   Kwargs: {list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful exit
            logger.debug(f"✅ EXIT {func_name} - Success in {execution_time:.3f}s")
            logger.bind(category="performance").info(f"{func_name}: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error exit
            logger.error(f"❌ EXIT {func_name} - Error in {execution_time:.3f}s: {str(e)}")
            logger.bind(category="performance").info(f"{func_name}: {execution_time:.3f}s (FAILED)")
            raise
            
    return wrapper

def log_aws_operation(operation_name):
    """Decorator specifically for AWS operations"""
    def decorator(func):
        from functools import wraps
        import time
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            logger.bind(category="aws").info(f"🌐 AWS {operation_name} - Starting")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.bind(category="aws").info(f"✅ AWS {operation_name} - Success in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.bind(category="aws").error(f"❌ AWS {operation_name} - Failed in {execution_time:.3f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator

def log_user_interaction(interaction_type):
    """Decorator for user interactions"""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.bind(category="user").info(f"👤 {interaction_type}")
            
            try:
                result = func(*args, **kwargs)
                logger.bind(category="user").info(f"✅ {interaction_type} - Completed")
                return result
            except Exception as e:
                logger.bind(category="user").error(f"❌ {interaction_type} - Failed: {str(e)}")
                raise
                
        return wrapper
    return decorator
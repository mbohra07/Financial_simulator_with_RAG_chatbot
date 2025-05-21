"""
Redis cache utility for the Financial Crew application.
"""

import os
import json
import redis
import hashlib
from datetime import timedelta
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedisCache:
    """Redis cache utility for storing and retrieving data."""
    
    def __init__(self, namespace="financial_crew"):
        """
        Initialize Redis connection using environment variables.
        
        Args:
            namespace (str): Namespace prefix for cache keys
        """
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", ""),
            decode_responses=True  # Automatically decode responses to strings
        )
        self.namespace = namespace
        self.default_expiry = timedelta(hours=24)  # Default cache expiry time
    
    def _get_key(self, key):
        """
        Generate a namespaced key.
        
        Args:
            key (str): Original key
        
        Returns:
            str: Namespaced key
        """
        return f"{self.namespace}:{key}"
    
    def set(self, key, value, expiry=None):
        """
        Store data in Redis cache.
        
        Args:
            key (str): Cache key
            value (any): Data to cache (will be JSON serialized)
            expiry (timedelta, optional): Custom expiry time
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(value)
            
            # Set expiry time (in seconds)
            ex_seconds = int(expiry.total_seconds()) if expiry else int(self.default_expiry.total_seconds())
            
            # Store in Redis
            return self.redis_client.setex(self._get_key(key), ex_seconds, json_data)
        except Exception as e:
            print(f"Error setting cache data: {e}")
            return False
    
    def get(self, key):
        """
        Retrieve data from Redis cache.
        
        Args:
            key (str): Cache key
        
        Returns:
            any: Cached data or None if not found
        """
        try:
            data = self.redis_client.get(self._get_key(key))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting cache data: {e}")
            return None
    
    def delete(self, key):
        """
        Delete data from Redis cache.
        
        Args:
            key (str): Cache key
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return bool(self.redis_client.delete(self._get_key(key)))
        except Exception as e:
            print(f"Error deleting cache data: {e}")
            return False
    
    def clear_namespace(self):
        """
        Clear all keys in the current namespace.
        
        Returns:
            int: Number of keys deleted
        """
        try:
            keys = self.redis_client.keys(f"{self.namespace}:*")
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Error clearing namespace: {e}")
            return 0


def redis_cache(expiry=None, namespace="financial_crew"):
    """
    Decorator for caching function results in Redis.
    
    Args:
        expiry (timedelta, optional): Cache expiry time
        namespace (str): Redis namespace
    
    Returns:
        function: Decorated function
    """
    cache = RedisCache(namespace=namespace)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # If not in cache, call the function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(key, result, expiry)
            
            return result
        return wrapper
    return decorator

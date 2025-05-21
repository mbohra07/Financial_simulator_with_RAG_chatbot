"""
Redis-based rate limiter for the Financial Crew application.
"""

import os
import time
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedisRateLimiter:
    """Redis-based rate limiter to control API request rates."""
    
    def __init__(self, limit=100, window=3600):
        """
        Initialize Redis connection using environment variables.
        
        Args:
            limit (int): Maximum number of requests allowed in the time window
            window (int): Time window in seconds
        """
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", ""),
            decode_responses=True
        )
        self.limit = limit
        self.window = window
    
    def is_rate_limited(self, key, increment=True):
        """
        Check if a key is rate limited.
        
        Args:
            key (str): Identifier for the client (e.g., IP address, user ID)
            increment (bool): Whether to increment the counter if not rate limited
        
        Returns:
            tuple: (is_limited, remaining, reset_time)
                - is_limited (bool): True if rate limited, False otherwise
                - remaining (int): Number of requests remaining in the window
                - reset_time (int): Time in seconds until the window resets
        """
        # Get the current timestamp
        current_time = int(time.time())
        
        # Create a key with the current window
        window_start = current_time - (current_time % self.window)
        window_key = f"rate_limit:{key}:{window_start}"
        
        # Get the current count
        count = int(self.redis_client.get(window_key) or 0)
        
        # Check if rate limited
        is_limited = count >= self.limit
        
        # Increment the counter if not rate limited and increment is True
        if not is_limited and increment:
            # Use pipeline to ensure atomic operations
            pipe = self.redis_client.pipeline()
            pipe.incr(window_key)
            # Set expiry if it doesn't exist
            pipe.expire(window_key, self.window)
            pipe.execute()
            count += 1
        
        # Calculate remaining requests and reset time
        remaining = max(0, self.limit - count)
        reset_time = window_start + self.window - current_time
        
        return is_limited, remaining, reset_time
    
    def get_rate_limit_headers(self, key, increment=True):
        """
        Get rate limit headers for HTTP responses.
        
        Args:
            key (str): Identifier for the client
            increment (bool): Whether to increment the counter
        
        Returns:
            dict: Rate limit headers
        """
        is_limited, remaining, reset_time = self.is_rate_limited(key, increment)
        
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "X-RateLimit-Limited": "1" if is_limited else "0"
        }
    
    def reset_rate_limit(self, key):
        """
        Reset rate limit for a key.
        
        Args:
            key (str): Identifier for the client
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the current timestamp
            current_time = int(time.time())
            
            # Create a key with the current window
            window_start = current_time - (current_time % self.window)
            window_key = f"rate_limit:{key}:{window_start}"
            
            # Delete the key
            return bool(self.redis_client.delete(window_key))
        except Exception as e:
            print(f"Error resetting rate limit: {e}")
            return False

"""
Examples of using Redis in the Financial Crew application.
"""

import time
import uuid
from datetime import timedelta

# Import Redis utilities
from redis_session import RedisSession
from redis_cache_util import RedisCache, redis_cache
from redis_rate_limiter import RedisRateLimiter
from redis_pubsub import RedisPubSub

# Example 1: Using Redis Session
def session_example():
    """Example of using Redis for session management."""
    print("\n=== Redis Session Example ===")
    
    # Initialize session manager
    session_manager = RedisSession()
    
    # Create a new session
    session_id = str(uuid.uuid4())
    session_data = {
        "user_id": "user123",
        "username": "john_doe",
        "preferences": {
            "theme": "dark",
            "currency": "USD"
        },
        "last_login": time.time()
    }
    
    # Store session
    success = session_manager.set_session(session_id, session_data, expiry=timedelta(minutes=30))
    print(f"Session created: {success}, ID: {session_id}")
    
    # Retrieve session
    retrieved_session = session_manager.get_session(session_id)
    print(f"Retrieved session: {retrieved_session}")
    
    # Update session expiry
    success = session_manager.update_session_expiry(session_id, expiry=timedelta(hours=1))
    print(f"Session expiry updated: {success}")
    
    # Delete session
    success = session_manager.delete_session(session_id)
    print(f"Session deleted: {success}")
    
    # Verify deletion
    retrieved_session = session_manager.get_session(session_id)
    print(f"Session after deletion: {retrieved_session}")


# Example 2: Using Redis Cache
def cache_example():
    """Example of using Redis for caching."""
    print("\n=== Redis Cache Example ===")
    
    # Initialize cache
    cache = RedisCache(namespace="financial_data")
    
    # Store data in cache
    data = {
        "stock_prices": {
            "AAPL": 150.25,
            "MSFT": 290.75,
            "GOOGL": 2750.50
        },
        "timestamp": time.time()
    }
    success = cache.set("market_data", data, expiry=timedelta(minutes=15))
    print(f"Data cached: {success}")
    
    # Retrieve from cache
    cached_data = cache.get("market_data")
    print(f"Retrieved from cache: {cached_data}")
    
    # Delete from cache
    success = cache.delete("market_data")
    print(f"Data deleted from cache: {success}")
    
    # Verify deletion
    cached_data = cache.get("market_data")
    print(f"Data after deletion: {cached_data}")


# Example 3: Using Redis Cache Decorator
@redis_cache(expiry=timedelta(minutes=5), namespace="financial_calculations")
def expensive_calculation(a, b):
    """Simulate an expensive calculation that benefits from caching."""
    print("Performing expensive calculation...")
    time.sleep(2)  # Simulate long processing time
    return a * b + a / b


def cache_decorator_example():
    """Example of using Redis cache decorator."""
    print("\n=== Redis Cache Decorator Example ===")
    
    # First call - will execute the function
    start = time.time()
    result1 = expensive_calculation(10, 5)
    end = time.time()
    print(f"First call result: {result1}, took {end - start:.2f} seconds")
    
    # Second call with same parameters - should use cache
    start = time.time()
    result2 = expensive_calculation(10, 5)
    end = time.time()
    print(f"Second call result: {result2}, took {end - start:.2f} seconds")
    
    # Call with different parameters - will execute the function again
    start = time.time()
    result3 = expensive_calculation(20, 10)
    end = time.time()
    print(f"Different parameters result: {result3}, took {end - start:.2f} seconds")


# Example 4: Using Redis Rate Limiter
def rate_limiter_example():
    """Example of using Redis for rate limiting."""
    print("\n=== Redis Rate Limiter Example ===")
    
    # Initialize rate limiter (5 requests per 10 seconds)
    rate_limiter = RedisRateLimiter(limit=5, window=10)
    
    # Simulate requests
    user_id = "user123"
    
    print("Simulating 7 requests in quick succession:")
    for i in range(7):
        is_limited, remaining, reset_time = rate_limiter.is_rate_limited(user_id)
        print(f"Request {i+1}: Limited: {is_limited}, Remaining: {remaining}, Reset in: {reset_time}s")
    
    # Get headers for HTTP response
    headers = rate_limiter.get_rate_limit_headers(user_id, increment=False)
    print(f"Rate limit headers: {headers}")
    
    # Reset rate limit
    success = rate_limiter.reset_rate_limit(user_id)
    print(f"Rate limit reset: {success}")
    
    # Check after reset
    is_limited, remaining, reset_time = rate_limiter.is_rate_limited(user_id)
    print(f"After reset: Limited: {is_limited}, Remaining: {remaining}, Reset in: {reset_time}s")


# Example 5: Using Redis Pub/Sub
def pubsub_example():
    """Example of using Redis for pub/sub messaging."""
    print("\n=== Redis Pub/Sub Example ===")
    
    # Initialize pub/sub
    pubsub = RedisPubSub()
    
    # Define message handler
    def handle_message(message):
        print(f"Received message: {message}")
    
    # Subscribe to channel
    success = pubsub.subscribe("financial_updates", handle_message)
    print(f"Subscribed to channel: {success}")
    
    # Publish messages
    for i in range(3):
        message = {
            "type": "market_update",
            "data": {
                "index": "S&P 500",
                "value": 4200 + i * 10,
                "change": 0.5 + i * 0.1
            },
            "timestamp": time.time()
        }
        recipients = pubsub.publish("financial_updates", message)
        print(f"Published message {i+1}, received by {recipients} clients")
        time.sleep(1)
    
    # Unsubscribe
    success = pubsub.unsubscribe("financial_updates")
    print(f"Unsubscribed from channel: {success}")
    
    # Close connection
    pubsub.close()


# Run examples
if __name__ == "__main__":
    session_example()
    cache_example()
    cache_decorator_example()
    rate_limiter_example()
    pubsub_example()

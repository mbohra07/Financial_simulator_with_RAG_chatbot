"""
Test Redis connection and basic operations.
"""

import os
import json
import time
from dotenv import load_dotenv
import redis

# Load environment variables
load_dotenv()

def test_redis_connection():
    """Test basic Redis connection and operations."""
    print("Testing Redis connection...")
    
    # Get Redis connection details from environment variables
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_password = os.getenv("REDIS_PASSWORD", "")
    
    # Connect to Redis
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True  # Automatically decode responses to strings
    )
    
    # Test connection
    ping_result = r.ping()
    print(f"Redis connection successful: {ping_result}")
    
    # Test basic operations
    print("\nTesting basic Redis operations:")
    
    # String operations
    print("\n1. String operations:")
    r.set("test:string", "Hello Redis!")
    value = r.get("test:string")
    print(f"  - SET and GET: {value}")
    
    # Set expiration
    r.setex("test:expiring_string", 10, "This will expire in 10 seconds")
    value = r.get("test:expiring_string")
    ttl = r.ttl("test:expiring_string")
    print(f"  - SETEX: {value} (expires in {ttl} seconds)")
    
    # Hash operations
    print("\n2. Hash operations:")
    r.hset("test:hash", mapping={"name": "John", "age": "30", "city": "New York"})
    hash_value = r.hgetall("test:hash")
    print(f"  - HSET and HGETALL: {hash_value}")
    
    # List operations
    print("\n3. List operations:")
    r.lpush("test:list", "item3", "item2", "item1")
    list_value = r.lrange("test:list", 0, -1)
    print(f"  - LPUSH and LRANGE: {list_value}")
    
    # Set operations
    print("\n4. Set operations:")
    r.sadd("test:set", "item1", "item2", "item3", "item1")  # Duplicate will be ignored
    set_value = r.smembers("test:set")
    print(f"  - SADD and SMEMBERS: {set_value}")
    
    # Sorted set operations
    print("\n5. Sorted set operations:")
    r.zadd("test:zset", {"item1": 1, "item2": 2, "item3": 3})
    zset_value = r.zrange("test:zset", 0, -1, withscores=True)
    print(f"  - ZADD and ZRANGE: {zset_value}")
    
    # JSON operations (using string as JSON)
    print("\n6. JSON operations (using string):")
    json_data = {"name": "Alice", "age": 25, "skills": ["Python", "Redis", "FastAPI"]}
    r.set("test:json", json.dumps(json_data))
    json_value = json.loads(r.get("test:json"))
    print(f"  - JSON SET and GET: {json_value}")
    
    # Clean up
    print("\nCleaning up test keys...")
    keys = r.keys("test:*")
    if keys:
        r.delete(*keys)
        print(f"Deleted {len(keys)} test keys")
    
    print("\nRedis test completed successfully!")

if __name__ == "__main__":
    test_redis_connection()

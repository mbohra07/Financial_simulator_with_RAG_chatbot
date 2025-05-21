"""
Test script for Redis integration with the financial simulation.
"""

import os
import json
import time
from dotenv import load_dotenv

# Import Redis utilities
from langgraph_implementation import (
    redis_cache_get,
    redis_cache_set,
    redis_cache_delete,
    redis_rate_limit,
    simulate_timeline_langgraph
)

# Load environment variables
load_dotenv()

def test_redis_simulation():
    """Test Redis integration with the financial simulation."""
    print("Testing Redis integration with financial simulation...")
    
    # Test inputs
    test_inputs = {
        "user_id": "test_redis_user",
        "user_name": "Redis Test User",
        "age": 30,
        "occupation": "Software Engineer",
        "income": 6000,
        "expenses": [
            {"name": "Rent", "amount": 1500},
            {"name": "Groceries", "amount": 500},
            {"name": "Utilities", "amount": 300},
            {"name": "Transportation", "amount": 200}
        ],
        "financial_goal": "Save for a down payment on a house",
        "financial_type": "Saver",
        "risk_level": "Moderate"
    }
    
    # Run simulation for 1 month with Redis caching
    print("\n1. Running simulation with Redis caching...")
    start_time = time.time()
    simulation_result = simulate_timeline_langgraph(
        n_months=1,
        simulation_unit="Months",
        user_inputs=test_inputs,
        use_cache=True
    )
    end_time = time.time()
    print(f"First run completed in {end_time - start_time:.2f} seconds")
    
    # Get the simulation ID
    simulation_id = simulation_result.get("simulation_id")
    print(f"Simulation ID: {simulation_id}")
    
    # Run the same simulation again (should use cache)
    print("\n2. Running the same simulation again (should use cache)...")
    start_time = time.time()
    cached_result = simulate_timeline_langgraph(
        n_months=1,
        simulation_unit="Months",
        user_inputs=test_inputs,
        simulation_id=simulation_id,
        use_cache=True
    )
    end_time = time.time()
    print(f"Second run completed in {end_time - start_time:.2f} seconds")
    
    # Test rate limiting
    print("\n3. Testing rate limiting...")
    user_id = "test_user"
    for i in range(5):
        is_limited, remaining, reset_time = redis_rate_limit(
            key=user_id,
            limit=3,
            window=60,
            namespace="test_rate_limits"
        )
        print(f"Request {i+1}: Limited: {is_limited}, Remaining: {remaining}, Reset in: {reset_time}s")
    
    # Test cache operations
    print("\n4. Testing basic cache operations...")
    test_key = "test_key"
    test_data = {"name": "Test User", "score": 95, "tags": ["finance", "simulation"]}
    
    # Set data in cache
    success = redis_cache_set(test_key, test_data, namespace="test_namespace")
    print(f"Set data in cache: {success}")
    
    # Get data from cache
    cached_data = redis_cache_get(test_key, namespace="test_namespace")
    print(f"Retrieved from cache: {cached_data}")
    
    # Delete from cache
    success = redis_cache_delete(test_key, namespace="test_namespace")
    print(f"Deleted from cache: {success}")
    
    # Verify deletion
    cached_data = redis_cache_get(test_key, namespace="test_namespace")
    print(f"After deletion: {cached_data}")
    
    print("\nRedis integration test completed successfully!")

if __name__ == "__main__":
    test_redis_simulation()

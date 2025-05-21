"""
Redis session management for the Financial Crew application.
"""

import os
import json
import redis
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedisSession:
    """Redis session management for storing user session data."""
    
    def __init__(self):
        """Initialize Redis connection using environment variables."""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", ""),
            decode_responses=True  # Automatically decode responses to strings
        )
        self.default_expiry = timedelta(hours=24)  # Default session expiry time
    
    def set_session(self, session_id, data, expiry=None):
        """
        Store session data in Redis.
        
        Args:
            session_id (str): Unique session identifier
            data (dict): Session data to store
            expiry (timedelta, optional): Custom expiry time
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data)
            
            # Set expiry time (in seconds)
            ex_seconds = int(expiry.total_seconds()) if expiry else int(self.default_expiry.total_seconds())
            
            # Store in Redis
            return self.redis_client.setex(f"session:{session_id}", ex_seconds, json_data)
        except Exception as e:
            print(f"Error setting session data: {e}")
            return False
    
    def get_session(self, session_id):
        """
        Retrieve session data from Redis.
        
        Args:
            session_id (str): Unique session identifier
        
        Returns:
            dict: Session data or None if not found
        """
        try:
            data = self.redis_client.get(f"session:{session_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting session data: {e}")
            return None
    
    def delete_session(self, session_id):
        """
        Delete a session from Redis.
        
        Args:
            session_id (str): Unique session identifier
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return bool(self.redis_client.delete(f"session:{session_id}"))
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def update_session_expiry(self, session_id, expiry=None):
        """
        Update the expiry time of a session.
        
        Args:
            session_id (str): Unique session identifier
            expiry (timedelta, optional): New expiry time
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current session data
            data = self.get_session(session_id)
            if not data:
                return False
            
            # Set with new expiry
            return self.set_session(session_id, data, expiry)
        except Exception as e:
            print(f"Error updating session expiry: {e}")
            return False

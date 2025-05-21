"""
Redis Pub/Sub messaging for the Financial Crew application.
"""

import os
import json
import threading
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedisPubSub:
    """Redis Pub/Sub messaging for real-time communication."""
    
    def __init__(self):
        """Initialize Redis connection using environment variables."""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", ""),
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
        self.subscribers = {}
        self.running = False
        self.thread = None
    
    def publish(self, channel, message):
        """
        Publish a message to a channel.
        
        Args:
            channel (str): Channel name
            message (dict): Message to publish
        
        Returns:
            int: Number of clients that received the message
        """
        try:
            # Convert message to JSON string
            json_message = json.dumps(message)
            
            # Publish to channel
            return self.redis_client.publish(channel, json_message)
        except Exception as e:
            print(f"Error publishing message: {e}")
            return 0
    
    def subscribe(self, channel, callback):
        """
        Subscribe to a channel.
        
        Args:
            channel (str): Channel name
            callback (function): Callback function to handle messages
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Subscribe to channel
            self.pubsub.subscribe(**{channel: self._message_handler})
            
            # Store callback
            self.subscribers[channel] = callback
            
            # Start listening thread if not already running
            if not self.running:
                self._start_listener()
            
            return True
        except Exception as e:
            print(f"Error subscribing to channel: {e}")
            return False
    
    def unsubscribe(self, channel):
        """
        Unsubscribe from a channel.
        
        Args:
            channel (str): Channel name
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Unsubscribe from channel
            self.pubsub.unsubscribe(channel)
            
            # Remove callback
            if channel in self.subscribers:
                del self.subscribers[channel]
            
            # Stop listener if no more subscribers
            if not self.subscribers and self.running:
                self._stop_listener()
            
            return True
        except Exception as e:
            print(f"Error unsubscribing from channel: {e}")
            return False
    
    def _message_handler(self, message):
        """
        Handle incoming messages.
        
        Args:
            message (dict): Redis message
        """
        try:
            # Skip non-message events
            if message['type'] != 'message':
                return
            
            # Get channel and data
            channel = message['channel']
            data = json.loads(message['data'])
            
            # Call subscriber callback
            if channel in self.subscribers:
                self.subscribers[channel](data)
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def _start_listener(self):
        """Start the message listener thread."""
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()
    
    def _stop_listener(self):
        """Stop the message listener thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
    
    def _listen(self):
        """Listen for messages in a separate thread."""
        while self.running:
            try:
                # Get message with timeout to allow for clean shutdown
                self.pubsub.get_message(timeout=0.1)
            except Exception as e:
                print(f"Error in listener thread: {e}")
                # Small delay to prevent CPU spinning on errors
                import time
                time.sleep(0.1)
    
    def close(self):
        """Close the Redis connection and stop the listener."""
        try:
            self._stop_listener()
            self.pubsub.close()
            self.redis_client.close()
        except Exception as e:
            print(f"Error closing Redis connection: {e}")

"""
MongoDB client for the Financial Simulation system.
This module provides functions to connect to MongoDB Atlas and interact with the database.
If MongoDB is not available, it falls back to a file-based mock implementation.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import pathlib

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Atlas connection strings
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_URI_FALLBACK = os.getenv("MONGODB_URI_FALLBACK")

# Flag to determine if we're using real MongoDB or mock
USE_MOCK_DB = False

# Check if MongoDB URI is available
if not MONGODB_URI and not MONGODB_URI_FALLBACK:
    # Use a mock MongoDB implementation
    USE_MOCK_DB = True
    print(f"⚠️ No MongoDB URI found in environment variables")
    print(f"⚠️ Using mock MongoDB implementation with file storage")

    # Create mock DB directories
    os.makedirs("mock_db/user_inputs", exist_ok=True)
    os.makedirs("mock_db/agent_outputs", exist_ok=True)
    os.makedirs("mock_db/chat_history", exist_ok=True)
    os.makedirs("mock_db/pdf_content", exist_ok=True)
else:
    print(f"📊 MongoDB URI found. Will attempt connection.")

# Database and collection names
DB_NAME = "financial_simulation"
USER_INPUTS_COLLECTION = "user_inputs"
AGENT_OUTPUTS_COLLECTION = "agent_outputs"
CHAT_HISTORY_COLLECTION = "chat_history"

# MongoDB client instance
_client: Optional[MongoClient] = None

def get_client() -> MongoClient:
    """Get MongoDB client instance."""
    global _client, USE_MOCK_DB
    if not USE_MOCK_DB:
        if _client is None:
            # Try primary connection string first
            if MONGODB_URI:
                try:
                    print(f"🔌 Connecting to MongoDB using primary URI...")
                    # Create client with simplified settings
                    _client = MongoClient(
                        MONGODB_URI,
                        serverSelectionTimeoutMS=5000,  # Fail fast if can't connect
                        connectTimeoutMS=5000,
                        socketTimeoutMS=5000
                    )

                    # Test connection
                    _client.server_info()
                    print(f"✅ Successfully connected to MongoDB using primary URI")
                    return _client
                except Exception as e:
                    print(f"⚠️ Primary MongoDB connection failed: {e}")
                    print(f"⚠️ Trying fallback connection string...")

            # Try fallback connection string
            if MONGODB_URI_FALLBACK:
                try:
                    print(f"🔌 Connecting to MongoDB using fallback URI...")
                    # Create client with simplified settings
                    _client = MongoClient(
                        MONGODB_URI_FALLBACK,
                        serverSelectionTimeoutMS=5000,
                        connectTimeoutMS=5000,
                        socketTimeoutMS=5000
                    )

                    # Test connection
                    _client.server_info()
                    print(f"✅ Successfully connected to MongoDB using fallback URI")
                    return _client
                except Exception as e:
                    print(f"⚠️ Fallback MongoDB connection failed: {e}")

            # If we get here, both connection attempts failed
            print("⚠️ All MongoDB connection attempts failed. Falling back to mock implementation")
            USE_MOCK_DB = True
            # Create mock DB directories
            os.makedirs("mock_db/user_inputs", exist_ok=True)
            os.makedirs("mock_db/agent_outputs", exist_ok=True)
            os.makedirs("mock_db/chat_history", exist_ok=True)
            os.makedirs("mock_db/pdf_content", exist_ok=True)
            print("✅ Created mock database directories for local storage")
    return _client

def get_database() -> Database:
    """Get MongoDB database instance."""
    if not USE_MOCK_DB:
        client = get_client()
        return client[DB_NAME]
    return None

def get_user_inputs_collection() -> Collection:
    """Get user inputs collection."""
    if not USE_MOCK_DB:
        db = get_database()
        return db[USER_INPUTS_COLLECTION]
    return None

def get_agent_outputs_collection() -> Collection:
    """Get agent outputs collection."""
    if not USE_MOCK_DB:
        db = get_database()
        return db[AGENT_OUTPUTS_COLLECTION]
    return None

def get_chat_history_collection() -> Collection:
    """Get chat history collection."""
    if not USE_MOCK_DB:
        db = get_database()
        return db[CHAT_HISTORY_COLLECTION]
    return None

def close_connection():
    """Close MongoDB connection."""
    global _client
    if not USE_MOCK_DB and _client is not None:
        _client.close()
        _client = None

def save_user_input(user_input: Dict[str, Any], simulation_id: str) -> str:
    """
    Save user input to MongoDB or mock storage.

    Args:
        user_input: User input data
        simulation_id: Simulation ID

    Returns:
        ID of the inserted document
    """
    # Add metadata
    document = {
        "user_id": user_input.get("user_id", "default_user"),
        "simulation_id": simulation_id,
        "timestamp": datetime.now().isoformat(),
        "data": user_input
    }

    if not USE_MOCK_DB:
        # Use real MongoDB
        collection = get_user_inputs_collection()
        result = collection.insert_one(document)
        return str(result.inserted_id)
    else:
        # Use mock implementation with file storage
        doc_id = str(uuid.uuid4())
        document["_id"] = doc_id

        # Save to file
        file_path = f"mock_db/user_inputs/{doc_id}.json"
        with open(file_path, "w") as f:
            json.dump(document, f, indent=2)

        print(f"💾 Saved user input to mock DB: {file_path}")
        return doc_id

def save_agent_output(
    user_id: str,
    simulation_id: str,
    month: int,
    agent_name: str,
    output_data: Dict[str, Any]
) -> str:
    """
    Save agent output to MongoDB or mock storage.

    Args:
        user_id: User ID
        simulation_id: Simulation ID
        month: Month number
        agent_name: Name of the agent
        output_data: Agent output data

    Returns:
        ID of the inserted document
    """
    # Add metadata
    document = {
        "user_id": user_id,
        "simulation_id": simulation_id,
        "month": month,
        "agent_name": agent_name,
        "timestamp": datetime.now().isoformat(),
        "data": output_data
    }

    if not USE_MOCK_DB:
        # Use real MongoDB
        collection = get_agent_outputs_collection()
        result = collection.insert_one(document)
        return str(result.inserted_id)
    else:
        # Use mock implementation with file storage
        doc_id = str(uuid.uuid4())
        document["_id"] = doc_id

        # Save to file
        file_path = f"mock_db/agent_outputs/{doc_id}.json"
        with open(file_path, "w") as f:
            json.dump(document, f, indent=2)

        print(f"💾 Saved agent output to mock DB: {file_path}")
        return doc_id

def get_agent_outputs_for_month(
    user_id: str,
    month: int,
    agent_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get agent outputs for a specific month from MongoDB or mock storage.

    Args:
        user_id: User ID
        month: Month number
        agent_name: Optional agent name to filter by

    Returns:
        List of agent outputs
    """
    if not USE_MOCK_DB:
        # Use real MongoDB
        collection = get_agent_outputs_collection()

        # Build query
        query = {
            "user_id": user_id,
            "month": month
        }

        if agent_name:
            query["agent_name"] = agent_name

        # Execute query
        results = list(collection.find(query).sort("timestamp", pymongo.DESCENDING))

        # Convert ObjectId to string for JSON serialization
        for result in results:
            result["_id"] = str(result["_id"])

        return results
    else:
        # Use mock implementation with file storage
        results = []

        # Get all files in the agent_outputs directory
        agent_output_dir = "mock_db/agent_outputs"
        if os.path.exists(agent_output_dir):
            for filename in os.listdir(agent_output_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(agent_output_dir, filename)

                    try:
                        with open(file_path, "r") as f:
                            document = json.load(f)

                            # Check if document matches the query
                            if (document.get("user_id") == user_id and
                                document.get("month") == month and
                                (agent_name is None or document.get("agent_name") == agent_name)):
                                results.append(document)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        # Sort by timestamp (descending)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

def get_previous_month_outputs(
    user_id: str,
    current_month: int,
    agent_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get agent outputs from the previous month.

    Args:
        user_id: User ID
        current_month: Current month number
        agent_name: Optional agent name to filter by

    Returns:
        List of agent outputs from the previous month
    """
    if current_month <= 1:
        return []  # No previous month for month 1

    previous_month = current_month - 1
    return get_agent_outputs_for_month(user_id, previous_month, agent_name)

def get_all_agent_outputs_for_user(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all agent outputs for a user from MongoDB or mock storage.

    Args:
        user_id: User ID

    Returns:
        List of all agent outputs for the user
    """
    if not USE_MOCK_DB:
        # Use real MongoDB
        collection = get_agent_outputs_collection()

        # Execute query
        results = list(collection.find({"user_id": user_id}).sort("month", pymongo.ASCENDING))

        # Convert ObjectId to string for JSON serialization
        for result in results:
            result["_id"] = str(result["_id"])

        return results
    else:
        # Use mock implementation with file storage
        results = []

        # Get all files in the agent_outputs directory
        agent_output_dir = "mock_db/agent_outputs"
        if os.path.exists(agent_output_dir):
            for filename in os.listdir(agent_output_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(agent_output_dir, filename)

                    try:
                        with open(file_path, "r") as f:
                            document = json.load(f)

                            # Check if document matches the query
                            if document.get("user_id") == user_id:
                                results.append(document)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        # Sort by month (ascending)
        results.sort(key=lambda x: x.get("month", 0))
        return results

def generate_simulation_id() -> str:
    """Generate a unique simulation ID."""
    return str(uuid.uuid4())

def save_chat_message(user_id: str, role: str, content: str) -> str:
    """
    Save a chat message to MongoDB or mock storage.

    Args:
        user_id: User ID
        role: Message role (user or assistant)
        content: Message content

    Returns:
        ID of the saved message
    """
    # Create document
    document = {
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }

    if not USE_MOCK_DB:
        # Use real MongoDB
        collection = get_chat_history_collection()
        result = collection.insert_one(document)
        return str(result.inserted_id)
    else:
        # Use mock implementation with file storage
        doc_id = str(uuid.uuid4())
        document["_id"] = doc_id

        # Save to file
        file_path = f"mock_db/chat_history/{doc_id}.json"
        os.makedirs("mock_db/chat_history", exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(document, f, indent=2)

        print(f"💾 Saved chat message to mock DB: {file_path}")
        return doc_id

def get_chat_history_for_user(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get chat history for a user from MongoDB or mock storage.

    Args:
        user_id: User ID
        limit: Maximum number of messages to return

    Returns:
        List of chat messages
    """
    if not USE_MOCK_DB:
        # Use real MongoDB
        collection = get_chat_history_collection()

        # Get the most recent messages by sorting by timestamp in descending order
        cursor = collection.find(
            {"user_id": user_id},
            sort=[("timestamp", pymongo.DESCENDING)],
            limit=limit
        )

        # Convert ObjectId to string for JSON serialization
        results = list(cursor)
        for result in results:
            result["_id"] = str(result["_id"])

        # Reverse the results to get them in chronological order (oldest first)
        results.reverse()

        print(f"📊 Retrieved {len(results)} chat messages from MongoDB for user {user_id}")
        if results:
            print(f"📊 Most recent message: {results[-1].get('content', '')[:50]}...")

        return results
    else:
        # Use mock implementation with file storage
        chat_dir = "mock_db/chat_history"
        if not os.path.exists(chat_dir):
            return []

        # Get all chat files
        chat_files = []
        for filename in os.listdir(chat_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(chat_dir, filename)
                with open(file_path, "r") as f:
                    chat_data = json.load(f)
                    if chat_data.get("user_id") == user_id:
                        chat_files.append(chat_data)

        # Sort by timestamp (newest first)
        chat_files.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Take the most recent messages
        recent_messages = chat_files[:limit]

        # Reverse to get chronological order (oldest first)
        recent_messages.reverse()

        print(f"📊 Retrieved {len(recent_messages)} chat messages from mock DB for user {user_id}")
        if recent_messages:
            print(f"📊 Most recent message: {recent_messages[-1].get('content', '')[:50]}...")

        return recent_messages

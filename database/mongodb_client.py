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

# MongoDB Atlas connection string
MONGODB_URI = os.getenv("MONGODB_URI")

# Flag to determine if we're using real MongoDB or mock
USE_MOCK_DB = False

# Check if MongoDB URI is available
if not MONGODB_URI:
    # Use a mock MongoDB implementation
    USE_MOCK_DB = True
    print(f"⚠️ No MongoDB URI found in environment variables")
    print(f"⚠️ Using mock MongoDB implementation with file storage")

    # Create mock DB directories
    os.makedirs("mock_db/user_inputs", exist_ok=True)
    os.makedirs("mock_db/agent_outputs", exist_ok=True)
else:
    print(f"📊 MongoDB URI found. Will attempt connection with enhanced security options.")

# Database and collection names
DB_NAME = "financial_simulation"
USER_INPUTS_COLLECTION = "user_inputs"
AGENT_OUTPUTS_COLLECTION = "agent_outputs"

# MongoDB client instance
_client: Optional[MongoClient] = None

def get_client() -> MongoClient:
    """Get MongoDB client instance."""
    global _client, USE_MOCK_DB
    if not USE_MOCK_DB:
        if _client is None:
            try:
                # Add explicit TLS options to the connection
                if MONGODB_URI and "?" in MONGODB_URI:
                    # If URI already has query parameters, append new ones
                    connection_uri = f"{MONGODB_URI}&tls=true&tlsAllowInvalidCertificates=false&retryWrites=true&w=majority"
                elif MONGODB_URI:
                    # If URI has no query parameters, add them with ?
                    connection_uri = f"{MONGODB_URI}?tls=true&tlsAllowInvalidCertificates=false&retryWrites=true&w=majority"
                else:
                    connection_uri = MONGODB_URI

                print(f"🔌 Connecting to MongoDB with enhanced TLS options...")
                _client = MongoClient(connection_uri)
                # Test connection
                _client.server_info()
                print(f"✅ Successfully connected to MongoDB")
            except Exception as e:
                print(f"⚠️ MongoDB connection failed: {e}")
                print("⚠️ Falling back to mock implementation")
                USE_MOCK_DB = True
                # Create mock DB directories
                os.makedirs("mock_db/user_inputs", exist_ok=True)
                os.makedirs("mock_db/agent_outputs", exist_ok=True)
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

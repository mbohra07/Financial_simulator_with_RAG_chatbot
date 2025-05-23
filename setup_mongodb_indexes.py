"""
Script to set up MongoDB Atlas Vector Search indexes for the Financial Guru application.
This script creates the necessary vector search indexes with proper field configurations.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

# Load environment variables
load_dotenv()

# MongoDB Atlas connection strings
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_URI_FALLBACK = os.getenv("MONGODB_URI_FALLBACK")

# Database name
DB_NAME = "financial_simulation"

def get_mongodb_client() -> Optional[MongoClient]:
    """Get MongoDB client instance."""
    if MONGODB_URI:
        try:
            print(f"🔌 Connecting to MongoDB using primary URI...")
            client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test connection
            client.server_info()
            print(f"✅ Successfully connected to MongoDB using primary URI")
            return client
        except Exception as e:
            print(f"⚠️ Primary MongoDB connection failed: {e}")
            print(f"⚠️ Trying fallback connection string...")

    if MONGODB_URI_FALLBACK:
        try:
            print(f"🔌 Connecting to MongoDB using fallback URI...")
            client = MongoClient(
                MONGODB_URI_FALLBACK,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test connection
            client.server_info()
            print(f"✅ Successfully connected to MongoDB using fallback URI")
            return client
        except Exception as e:
            print(f"⚠️ Fallback MongoDB connection failed: {e}")

    print("❌ All MongoDB connection attempts failed.")
    return None

def create_pdf_vector_index(db: Database) -> bool:
    """
    Create or update the pdf_vector_index for the pdf_vectors collection.
    This index is used for vector similarity search on PDF content.

    Args:
        db: MongoDB database instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        collection = db["pdf_vectors"]

        # Check if index already exists
        existing_indexes = collection.list_search_indexes()
        index_exists = False
        for idx in existing_indexes:
            if idx.get("name") == "pdf_vector_index":
                print(f"ℹ️ pdf_vector_index already exists. Will update it...")
                index_exists = True
                break

        # Define the index configuration
        index_config = {
            "name": "pdf_vector_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "dimensions": 768,  # Dimensions for sentence-transformers/all-mpnet-base-v2
                            "similarity": "cosine",
                            "type": "knnVector"
                        }
                    }
                }
            }
        }

        # Create or update the index
        if index_exists:
            try:
                # Update existing index
                print(f"🔄 Updating pdf_vector_index...")
                collection.update_search_index("pdf_vector_index", index_config["definition"])
                print(f"✅ Successfully updated pdf_vector_index")
            except Exception as update_error:
                # If update fails (not supported in all MongoDB versions), drop and recreate
                print(f"⚠️ Could not update index: {update_error}")
                print(f"🔄 Dropping and recreating pdf_vector_index...")
                collection.drop_search_index("pdf_vector_index")
                result = collection.create_search_index(index_config)
                print(f"✅ Successfully recreated pdf_vector_index: {result}")
        else:
            # Create new index
            print(f"🔍 Creating pdf_vector_index...")
            result = collection.create_search_index(index_config)
            print(f"✅ Successfully created pdf_vector_index: {result}")

        # Wait for index to be ready
        print(f"⏳ Waiting for index to be ready...")
        max_wait_time = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            indexes = collection.list_search_indexes()
            for idx in indexes:
                if idx.get("name") == "pdf_vector_index" and idx.get("status") == "READY":
                    print(f"✅ pdf_vector_index is now ready")
                    return True
            print(f"⏳ Index still being built, waiting...")
            time.sleep(5)

        print(f"⚠️ Index creation/update may still be in progress. Check Atlas UI for status.")
        return True
    except Exception as e:
        print(f"❌ Error creating pdf_vector_index: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_financial_knowledge_index(db: Database) -> bool:
    """
    Create or update the financial_knowledge_index for the financial_knowledge collection.
    This index is used for vector similarity search on financial knowledge content.

    Args:
        db: MongoDB database instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        collection = db["financial_knowledge"]

        # Check if index already exists
        existing_indexes = collection.list_search_indexes()
        index_exists = False
        for idx in existing_indexes:
            if idx.get("name") == "financial_knowledge_index":
                print(f"ℹ️ financial_knowledge_index already exists. Will update it...")
                index_exists = True
                break

        # Define the index configuration
        index_config = {
            "name": "financial_knowledge_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "dimensions": 768,  # Dimensions for sentence-transformers/all-mpnet-base-v2
                            "similarity": "cosine",
                            "type": "knnVector"
                        },
                        "title": {
                            "type": "string"  # Index title as string for text search
                        },
                        "doc_id": {
                            "type": "token"  # Index doc_id as token for filtering
                        }
                    }
                }
            }
        }

        # Create or update the index
        if index_exists:
            try:
                # Update existing index
                print(f"🔄 Updating financial_knowledge_index...")
                collection.update_search_index("financial_knowledge_index", index_config["definition"])
                print(f"✅ Successfully updated financial_knowledge_index")
            except Exception as update_error:
                # If update fails (not supported in all MongoDB versions), drop and recreate
                print(f"⚠️ Could not update index: {update_error}")
                print(f"🔄 Dropping and recreating financial_knowledge_index...")
                collection.drop_search_index("financial_knowledge_index")
                result = collection.create_search_index(index_config)
                print(f"✅ Successfully recreated financial_knowledge_index: {result}")
        else:
            # Create new index
            print(f"🔍 Creating financial_knowledge_index...")
            result = collection.create_search_index(index_config)
            print(f"✅ Successfully created financial_knowledge_index: {result}")

        # Wait for index to be ready
        print(f"⏳ Waiting for index to be ready...")
        max_wait_time = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            indexes = collection.list_search_indexes()
            for idx in indexes:
                if idx.get("name") == "financial_knowledge_index" and idx.get("status") == "READY":
                    print(f"✅ financial_knowledge_index is now ready")
                    return True
            print(f"⏳ Index still being built, waiting...")
            time.sleep(5)

        print(f"⚠️ Index creation/update may still be in progress. Check Atlas UI for status.")
        return True
    except Exception as e:
        print(f"❌ Error creating financial_knowledge_index: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to set up MongoDB Atlas Vector Search indexes."""
    print("🚀 Setting up MongoDB Atlas Vector Search indexes...")

    # Get MongoDB client
    client = get_mongodb_client()
    if not client:
        print("❌ Failed to connect to MongoDB. Cannot set up indexes.")
        return

    try:
        # Get database
        db = client[DB_NAME]

        # Create indexes
        pdf_index_success = create_pdf_vector_index(db)
        financial_index_success = create_financial_knowledge_index(db)

        if pdf_index_success and financial_index_success:
            print("✅ Successfully set up all MongoDB Atlas Vector Search indexes")
        else:
            print("⚠️ Some indexes may not have been created successfully")

        # Print instructions for manual setup if needed
        print("\n📝 If you encounter issues with the indexes, you can create them manually in the MongoDB Atlas UI:")
        print("1. Go to your MongoDB Atlas cluster")
        print("2. Navigate to the 'Search' tab")
        print("3. Create a new index on the 'pdf_vectors' collection named 'pdf_vector_index'")
        print("   - Configure it to index the 'embedding' field as a vector with 768 dimensions")
        print("   - Add 'metadata.user_id', 'metadata.pdf_id', and 'metadata.chunk_id' as token fields")
        print("4. Create a new index on the 'financial_knowledge' collection named 'financial_knowledge_index'")
        print("   - Configure it to index the 'embedding' field as a vector with 768 dimensions")
        print("   - Add 'doc_id' as a token field and 'title' as a string field")
    except Exception as e:
        print(f"❌ Error setting up MongoDB Atlas Vector Search indexes: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close client connection
        client.close()
        print("🔌 MongoDB connection closed")

if __name__ == "__main__":
    main()

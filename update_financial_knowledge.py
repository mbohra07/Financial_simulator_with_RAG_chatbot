"""
Script to update the financial_knowledge collection to add required fields for vector search.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# MongoDB Atlas connection strings
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_URI_FALLBACK = os.getenv("MONGODB_URI_FALLBACK")

# Database name
DB_NAME = "financial_simulation"

def get_mongodb_client():
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

def update_financial_knowledge_collection(db):
    """Update the financial_knowledge collection to add required fields."""
    print("\n🔄 Updating financial_knowledge collection...")
    
    collection = db["financial_knowledge"]
    
    # Count documents
    doc_count = collection.count_documents({})
    print(f"📊 Found {doc_count} documents in financial_knowledge collection")
    
    if doc_count == 0:
        print("⚠️ No documents found in financial_knowledge collection.")
        return
    
    # Update all documents to add doc_id and title fields
    updated_count = 0
    for doc in collection.find({}):
        # Check if doc_id already exists
        if "doc_id" not in doc:
            # Generate a doc_id
            doc_id = f"concept_{str(uuid.uuid4())[:8]}"
            
            # Extract title from metadata if available
            title = "Financial Concept"
            if "metadata" in doc and "title" in doc["metadata"]:
                title = doc["metadata"]["title"]
            
            # Update the document
            result = collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "doc_id": doc_id,
                    "title": title
                }}
            )
            
            if result.modified_count > 0:
                updated_count += 1
    
    print(f"✅ Updated {updated_count} documents in financial_knowledge collection")

def main():
    """Main function to update MongoDB collections."""
    print("🚀 Updating MongoDB collections for vector search compatibility...")
    
    # Get MongoDB client
    client = get_mongodb_client()
    if client is None:
        print("❌ Failed to connect to MongoDB. Cannot update collections.")
        return
    
    try:
        # Get database
        db = client[DB_NAME]
        
        # Update collections
        update_financial_knowledge_collection(db)
        
        print("\n✅ Successfully updated MongoDB collections")
        
    except Exception as e:
        print(f"❌ Error updating MongoDB collections: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close client connection
        client.close()
        print("\n🔌 MongoDB connection closed")

if __name__ == "__main__":
    main()

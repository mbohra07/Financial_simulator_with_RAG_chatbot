"""
Example script demonstrating how to use MongoDB Atlas Vector Search with LangChain.
This script shows the correct way to use the MongoDBAtlasVectorSearch class.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import the updated MongoDB Atlas Vector Search
try:
    from langchain_mongodb import MongoDBAtlasVectorSearch
    print("✅ Using updated MongoDB Atlas Vector Search from langchain_mongodb")
except ImportError:
    # Fall back to the deprecated version
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch
    print("⚠️ Using deprecated MongoDB Atlas Vector Search from langchain_community")

# Try to import the updated HuggingFaceEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Using updated HuggingFaceEmbeddings from langchain_huggingface")
except ImportError:
    # Fall back to the deprecated version
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("⚠️ Using deprecated HuggingFaceEmbeddings from langchain_community")

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

def test_pdf_vector_search():
    """Test the pdf_vector_index for vector search."""
    print("\n🔍 Testing PDF Vector Search...")
    
    # Get MongoDB client
    client = get_mongodb_client()
    if client is None:
        print("❌ Failed to connect to MongoDB. Cannot test vector search.")
        return False
    
    try:
        # Get database and collection
        db = client[DB_NAME]
        collection = db["pdf_vectors"]
        
        # Count documents
        doc_count = collection.count_documents({})
        print(f"📊 Found {doc_count} documents in pdf_vectors collection")
        
        if doc_count == 0:
            print("⚠️ No documents found in pdf_vectors collection. Skipping test.")
            return False
        
        # Get a sample user_id from the collection
        sample_doc = collection.find_one({})
        if not sample_doc or "metadata" not in sample_doc or "user_id" not in sample_doc["metadata"]:
            print("⚠️ No valid documents found with metadata.user_id. Skipping test.")
            return False
        
        user_id = sample_doc["metadata"]["user_id"]
        print(f"📝 Using user_id: {user_id} for test")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Create vector store
        print("🔍 Creating vector store with pdf_vector_index...")
        vector_store = MongoDBAtlasVectorSearch(
            collection,
            embeddings,
            index_name="pdf_vector_index"
        )
        
        # Test query
        query = "financial planning"
        print(f"🔍 Testing query: '{query}'")
        
        # Perform search
        results = vector_store.similarity_search(query, k=10)
        
        # Filter results manually
        filtered_results = []
        for doc in results:
            if doc.metadata.get("user_id") == user_id:
                filtered_results.append(doc)
        
        # Take only the top 2 results
        filtered_results = filtered_results[:2]
        
        # Check results
        if filtered_results:
            print(f"✅ Vector search successful! Found {len(filtered_results)} results after filtering")
            for i, result in enumerate(filtered_results):
                print(f"\n📄 Result {i+1}:")
                print(f"Content: {result.page_content[:100]}...")
            return True
        else:
            print("⚠️ Vector search returned no results after filtering")
            
            # Print the original results
            if results:
                print(f"ℹ️ Original search returned {len(results)} results before filtering")
                for i, result in enumerate(results[:2]):
                    print(f"\n📄 Original Result {i+1}:")
                    print(f"User ID: {result.metadata.get('user_id', 'N/A')}")
                    print(f"Content: {result.page_content[:100]}...")
            
            return False
    except Exception as e:
        print(f"❌ Error testing PDF vector search: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Close client connection
        client.close()

def test_financial_knowledge_search():
    """Test the financial_knowledge_index for vector search."""
    print("\n🔍 Testing Financial Knowledge Vector Search...")
    
    # Get MongoDB client
    client = get_mongodb_client()
    if client is None:
        print("❌ Failed to connect to MongoDB. Cannot test vector search.")
        return False
    
    try:
        # Get database and collection
        db = client[DB_NAME]
        collection = db["financial_knowledge"]
        
        # Count documents
        doc_count = collection.count_documents({})
        print(f"📊 Found {doc_count} documents in financial_knowledge collection")
        
        if doc_count == 0:
            print("⚠️ No documents found in financial_knowledge collection. Skipping test.")
            return False
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Create vector store
        print("🔍 Creating vector store with financial_knowledge_index...")
        vector_store = MongoDBAtlasVectorSearch(
            collection,
            embeddings,
            index_name="financial_knowledge_index"
        )
        
        # Test query
        query = "budgeting tips"
        print(f"🔍 Testing query: '{query}'")
        
        # Perform search
        results = vector_store.similarity_search(query, k=2)
        
        # Check results
        if results:
            print(f"✅ Vector search successful! Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"\n📄 Result {i+1}:")
                print(f"Title: {result.metadata.get('title', 'N/A')}")
                print(f"Content: {result.page_content[:100]}...")
            return True
        else:
            print("⚠️ Vector search returned no results")
            return False
    except Exception as e:
        print(f"❌ Error testing financial knowledge vector search: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Close client connection
        client.close()

def main():
    """Main function to test MongoDB Atlas Vector Search with LangChain."""
    print("🚀 Testing MongoDB Atlas Vector Search with LangChain...")
    
    # Test vector search
    pdf_search_success = test_pdf_vector_search()
    financial_search_success = test_financial_knowledge_search()
    
    # Print summary
    print("\n📋 Test Summary:")
    print(f"PDF Vector Search: {'✅ Passed' if pdf_search_success else '❌ Failed'}")
    print(f"Financial Knowledge Search: {'✅ Passed' if financial_search_success else '❌ Failed'}")
    
    if pdf_search_success and financial_search_success:
        print("\n🎉 All tests passed! MongoDB Atlas Vector Search with LangChain is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the logs for details.")

if __name__ == "__main__":
    main()

"""
Teacher Agent implementation using LangGraph.
This agent explains financial concepts in simple terms and can incorporate PDF content.
"""

from typing import Dict, List, Any, TypedDict, Annotated, Literal, Optional, Union, Set
import json
import os
from datetime import datetime
import time
import uuid
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain_community.document_loaders import PyPDFLoader
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
from langchain.text_splitter import RecursiveCharacterTextSplitter

import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
from pymongo import MongoClient

# Import MongoDB client
from database.mongodb_client import (
    get_client,
    get_database,
    USE_MOCK_DB
)

# Load environment variables
load_dotenv()

# Initialize LLM
def get_llm(model_name="groq/llama3-70b-8192"):
    """Get the LLM based on model name."""
    if model_name.startswith("groq/"):
        return ChatGroq(
            model_name=model_name.replace("groq/", ""),
            temperature=0.2,
            max_tokens=4000
        )
    elif model_name.startswith("openai/"):
        return ChatOpenAI(
            model_name=model_name.replace("openai/", ""),
            temperature=0.2,
            max_tokens=4000
        )
    else:
        # Default to Groq with Llama 3
        return ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2,
            max_tokens=4000
        )

# Define state schema
class TeacherAgentState(TypedDict):
    """State for the teacher agent workflow."""
    # Input data
    user_query: str
    chat_history: List[Dict[str, str]]
    user_id: str

    # PDF data
    pdf_path: Optional[str]
    pdf_id: Optional[Union[str, List[str]]]  # PDF ID(s) for specific PDF searches
    pdf_content: Optional[List[Document]]
    pdf_metadata: Optional[Dict[str, Any]]  # Metadata for the current PDF

    # Vector search results
    vector_search_results: Optional[List[Document]]

    # Response
    response: Optional[str]

# PDF Processing Functions
def process_pdf(pdf_path: str) -> List[Document]:
    """Process a PDF file and return a list of documents."""
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)

        return split_docs
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

def vectorize_and_store_pdf(documents: List[Document], user_id: str, pdf_name: str) -> str:
    """
    Vectorize and store PDF content in MongoDB Atlas.

    Args:
        documents: List of document chunks from the PDF
        user_id: User identifier
        pdf_name: Name of the PDF file

    Returns:
        PDF ID if successful, raises ValueError otherwise
    """
    try:
        # Import numpy - required for vector operations
        import numpy as np
        print("✅ NumPy is available, version:", np.__version__)

        # Generate a unique PDF ID
        pdf_id = f"pdf_{user_id}_{str(uuid.uuid4())[:8]}"

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("❌ MongoDB connection not available. Vector database requires MongoDB Atlas.")
            raise ValueError("MongoDB connection not available. Cannot proceed with vector storage.")

        # Get MongoDB database
        db = get_database()
        collection = db["pdf_vectors"]

        # Store PDF metadata
        pdf_metadata = {
            "pdf_id": pdf_id,
            "user_id": user_id,
            "pdf_name": pdf_name,
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(documents)
        }

        # Store in PDF metadata collection
        db["pdf_metadata"].insert_one(pdf_metadata)

        # Add user_id and pdf_id to metadata for each document before vectorization
        for i, doc in enumerate(documents):
            doc.metadata["user_id"] = user_id
            doc.metadata["pdf_id"] = pdf_id
            doc.metadata["chunk_id"] = f"{pdf_id}_{i}"
            doc.metadata["pdf_name"] = pdf_name

        # Direct insertion with embeddings - most reliable method
        print(f"🔄 Directly inserting {len(documents)} documents with embeddings...")

        # Create a list to store successful insertions
        successful_insertions = 0

        # Process documents in batches to avoid overwhelming the embedding model
        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}...")

            try:
                # Get embeddings for the batch
                texts = [doc.page_content for doc in batch]
                batch_embeddings = embeddings.embed_documents(texts)

                # Insert documents with embeddings
                for j, doc in enumerate(batch):
                    try:
                        # Create document with embedding
                        vector_doc = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                            "embedding": batch_embeddings[j]
                        }

                        # Insert into MongoDB
                        collection.insert_one(vector_doc)
                        successful_insertions += 1
                    except Exception as e2:
                        print(f"❌ Error inserting document {i+j}: {e2}")
            except Exception as e:
                print(f"❌ Error processing batch: {e}")
                # Process documents individually as a fallback for this batch only
                for j, doc in enumerate(batch):
                    try:
                        # Get embedding for document
                        embedding = embeddings.embed_query(doc.page_content)

                        # Create document with embedding
                        vector_doc = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                            "embedding": embedding
                        }

                        # Insert into MongoDB
                        collection.insert_one(vector_doc)
                        successful_insertions += 1
                    except Exception as e2:
                        print(f"❌ Error inserting document {i+j}: {e2}")

        # Verify documents were stored properly
        count = collection.count_documents({"metadata.pdf_id": pdf_id})
        print(f"✅ Successfully inserted {successful_insertions}/{len(documents)} documents")
        print(f"✅ Verified {count} documents in vector store for PDF ID: {pdf_id}")

        # If we didn't insert any documents successfully, raise an error
        if successful_insertions == 0:
            raise ValueError("Failed to insert any documents into vector database")

        print(f"✅ Successfully stored PDF with ID: {pdf_id}")
        return pdf_id
    except Exception as e:
        print(f"❌ Error vectorizing PDF: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to vectorize and store PDF: {str(e)}")



def search_vector_db(query: str, user_id: str, k: int = 5, pdf_id: Union[str, List[str]] = None) -> List[Document]:
    """
    Search the vector database for relevant documents using direct MongoDB aggregation.

    Args:
        query: The search query
        user_id: User identifier
        k: Number of results to return
        pdf_id: Optional PDF ID or list of PDF IDs to filter by

    Returns:
        List of relevant documents
    """
    try:
        # Import numpy - required for vector operations
        import numpy as np
        print("✅ Using vector search with MongoDB Atlas")

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("❌ MongoDB connection not available. Vector search requires MongoDB Atlas.")
            raise ValueError("MongoDB connection not available. Cannot perform vector search.")

        # Get MongoDB database
        db = get_database()
        collection = db["pdf_vectors"]

        # Build filter
        if pdf_id:
            if isinstance(pdf_id, list):
                # Multiple PDF IDs
                if len(pdf_id) > 0:
                    # Search with user_id and multiple pdf_ids filter
                    filter_dict = {
                        "metadata.user_id": user_id,
                        "metadata.pdf_id": {"$in": pdf_id}
                    }
                    print(f"🔍 Searching for multiple PDF IDs: {', '.join(pdf_id)}")
                else:
                    # Empty list, fall back to user_id filter only
                    filter_dict = {"metadata.user_id": user_id}
                    print(f"🔍 Searching all PDFs for user: {user_id}")
            else:
                # Single PDF ID
                filter_dict = {"metadata.user_id": user_id, "metadata.pdf_id": pdf_id}
                print(f"🔍 Searching for PDF ID: {pdf_id}")
        else:
            # Search with user_id filter only
            filter_dict = {"metadata.user_id": user_id}
            print(f"🔍 Searching all PDFs for user: {user_id}")

        # Check if documents exist for this filter
        doc_count = collection.count_documents(filter_dict)
        print(f"📊 Found {doc_count} documents matching filter criteria")

        if doc_count == 0:
            print("⚠️ No documents found matching filter criteria")
            return []

        # Try direct MongoDB Atlas Vector Search using aggregation pipeline
        try:
            print("🔍 Using MongoDB Atlas Vector Search with direct aggregation...")

            # Generate embedding for query
            query_embedding = embeddings.embed_query(query)

            # Create aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "pdf_vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,  # Get more candidates for better results
                        "limit": k * 2  # Get more results to filter
                    }
                },
                {
                    "$match": filter_dict
                },
                {
                    "$limit": k
                },
                {
                    "$project": {
                        "_id": 0,
                        "page_content": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            # Execute aggregation
            results_cursor = collection.aggregate(pipeline)
            results_list = list(results_cursor)

            print(f"✅ MongoDB Atlas Vector Search completed with {len(results_list)} results")

            # Convert to Document objects
            results = []
            for doc in results_list:
                results.append(Document(
                    page_content=doc.get("page_content", ""),
                    metadata=doc.get("metadata", {})
                ))

            return results

        except Exception as e:
            print(f"⚠️ MongoDB Atlas Vector Search failed: {e}")
            print("🔍 Falling back to direct vector similarity calculation...")

            # Get query embedding
            query_embedding = embeddings.embed_query(query)

            # Get all documents matching filter
            matching_docs = list(collection.find(filter_dict))
            print(f"📄 Retrieved {len(matching_docs)} documents for processing")

            # Calculate cosine similarity manually
            from numpy import dot
            from numpy.linalg import norm

            def cosine_similarity(a, b):
                if not a or not b:
                    return 0
                return dot(a, b) / (norm(a) * norm(b))

            # Score documents based on embedding similarity
            scored_docs = []
            for doc in matching_docs:
                if "embedding" in doc and doc["embedding"]:
                    # Calculate similarity
                    similarity = cosine_similarity(query_embedding, doc["embedding"])

                    # Add to scored docs
                    scored_docs.append((similarity, doc))

            # Sort by similarity (descending)
            scored_docs.sort(reverse=True, key=lambda x: x[0])

            # Take top k results
            top_docs = [doc for _, doc in scored_docs[:k]]

            # Convert to Document objects
            results = []
            for doc in top_docs:
                results.append(Document(
                    page_content=doc.get("page_content", ""),
                    metadata=doc.get("metadata", {})
                ))

            print(f"✅ Found {len(results)} relevant documents using direct vector similarity")
            return results

    except Exception as e:
        print(f"❌ Error searching vector DB: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results instead of falling back to non-vector search
        print("❌ Vector search failed. No fallback to direct content search available.")
        return []



def search_financial_knowledge_base(query: str, k: int = 5) -> List[Document]:
    """Search the general financial knowledge base using MongoDB Atlas Vector Search with direct aggregation."""
    try:
        # Import numpy - required for vector operations
        import numpy as np
        print("✅ Using vector search for financial knowledge base")

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("❌ MongoDB connection not available. Knowledge base search requires MongoDB Atlas.")
            return []

        # Get MongoDB database
        db = get_database()
        collection = db["financial_knowledge"]

        # Try direct MongoDB Atlas Vector Search using aggregation pipeline
        try:
            print("🔍 Using MongoDB Atlas Vector Search with direct aggregation...")

            # Generate embedding for query
            query_embedding = embeddings.embed_query(query)

            # Create aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "financial_knowledge_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,  # Get more candidates for better results
                        "limit": k
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "page_content": 1,
                        "metadata": 1,
                        "title": 1,
                        "doc_id": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            # Execute aggregation
            results_cursor = collection.aggregate(pipeline)
            results_list = list(results_cursor)

            print(f"✅ Found {len(results_list)} relevant documents in financial knowledge base")

            # Convert to Document objects
            results = []
            for doc in results_list:
                # Create metadata dictionary
                metadata = doc.get("metadata", {})

                # Add title and doc_id to metadata if they exist at the top level
                if "title" in doc:
                    metadata["title"] = doc["title"]
                if "doc_id" in doc:
                    metadata["doc_id"] = doc["doc_id"]

                results.append(Document(
                    page_content=doc.get("page_content", ""),
                    metadata=metadata
                ))

            return results
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            import traceback
            traceback.print_exc()
            # Return empty results instead of falling back to non-vector search
            print("❌ Knowledge base vector search failed. No fallback available.")
            return []
    except Exception as e:
        print(f"❌ Error searching knowledge base: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results instead of falling back to non-vector search
        print("❌ Knowledge base vector search failed. No fallback available.")
        return []



def remove_pdf(user_id: str, pdf_id: str = None) -> bool:
    """
    Remove PDF data for a specific user or a specific PDF.

    Args:
        user_id: User identifier
        pdf_id: Optional PDF ID to remove a specific PDF

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, PDF removal not available")
            return False

        # Get MongoDB database
        db = get_database()

        if pdf_id:
            # Remove specific PDF
            print(f"🗑️ Removing PDF with ID: {pdf_id}")

            # Remove from pdf_vectors collection
            vector_result = db["pdf_vectors"].delete_many({"metadata.pdf_id": pdf_id})

            # Remove from pdf_content collection
            content_result = db["pdf_content"].delete_many({"pdf_id": pdf_id})

            # Remove from pdf_metadata collection
            metadata_result = db["pdf_metadata"].delete_one({"pdf_id": pdf_id})

            total_deleted = vector_result.deleted_count + content_result.deleted_count
            if metadata_result.deleted_count > 0:
                total_deleted += 1

            print(f"🗑️ Removed PDF with ID {pdf_id}: {total_deleted} documents deleted")
            return True
        else:
            # Remove all PDFs for user
            print(f"🗑️ Removing all PDFs for user: {user_id}")

            # Get all PDF IDs for this user
            pdf_ids = [doc.get("pdf_id") for doc in db["pdf_metadata"].find({"user_id": user_id})]

            # Remove from pdf_vectors collection
            vector_result = db["pdf_vectors"].delete_many({"metadata.user_id": user_id})

            # Remove from pdf_content collection
            content_result = db["pdf_content"].delete_many({"user_id": user_id})

            # Remove from pdf_metadata collection
            metadata_result = db["pdf_metadata"].delete_many({"user_id": user_id})

            total_deleted = vector_result.deleted_count + content_result.deleted_count + metadata_result.deleted_count

            print(f"🗑️ Removed {len(pdf_ids)} PDFs for user {user_id}: {total_deleted} documents deleted")
            return True
    except Exception as e:
        print(f"❌ Error removing PDF data: {e}")
        import traceback
        traceback.print_exc()
        return False

# Define agent nodes
def retrieve_context_node(state: TeacherAgentState) -> TeacherAgentState:
    """Retrieve relevant context from vector databases."""
    # Extract the current query directly from state to ensure we're using the latest
    current_query = state["user_query"]
    print(f"🔍 Retrieving context for query: '{current_query}'")

    user_id = state["user_id"]
    pdf_id = state.get("pdf_id")  # Get PDF ID(s) if available

    # Log the state to help with debugging
    print(f"📋 Current state - user_id: {user_id}, query: '{current_query}'")
    if pdf_id:
        if isinstance(pdf_id, list):
            print(f"📋 PDF IDs: {', '.join(pdf_id)}")
        else:
            print(f"📋 PDF ID: {pdf_id}")

    # Search PDF vectors if available
    if pdf_id:
        if isinstance(pdf_id, list):
            print(f"🔍 Searching multiple PDFs: {', '.join(pdf_id)}")
        else:
            print(f"🔍 Searching specific PDF: {pdf_id}")

        # Use MongoDB Atlas vector search
        pdf_results = search_vector_db(current_query, user_id, pdf_id=pdf_id)

        if len(pdf_results) == 0:
            print("ℹ️ Vector search returned no results for the specific PDFs")
    else:
        print(f"🔍 Searching all PDFs for user: {user_id}")

        # Use MongoDB Atlas vector search
        pdf_results = search_vector_db(current_query, user_id)

        if len(pdf_results) == 0:
            print("ℹ️ Vector search returned no results across all user PDFs")

    # Search financial knowledge base
    kb_results = search_financial_knowledge_base(current_query)

    # Get PDF metadata if available
    pdf_metadata = None
    if pdf_id and len(pdf_results) > 0:
        try:
            # Get MongoDB database
            db = get_database()
            # Check if db is None explicitly
            if db is not None:
                # Get PDF metadata
                if isinstance(pdf_id, list) and len(pdf_id) > 0:
                    # Use the first PDF ID if multiple are provided
                    metadata = db["pdf_metadata"].find_one({"pdf_id": pdf_id[0]})
                else:
                    metadata = db["pdf_metadata"].find_one({"pdf_id": pdf_id})

                if metadata:
                    pdf_metadata = metadata
                    print(f"✅ Found metadata for PDF: {metadata.get('pdf_name', 'Unknown')}")
        except Exception as e:
            print(f"⚠️ Could not retrieve PDF metadata: {e}")

    # Prioritize PDF results over knowledge base results
    # If we have PDF results, use fewer knowledge base results
    if len(pdf_results) > 0:
        # Use all PDF results and fewer knowledge base results
        all_results = pdf_results + kb_results[:2]  # Limit knowledge base results
    else:
        # If no PDF results, use more knowledge base results
        all_results = kb_results

    print(f"✅ Found {len(pdf_results)} PDF results and {len(kb_results)} knowledge base results for query: '{current_query}'")

    # Update state with both results and metadata
    return {
        **state,
        "vector_search_results": all_results,
        "pdf_metadata": pdf_metadata
    }

def generate_response_node(state: TeacherAgentState) -> TeacherAgentState:
    """Generate a response using the teacher agent."""
    # Extract the current query directly from state to ensure we're using the latest
    current_query = state["user_query"]
    print(f"🧠 Generating response for query: '{current_query}'")

    # Format context from vector search
    context = ""

    # Add PDF metadata if available
    pdf_metadata = state.get("pdf_metadata")
    if pdf_metadata:
        pdf_name = pdf_metadata.get("pdf_name", "Unknown")
        context += f"CURRENT PDF: {pdf_name}\n\n"
        print(f"📄 Added PDF name to context: {pdf_name}")
    else:
        print("⚠️ No PDF metadata available")

    # Add PDF ID if available
    pdf_id = state.get("pdf_id")
    if pdf_id:
        if isinstance(pdf_id, list):
            context += f"PDF ID(s): {', '.join(pdf_id)}\n\n"
            print(f"📄 Added PDF IDs to context: {', '.join(pdf_id)}")
        else:
            context += f"PDF ID: {pdf_id}\n\n"
            print(f"📄 Added PDF ID to context: {pdf_id}")

    # Add vector search results
    vector_results = state.get("vector_search_results")
    print(f"📄 Vector results available: {vector_results is not None}")
    print(f"📄 Vector results count: {len(vector_results) if vector_results else 0}")

    if vector_results and len(vector_results) > 0:
        # Filter for PDF results
        pdf_results = []
        kb_results = []

        for doc in vector_results:
            # Check if this is a PDF result (has metadata.pdf_id)
            if doc.metadata and "pdf_id" in doc.metadata:
                pdf_results.append(doc)
            else:
                kb_results.append(doc)

        print(f"📄 PDF results count: {len(pdf_results)}")
        print(f"📄 KB results count: {len(kb_results)}")

        # Add PDF results first
        if pdf_results:
            context += "CONTENT FROM CURRENT PDF:\n"
            pdf_content = "\n\n".join([doc.page_content for doc in pdf_results])
            context += pdf_content
            context += "\n\n"
            print(f"📄 Added PDF content to context: {pdf_content[:100]}...")
        else:
            print("⚠️ No PDF results to add to context")

        # Add knowledge base results if needed
        if kb_results and (not pdf_results or len(pdf_results) < 2):
            context += "ADDITIONAL FINANCIAL KNOWLEDGE:\n"
            kb_content = "\n\n".join([doc.page_content for doc in kb_results[:2]])
            context += kb_content
            print(f"📄 Added KB content to context: {kb_content[:100]}...")

    # If no context was added, add a note
    if not context.strip():
        context = "No relevant content found in the specified PDF."
        print("⚠️ No context was added, using fallback message")

    # Print the final context for debugging
    print(f"📄 Final context length: {len(context)}")
    print(f"📄 Final context preview: {context[:200]}...")

    # Format chat history - we'll keep it for context but ensure the model prioritizes the current query
    formatted_history = []
    for message in state.get("chat_history", []):
        if message.get("role") == "user":
            formatted_history.append(HumanMessage(content=message.get("content", "")))
        elif message.get("role") == "assistant":
            formatted_history.append(AIMessage(content=message.get("content", "")))

    # Create prompt with explicit query reference and clear instructions
    # Using f-strings to directly insert the query into the prompt
    system_message = SystemMessage(content=f"""You are a friendly and knowledgeable financial teacher.
Your goal is to explain concepts in simple, easy-to-understand language.
Always be supportive, patient, and encouraging. Use analogies and examples to make complex concepts accessible.
When explaining terms, avoid jargon and break down concepts step by step.
If you're not sure about something, be honest about it rather than making up information.

CRITICAL INSTRUCTION: You are answering this specific question: "{current_query}"
You MUST prioritize the context from the CURRENT PDF being queried over any previous conversations.
When the user asks about content in a specific PDF (like a resume or research paper), your answer
should be based PRIMARILY on the content found in that specific PDF.

DO NOT include phrases like "User Query:" in your response.
DO NOT repeat the question in your response.
Just answer the question directly and conversationally.

If you have access to PDF content or knowledge base information, use it to enhance your explanations,
but always maintain a conversational and educational tone.""")

    # Add chat history messages if available
    messages = [system_message]
    if formatted_history:
        messages.extend(formatted_history[-4:])  # Include last 4 messages for context

    # Add the current query as the final message
    human_message_content = f"""
ANSWER THIS QUESTION: {current_query}

Relevant Context from the CURRENT PDF:
{context}

IMPORTANT REMINDER:
1. Answer ONLY the question above: "{current_query}"
2. Base your answer PRIMARILY on the context provided above from the CURRENT PDF
3. If asked about a specific document (like a resume), ONLY mention information that is explicitly in the context
4. Do NOT include "User Query:" in your response
5. Do NOT repeat the question in your response
6. Just answer directly and conversationally
"""
    print(f"📝 Human message content preview: {human_message_content[:200]}...")
    messages.append(HumanMessage(content=human_message_content))

    prompt = ChatPromptTemplate.from_messages(messages)

    # Get LLM
    llm = get_llm("groq/llama3-70b-8192")  # Using Llama 3 for teaching

    # Create chain
    chain = prompt | llm

    # Execute chain
    try:
        # Log the exact query being sent to the LLM
        print(f"📝 Sending query to LLM: '{current_query}'")

        # Debug the context being sent to the LLM
        print(f"📝 Context being sent to LLM (length: {len(context)})")

        # Create a dictionary with the context
        input_dict = {"context": context}

        # Debug the input dictionary
        print(f"📝 Input dictionary keys: {input_dict.keys()}")
        print(f"📝 Context key exists: {'context' in input_dict}")
        print(f"📝 Context value type: {type(input_dict['context'])}")

        # Pass the context - the query and chat history are already in the prompt
        result = chain.invoke(input_dict)

        print(f"✅ LLM response generated for query: '{current_query}'")
        print(f"✅ LLM response: {result.content[:100]}...")

        # Update state
        return {
            **state,
            "response": result.content
        }
    except Exception as e:
        print(f"❌ Error in generate_response_node: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "response": "I'm sorry, I encountered an error while trying to answer your question. Please try again."
        }

# Define the LangGraph workflow
def create_teacher_agent_graph():
    """Create the teacher agent workflow graph."""
    # Create the graph
    workflow = StateGraph(TeacherAgentState)

    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_response", generate_response_node)

    # Define the edges
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)

    # Set the entry point
    workflow.set_entry_point("retrieve_context")

    # Compile the graph
    return workflow.compile()

# Main function to run the teacher agent
def run_teacher_agent(user_query: str, user_id: str, chat_history: List[Dict[str, str]] = None, pdf_id: Union[str, List[str]] = None) -> Dict[str, Any]:
    """
    Run the teacher agent to answer a user query.

    Args:
        user_query: The user's question
        user_id: User identifier
        chat_history: Optional chat history
        pdf_id: Optional PDF ID or list of PDF IDs to search in specific PDF(s)

    Returns:
        Dict with response and updated chat history
    """
    print(f"🚀 Running teacher agent for user {user_id} with query: '{user_query}'")

    # Log PDF ID information if provided
    if pdf_id:
        if isinstance(pdf_id, list):
            print(f"📚 Using multiple PDFs: {', '.join(pdf_id)}")
        else:
            print(f"📚 Using specific PDF: {pdf_id}")

    # Create the workflow graph
    workflow = create_teacher_agent_graph()

    # Initialize chat history if None
    if chat_history is None:
        chat_history = []

    # Make a copy of chat history to avoid modifying the original
    chat_history_copy = chat_history.copy()

    # Print the last few messages from chat history for debugging
    if chat_history_copy:
        print(f"📜 Chat history has {len(chat_history_copy)} messages")
        if len(chat_history_copy) > 0:
            print(f"📜 Last message in chat history - Role: {chat_history_copy[-1].get('role')}, Content: {chat_history_copy[-1].get('content', '')[:50]}...")
            print(f"📜 Current query: '{user_query}'")
    else:
        print(f"📜 No chat history available. Starting fresh conversation with query: '{user_query}'")

    # Initialize state with explicit query
    initial_state = {
        "user_query": user_query.strip(),  # Ensure query is trimmed
        "chat_history": chat_history_copy,
        "user_id": user_id,
        "pdf_path": None,
        "pdf_id": pdf_id,  # Include PDF ID if provided
        "pdf_content": None,
        "pdf_metadata": None,  # Will be populated during context retrieval
        "vector_search_results": None,
        "response": None
    }

    print(f"🔄 Initializing workflow with query: '{user_query}'")

    # Run the workflow
    try:
        # Execute the workflow
        result = workflow.invoke(initial_state)

        # Get the response
        response = result.get("response", "I'm sorry, I couldn't generate a response.")

        print(f"✅ Workflow completed for query: '{user_query}'")
        print(f"📝 Response generated: '{response[:50]}...'")

        # Update chat history
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "chat_history": chat_history
        }
    except Exception as e:
        print(f"❌ Error in run_teacher_agent: {e}")
        import traceback
        traceback.print_exc()

        # Add error message to chat history
        error_response = "I'm sorry, I encountered an error while processing your question. Please try again."
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": error_response})

        return {
            "response": error_response,
            "chat_history": chat_history
        }

# Function to process and store a PDF
def handle_pdf_upload(pdf_path: str, user_id: str) -> dict:
    """
    Process and store a PDF for a user using MongoDB Atlas Vector Search.

    Args:
        pdf_path: Path to the PDF file
        user_id: User identifier

    Returns:
        Dictionary with status and PDF ID if successful
    """
    try:
        # Extract PDF filename from path
        pdf_name = os.path.basename(pdf_path)
        print(f"📄 Processing PDF for user {user_id}: {pdf_name}")

        # Process PDF
        documents = process_pdf(pdf_path)

        if not documents:
            print("❌ No documents extracted from PDF")
            return {"success": False, "pdf_id": "", "message": "No documents extracted from PDF"}

        # Vectorize and store in MongoDB Atlas
        try:
            pdf_id = vectorize_and_store_pdf(documents, user_id, pdf_name)

            return {
                "success": True,
                "pdf_id": pdf_id,
                "message": f"Successfully processed and stored PDF: {pdf_name}",
                "chunk_count": len(documents)
            }
        except ValueError as ve:
            # Handle specific vectorization errors
            print(f"❌ Error vectorizing PDF: {ve}")
            return {
                "success": False,
                "pdf_id": "",
                "message": f"Failed to vectorize PDF: {str(ve)}. MongoDB Atlas is required for vector storage."
            }

    except Exception as e:
        print(f"❌ Error in handle_pdf_upload: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "pdf_id": "", "message": str(e)}

# Function to remove PDF data
def handle_pdf_removal(user_id: str, pdf_id: Union[str, List[str]] = None) -> dict:
    """
    Remove PDF data for a user or specific PDF(s).

    Args:
        user_id: User identifier
        pdf_id: Optional PDF ID or list of PDF IDs to remove

    Returns:
        Dictionary with status and message
    """
    try:
        if pdf_id:
            if isinstance(pdf_id, list):
                # Multiple PDF IDs
                if len(pdf_id) > 0:
                    print(f"🗑️ Removing multiple PDFs: {', '.join(pdf_id)} for user {user_id}")
                    success = True
                    for single_pdf_id in pdf_id:
                        # Remove each PDF individually
                        if not remove_pdf(user_id, single_pdf_id):
                            success = False

                    if success:
                        return {
                            "success": True,
                            "message": f"Successfully removed {len(pdf_id)} PDFs",
                            "user_id": user_id,
                            "pdf_id": pdf_id
                        }
                    else:
                        return {
                            "success": False,
                            "message": f"Failed to remove some or all of the specified PDFs",
                            "user_id": user_id,
                            "pdf_id": pdf_id
                        }
                else:
                    # Empty list, treat as removing all PDFs
                    print(f"🗑️ Removing all PDFs for user {user_id} (empty PDF ID list provided)")
                    success = remove_pdf(user_id)

                    if success:
                        return {
                            "success": True,
                            "message": f"Successfully removed all PDFs for user: {user_id}",
                            "user_id": user_id
                        }
                    else:
                        return {
                            "success": False,
                            "message": f"Failed to remove PDFs for user: {user_id}",
                            "user_id": user_id
                        }
            else:
                # Single PDF ID
                print(f"🗑️ Removing PDF with ID {pdf_id} for user {user_id}")
                success = remove_pdf(user_id, pdf_id)

                if success:
                    return {
                        "success": True,
                        "message": f"Successfully removed PDF with ID: {pdf_id}",
                        "user_id": user_id,
                        "pdf_id": pdf_id
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to remove PDF with ID: {pdf_id}",
                        "user_id": user_id,
                        "pdf_id": pdf_id
                    }
        else:
            print(f"🗑️ Removing all PDFs for user {user_id}")
            success = remove_pdf(user_id)

            if success:
                return {
                    "success": True,
                    "message": f"Successfully removed all PDFs for user: {user_id}",
                    "user_id": user_id
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to remove PDFs for user: {user_id}",
                    "user_id": user_id
                }
    except Exception as e:
        print(f"❌ Error in handle_pdf_removal: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e), "user_id": user_id}

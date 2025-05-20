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

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    pdf_id: Optional[str]  # PDF ID for specific PDF searches
    pdf_content: Optional[List[Document]]

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
        PDF ID if successful, empty string otherwise
    """
    try:
        # Check if numpy is available
        try:
            import numpy as np
            print("✅ NumPy is available, version:", np.__version__)
        except ImportError:
            print("❌ NumPy is not available, attempting to use alternative approach")
            # Store documents directly without vectorization
            return store_pdf_content_without_vectors(documents, user_id, pdf_name)

        # Generate a unique PDF ID
        pdf_id = f"pdf_{user_id}_{str(uuid.uuid4())[:8]}"

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, PDF vectorization not available")
            return ""

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

        # Create vector store
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embeddings,
            collection=collection,
            index_name="pdf_vector_index"
        )

        # Add user_id and pdf_id to metadata for each document
        for i, doc in enumerate(documents):
            doc.metadata["user_id"] = user_id
            doc.metadata["pdf_id"] = pdf_id
            doc.metadata["chunk_id"] = f"{pdf_id}_{i}"

            # Update in MongoDB
            collection.update_one(
                {"metadata.chunk_id": doc.metadata["chunk_id"]},
                {"$set": {
                    "metadata.user_id": user_id,
                    "metadata.pdf_id": pdf_id
                }}
            )

        print(f"✅ Successfully stored PDF with ID: {pdf_id}")
        return pdf_id
    except Exception as e:
        print(f"❌ Error vectorizing PDF: {e}")
        import traceback
        traceback.print_exc()
        # Try alternative approach
        return store_pdf_content_without_vectors(documents, user_id, pdf_name)

def store_pdf_content_without_vectors(documents: List[Document], user_id: str, pdf_name: str) -> str:
    """
    Store PDF content directly in MongoDB without vectorization.

    Args:
        documents: List of document chunks from the PDF
        user_id: User identifier
        pdf_name: Name of the PDF file

    Returns:
        PDF ID if successful, empty string otherwise
    """
    try:
        print("📄 Storing PDF content without vectorization")

        # Generate a unique PDF ID
        pdf_id = f"pdf_{user_id}_{str(uuid.uuid4())[:8]}"

        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, PDF storage not available")
            return ""

        # Get MongoDB database
        db = get_database()
        collection = db["pdf_content"]

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

        # Store documents
        for i, doc in enumerate(documents):
            # Create document
            document = {
                "user_id": user_id,
                "pdf_id": pdf_id,
                "chunk_id": f"{pdf_id}_{i}",
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": datetime.now().isoformat()
            }

            # Insert into MongoDB
            collection.insert_one(document)

        print(f"✅ Successfully stored {len(documents)} PDF pages with ID: {pdf_id}")
        return pdf_id
    except Exception as e:
        print(f"❌ Error storing PDF content: {e}")
        return ""

def search_vector_db(query: str, user_id: str, k: int = 5, pdf_id: str = None) -> List[Document]:
    """
    Search the vector database for relevant documents.

    Args:
        query: The search query
        user_id: User identifier
        k: Number of results to return
        pdf_id: Optional PDF ID to filter by

    Returns:
        List of relevant documents
    """
    try:
        # Check if numpy is available
        try:
            import numpy
            print("✅ Using vector search")
        except ImportError:
            print("❌ NumPy is not available, falling back to direct content search")
            return search_pdf_content_without_vectors(query, user_id, k, pdf_id)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, vector search not available")
            return []

        # Get MongoDB database
        db = get_database()
        collection = db["pdf_vectors"]

        # Create vector store
        vector_store = MongoDBAtlasVectorSearch(
            collection,
            embeddings,
            index_name="pdf_vector_index"
        )

        # Build filter
        if pdf_id:
            # Search with user_id and pdf_id filter
            pre_filter = {"metadata.user_id": user_id, "metadata.pdf_id": pdf_id}
            print(f"🔍 Searching for PDF ID: {pdf_id}")
        else:
            # Search with user_id filter only
            pre_filter = {"metadata.user_id": user_id}
            print(f"🔍 Searching all PDFs for user: {user_id}")

        # Search with filter
        results = vector_store.similarity_search(
            query,
            k=k,
            pre_filter=pre_filter
        )

        print(f"✅ Found {len(results)} relevant PDF pages using vector search")
        return results
    except Exception as e:
        print(f"❌ Error searching vector DB: {e}")
        import traceback
        traceback.print_exc()
        # Try alternative approach
        return search_pdf_content_without_vectors(query, user_id, k, pdf_id)

def search_pdf_content_without_vectors(query: str, user_id: str, k: int = 5, pdf_id: str = None) -> List[Document]:
    """
    Search PDF content directly without using vectors.

    Args:
        query: The search query
        user_id: User identifier
        k: Number of results to return
        pdf_id: Optional PDF ID to filter by

    Returns:
        List of relevant documents
    """
    try:
        print("🔍 Searching PDF content without vectors")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, PDF content search not available")
            return []

        # Get MongoDB database
        db = get_database()
        collection = db["pdf_content"]

        # Simple text search (not as effective as vector search)
        # Split query into words for basic keyword matching
        keywords = query.lower().split()

        # Build filter
        if pdf_id:
            # Get documents for this user and PDF
            filter_query = {"user_id": user_id, "pdf_id": pdf_id}
            print(f"🔍 Searching for PDF ID: {pdf_id}")
        else:
            # Get all documents for this user
            filter_query = {"user_id": user_id}
            print(f"🔍 Searching all PDFs for user: {user_id}")

        # Get documents
        user_docs = list(collection.find(filter_query))

        # Score documents based on keyword matches
        scored_docs = []
        for doc in user_docs:
            content = doc.get("page_content", "").lower()
            # Count keyword matches
            score = sum(1 for keyword in keywords if keyword in content)
            scored_docs.append((score, doc))

        # Sort by score (descending) and take top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for _, doc in scored_docs[:k]]

        # Convert to Document objects
        results = []
        for doc in top_docs:
            results.append(Document(
                page_content=doc.get("page_content", ""),
                metadata={
                    "user_id": doc.get("user_id"),
                    "pdf_id": doc.get("pdf_id"),
                    "chunk_id": doc.get("chunk_id"),
                    "pdf_name": doc.get("pdf_name", "")
                }
            ))

        print(f"✅ Found {len(results)} relevant PDF pages using direct search")
        return results
    except Exception as e:
        print(f"❌ Error searching PDF content: {e}")
        return []

def search_financial_knowledge_base(query: str, k: int = 5) -> List[Document]:
    """Search the general financial knowledge base."""
    try:
        # Check if numpy is available
        try:
            import numpy
        except ImportError:
            print("❌ NumPy is not available, falling back to direct content search")
            return search_knowledge_base_without_vectors(query, k)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, knowledge base search not available")
            return []

        # Get MongoDB database
        db = get_database()
        collection = db["financial_knowledge"]

        # Create vector store
        vector_store = MongoDBAtlasVectorSearch(
            collection,
            embeddings,
            index_name="financial_knowledge_index"
        )

        # Search
        results = vector_store.similarity_search(query, k=k)

        return results
    except Exception as e:
        print(f"❌ Error searching knowledge base: {e}")
        import traceback
        traceback.print_exc()
        # Try alternative approach
        return search_knowledge_base_without_vectors(query, k)

def search_knowledge_base_without_vectors(query: str, k: int = 5) -> List[Document]:
    """Search financial knowledge base directly without using vectors."""
    try:
        print("🔍 Searching knowledge base without vectors")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, knowledge base search not available")
            return []

        # Get MongoDB database
        db = get_database()
        collection = db["financial_knowledge"]

        # Simple text search (not as effective as vector search)
        # Split query into words for basic keyword matching
        keywords = query.lower().split()

        # Get all documents
        all_docs = list(collection.find())

        # Score documents based on keyword matches
        scored_docs = []
        for doc in all_docs:
            content = doc.get("page_content", "").lower()
            score = sum(1 for keyword in keywords if keyword in content)
            scored_docs.append((score, doc))

        # Sort by score (descending) and take top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for _, doc in scored_docs[:k]]

        # Convert to Document objects
        results = []
        for doc in top_docs:
            results.append(Document(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {})
            ))

        print(f"✅ Found {len(results)} relevant knowledge base entries")
        return results
    except Exception as e:
        print(f"❌ Error searching knowledge base: {e}")
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
    print(f"🔍 Retrieving context for query: {state['user_query']}")

    user_id = state["user_id"]
    query = state["user_query"]
    pdf_id = state.get("pdf_id")  # Get PDF ID if available

    # Search PDF vectors if available
    if pdf_id:
        print(f"🔍 Searching specific PDF: {pdf_id}")
        pdf_results = search_vector_db(query, user_id, pdf_id=pdf_id)
    else:
        print(f"🔍 Searching all PDFs for user: {user_id}")
        pdf_results = search_vector_db(query, user_id)

    # Search financial knowledge base
    kb_results = search_financial_knowledge_base(query)

    # Combine results
    all_results = pdf_results + kb_results

    print(f"✅ Found {len(pdf_results)} PDF results and {len(kb_results)} knowledge base results")

    # Update state
    return {
        **state,
        "vector_search_results": all_results
    }

def generate_response_node(state: TeacherAgentState) -> TeacherAgentState:
    """Generate a response using the teacher agent."""
    print(f"🧠 Generating response for query: {state['user_query']}")

    # Format context from vector search
    context = ""
    if state.get("vector_search_results"):
        context = "\n\n".join([doc.page_content for doc in state["vector_search_results"]])

    # Format chat history
    formatted_history = []
    for message in state.get("chat_history", []):
        if message.get("role") == "user":
            formatted_history.append(HumanMessage(content=message.get("content", "")))
        elif message.get("role") == "assistant":
            formatted_history.append(AIMessage(content=message.get("content", "")))

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a friendly and knowledgeable financial teacher.
Your goal is to explain financial concepts in simple, easy-to-understand language.
Always be supportive, patient, and encouraging. Use analogies and examples to make complex concepts accessible.
When explaining financial terms, avoid jargon and break down concepts step by step.
If you're not sure about something, be honest about it rather than making up information.

CRITICAL INSTRUCTION: You MUST respond ONLY to the exact query provided in the "User Query" field.
DO NOT respond to any other topic or query that might appear elsewhere in the context or history.
The only query you should respond to is the one explicitly labeled as "User Query: {query}".

If you have access to PDF content or knowledge base information, use it to enhance your explanations,
but always maintain a conversational and educational tone."""),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="""
The EXACT user query that you MUST respond to is: "{query}"

Relevant Context:
{context}

IMPORTANT: Your response must directly address this query: "{query}"
Do not respond to any other topic that might appear in the context or history.
Explain this financial concept in simple terms that anyone can understand.""")
    ])

    # Get LLM
    llm = get_llm("groq/llama3-70b-8192")  # Using Llama 3 for teaching

    # Create chain
    chain = prompt | llm

    # Execute chain
    try:
        result = chain.invoke({
            "history": formatted_history,
            "query": state["user_query"],
            "context": context
        })

        # Update state
        return {
            **state,
            "response": result.content
        }
    except Exception as e:
        print(f"❌ Error in generate_response_node: {e}")
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
def run_teacher_agent(user_query: str, user_id: str, chat_history: List[Dict[str, str]] = None, pdf_id: str = None) -> Dict[str, Any]:
    """
    Run the teacher agent to answer a user query.

    Args:
        user_query: The user's question
        user_id: User identifier
        chat_history: Optional chat history
        pdf_id: Optional PDF ID to search in a specific PDF

    Returns:
        Dict with response and updated chat history
    """
    print(f"🚀 Running teacher agent for user {user_id}")

    # Create the workflow graph
    workflow = create_teacher_agent_graph()

    # Initialize chat history if None
    if chat_history is None:
        chat_history = []

    # Initialize state
    initial_state = {
        "user_query": user_query,
        "chat_history": chat_history,
        "user_id": user_id,
        "pdf_path": None,
        "pdf_id": pdf_id,  # Include PDF ID if provided
        "pdf_content": None,
        "vector_search_results": None,
        "response": None
    }

    # Run the workflow
    try:
        # Execute the workflow
        result = workflow.invoke(initial_state)

        # Get the response
        response = result.get("response", "I'm sorry, I couldn't generate a response.")

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
    Process and store a PDF for a user.

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

        # Vectorize and store
        pdf_id = vectorize_and_store_pdf(documents, user_id, pdf_name)

        if pdf_id:
            return {
                "success": True,
                "pdf_id": pdf_id,
                "message": f"Successfully processed and stored PDF: {pdf_name}",
                "chunk_count": len(documents)
            }
        else:
            return {"success": False, "pdf_id": "", "message": "Failed to store PDF"}
    except Exception as e:
        print(f"❌ Error in handle_pdf_upload: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "pdf_id": "", "message": str(e)}

# Function to remove PDF data
def handle_pdf_removal(user_id: str, pdf_id: str = None) -> dict:
    """
    Remove PDF data for a user or a specific PDF.

    Args:
        user_id: User identifier
        pdf_id: Optional PDF ID to remove a specific PDF

    Returns:
        Dictionary with status and message
    """
    try:
        if pdf_id:
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

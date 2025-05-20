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
def get_llm(model_name="grok-1"):
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
    elif model_name == "grok-1":
        # Use Groq with Grok model
        return ChatGroq(
            model_name="grok-1",
            temperature=0.2,
            max_tokens=4000
        )
    else:
        # Default to Groq
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

def vectorize_and_store_pdf(documents: List[Document], user_id: str) -> bool:
    """Vectorize and store PDF content in MongoDB Atlas."""
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, PDF vectorization not available")
            return False

        # Get MongoDB database
        db = get_database()
        collection = db["pdf_vectors"]

        # Create vector store
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embeddings,
            collection=collection,
            index_name="pdf_vector_index"
        )

        # Add user_id to metadata for each document
        for i, doc in enumerate(documents):
            doc.metadata["user_id"] = user_id
            doc.metadata["doc_id"] = f"{user_id}_{i}"

            # Update in MongoDB
            collection.update_one(
                {"metadata.doc_id": doc.metadata["doc_id"]},
                {"$set": {"metadata.user_id": user_id}}
            )

        return True
    except Exception as e:
        print(f"❌ Error vectorizing PDF: {e}")
        return False

def search_vector_db(query: str, user_id: str, k: int = 5) -> List[Document]:
    """Search the vector database for relevant documents."""
    try:
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

        # Search with user_id filter
        results = vector_store.similarity_search(
            query,
            k=k,
            pre_filter={"metadata.user_id": user_id}
        )

        return results
    except Exception as e:
        print(f"❌ Error searching vector DB: {e}")
        return []

def search_financial_knowledge_base(query: str, k: int = 5) -> List[Document]:
    """Search the general financial knowledge base."""
    try:
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
        return []

def remove_user_pdf_vectors(user_id: str) -> bool:
    """Remove PDF vectors for a specific user."""
    try:
        # Get MongoDB client
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, PDF vector removal not available")
            return False

        # Get MongoDB database
        db = get_database()
        collection = db["pdf_vectors"]

        # Delete documents with user_id
        result = collection.delete_many({"metadata.user_id": user_id})

        print(f"🗑️ Removed {result.deleted_count} PDF vectors for user {user_id}")
        return True
    except Exception as e:
        print(f"❌ Error removing PDF vectors: {e}")
        return False

# Define agent nodes
def retrieve_context_node(state: TeacherAgentState) -> TeacherAgentState:
    """Retrieve relevant context from vector databases."""
    print(f"🔍 Retrieving context for query: {state['user_query']}")

    user_id = state["user_id"]
    query = state["user_query"]

    # Search PDF vectors if available
    pdf_results = search_vector_db(query, user_id)

    # Search financial knowledge base
    kb_results = search_financial_knowledge_base(query)

    # Combine results
    all_results = pdf_results + kb_results

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

If you have access to PDF content or knowledge base information, use it to enhance your explanations,
but always maintain a conversational and educational tone.
"""),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="""
User Query: {query}

Relevant Context:
{context}

Please explain this financial concept in simple terms that anyone can understand.
""")
    ])

    # Get LLM
    llm = get_llm("grok-1")  # Using Grok for teaching

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
def run_teacher_agent(user_query: str, user_id: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Run the teacher agent to answer a user query.

    Args:
        user_query: The user's question
        user_id: User identifier
        chat_history: Optional chat history

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
def handle_pdf_upload(pdf_path: str, user_id: str) -> bool:
    """
    Process and store a PDF for a user.

    Args:
        pdf_path: Path to the PDF file
        user_id: User identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📄 Processing PDF for user {user_id}: {pdf_path}")

        # Process PDF
        documents = process_pdf(pdf_path)

        if not documents:
            print("❌ No documents extracted from PDF")
            return False

        # Vectorize and store
        success = vectorize_and_store_pdf(documents, user_id)

        return success
    except Exception as e:
        print(f"❌ Error in handle_pdf_upload: {e}")
        return False

# Function to remove PDF data
def handle_pdf_removal(user_id: str) -> bool:
    """
    Remove PDF data for a user.

    Args:
        user_id: User identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🗑️ Removing PDF data for user {user_id}")

        # Remove vectors
        success = remove_user_pdf_vectors(user_id)

        return success
    except Exception as e:
        print(f"❌ Error in handle_pdf_removal: {e}")
        return False

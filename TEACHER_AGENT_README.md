# Teacher Agent for Financial Guru

This document explains how to use the new Teacher Agent functionality in the Financial Guru application.

## Overview

The Teacher Agent is designed to explain financial concepts in simple terms to users. It has the following features:

1. Chat-based learning interface
2. Access to a vector database of financial knowledge
3. Ability to incorporate PDF content into explanations
4. Simple, easy-to-understand explanations of complex financial concepts

## API Endpoints

The Teacher Agent functionality is accessible through the following API endpoints:

### 1. `/user/learning` - Chat with the Teacher Agent

This endpoint allows users to ask questions and receive explanations from the Teacher Agent.

**Request:**
```json
{
  "user_id": "user123",
  "query": "What is compound interest?",
  "chat_history": [
    {
      "role": "user",
      "content": "Can you explain budgeting to me?"
    },
    {
      "role": "assistant",
      "content": "Budgeting is the process of creating a plan for how you will spend your money..."
    }
  ]
}
```

**Response:**
```json
{
  "response": "Compound interest is when you earn interest on both the money you've saved and the interest you earn...",
  "chat_history": [
    {
      "role": "user",
      "content": "Can you explain budgeting to me?"
    },
    {
      "role": "assistant",
      "content": "Budgeting is the process of creating a plan for how you will spend your money..."
    },
    {
      "role": "user",
      "content": "What is compound interest?"
    },
    {
      "role": "assistant",
      "content": "Compound interest is when you earn interest on both the money you've saved and the interest you earn..."
    }
  ]
}
```

### 2. `/pdf/chat` - Upload a PDF for the Teacher Agent

This endpoint allows users to upload a PDF document that the Teacher Agent will use to enhance its explanations.

**Request:**
- Form data with:
  - `user_id`: The user's ID
  - `pdf_file`: The PDF file to upload

**Response:**
```json
{
  "status": "success",
  "message": "PDF uploaded and processed successfully: financial_guide.pdf",
  "user_id": "user123"
}
```

### 3. `/pdf/removed` - Remove PDF data

This endpoint removes previously uploaded PDF data for a specific user.

**Request:**
```json
{
  "user_id": "user123"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "PDF data removed successfully for user: user123",
  "user_id": "user123"
}
```

## Setup and Initialization

Before using the Teacher Agent, you need to initialize the financial knowledge base:

1. Make sure you have MongoDB Atlas configured in your environment variables
2. Run the initialization script:

```bash
python initialize_knowledge_base.py
```

This will create a vector database with basic financial concepts that the Teacher Agent can access.

## Usage Examples

### Example 1: Simple Question and Answer

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8001/user/learning",
    json={
        "user_id": "user123",
        "query": "What is an emergency fund?",
        "chat_history": []
    }
)

print(response.json()["response"])
```

### Example 2: Upload a PDF and Ask Related Questions

```python
import requests

# Upload a PDF
with open("financial_guide.pdf", "rb") as f:
    files = {"pdf_file": f}
    data = {"user_id": "user123"}
    response = requests.post(
        "http://localhost:8001/pdf/chat",
        files=files,
        data=data
    )

# Ask a question related to the PDF content
response = requests.post(
    "http://localhost:8001/user/learning",
    json={
        "user_id": "user123",
        "query": "Can you explain the investment strategies mentioned in the PDF?",
        "chat_history": []
    }
)

print(response.json()["response"])
```

### Example 3: Continuing a Conversation

```python
import requests

# First question
response1 = requests.post(
    "http://localhost:8001/user/learning",
    json={
        "user_id": "user123",
        "query": "What is debt management?",
        "chat_history": []
    }
)

chat_history = response1.json()["chat_history"]

# Follow-up question
response2 = requests.post(
    "http://localhost:8001/user/learning",
    json={
        "user_id": "user123",
        "query": "What's the difference between the avalanche and snowball methods?",
        "chat_history": chat_history
    }
)

print(response2.json()["response"])
```

## Technical Details

The Teacher Agent uses:

1. LangGraph for the agent workflow
2. MongoDB Atlas for vector storage
3. Grok LLM for generating explanations
4. LangChain for document processing and retrieval

The agent maintains conversation context through chat history, allowing for follow-up questions and a more natural learning experience.

## Troubleshooting

If you encounter issues:

1. Check that MongoDB Atlas is properly configured
2. Ensure OpenAI API keys are set for embeddings
3. Verify that the PDF is properly formatted and readable
4. Check the server logs for detailed error messages

For any persistent issues, please contact the development team.

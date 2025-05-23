# MongoDB Atlas Vector Search Setup Guide

This guide explains how to set up MongoDB Atlas Vector Search for the Financial Guru application. The application uses vector search to find relevant information in PDFs and the financial knowledge base.

## Prerequisites

1. A MongoDB Atlas account (M0 free tier or higher)
2. A MongoDB Atlas cluster (M10+ recommended for production)
3. MongoDB connection string in your `.env` file

## Automatic Setup

The easiest way to set up the required vector search indexes is to run the provided script:

```bash
# If using a virtual environment
source fresh_env/bin/activate

# Run the setup script
python setup_mongodb_indexes.py
```

This script will:
1. Connect to your MongoDB Atlas cluster
2. Create or update the `pdf_vector_index` for the `pdf_vectors` collection
   - If the index already exists, it will try to update it
   - If update fails, it will drop and recreate the index
3. Create or update the `financial_knowledge_index` for the `financial_knowledge` collection
   - If the index already exists, it will try to update it
   - If update fails, it will drop and recreate the index

## Manual Setup

If the automatic setup doesn't work, you can create the indexes manually in the MongoDB Atlas UI:

### Setting up pdf_vector_index

1. Go to your MongoDB Atlas cluster
2. Navigate to the "Search" tab
3. Click "Create Search Index"
4. Select the "financial_simulation" database and "pdf_vectors" collection
5. Choose "Visual Editor" or "JSON Editor"
6. If using JSON Editor, paste the following configuration:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 768,
        "similarity": "cosine",
        "type": "knnVector"
      },
      "metadata": {
        "fields": {
          "user_id": {
            "type": "token"
          },
          "pdf_id": {
            "type": "token"
          },
          "chunk_id": {
            "type": "token"
          },
          "pdf_name": {
            "type": "string"
          }
        }
      }
    }
  }
}
```

7. Name the index "pdf_vector_index"
8. Click "Create Index"

### Setting up financial_knowledge_index

1. Go to your MongoDB Atlas cluster
2. Navigate to the "Search" tab
3. Click "Create Search Index"
4. Select the "financial_simulation" database and "financial_knowledge" collection
5. Choose "Visual Editor" or "JSON Editor"
6. If using JSON Editor, paste the following configuration:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 768,
        "similarity": "cosine",
        "type": "knnVector"
      },
      "title": {
        "type": "string"
      },
      "doc_id": {
        "type": "token"
      }
    }
  }
}
```

7. Name the index "financial_knowledge_index"
8. Click "Create Index"

## Troubleshooting

### Common Errors

1. **"Path 'metadata.user_id' needs to be indexed as token"**
   - This error occurs when the `metadata.user_id` field is not properly indexed as a token type.
   - Solution: Make sure the `pdf_vector_index` includes `metadata.user_id` as a token field.
   - Alternative solution: Instead of using the `pre_filter` parameter in `similarity_search`, use a separate `$match` stage in the aggregation pipeline after the `$vectorSearch` stage.

2. **"Index not found"**
   - This error occurs when the vector search index doesn't exist or has a different name.
   - Solution: Verify that the index names match exactly: `pdf_vector_index` and `financial_knowledge_index`.

3. **"Vector dimensions mismatch"**
   - This error occurs when the vector dimensions in the index don't match the embedding model.
   - Solution: Ensure the index is configured for 768 dimensions for the `sentence-transformers/all-mpnet-base-v2` model.

### Cluster Requirements

For production use, MongoDB Atlas recommends:
- M10+ tier cluster (vector search is limited on M0 free tier)
- Adequate storage for your vector data
- Proper network access configuration

## Testing the Setup

After setting up the indexes, you can test the vector search functionality:

1. Upload a PDF through the application
2. Ask a question related to the PDF content
3. Verify that the response includes information from the PDF

If the vector search is working correctly, you should see log messages like:
```
✅ Using vector search with MongoDB Atlas
🔍 Using MongoDB Atlas Vector Search with direct aggregation...
✅ MongoDB Atlas Vector Search completed with X results
```

## Direct Aggregation vs. LangChain Integration

The application uses a direct MongoDB aggregation pipeline approach for vector search instead of relying on the LangChain `similarity_search` method with `pre_filter`. This is because:

1. The direct aggregation approach gives more control over the search pipeline
2. It allows for proper filtering of results using a separate `$match` stage
3. It avoids issues with the `pre_filter` parameter in the LangChain integration

The direct aggregation pipeline looks like this:

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "pdf_vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 10
        }
    },
    {
        "$match": {
            "metadata.user_id": user_id
        }
    },
    {
        "$limit": 5
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
```

## Additional Resources

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [LangChain MongoDB Integration](https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas)

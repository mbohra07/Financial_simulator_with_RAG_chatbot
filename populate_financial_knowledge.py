"""
Script to populate the financial_knowledge collection in MongoDB Atlas.

This script loads financial knowledge from a JSON file, generates embeddings,
and stores them in the MongoDB Atlas database for RAG retrieval.

Usage:
    1. Activate your virtual environment: source fresh_env/bin/activate
    2. Run the script: python populate_financial_knowledge.py
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import uuid

# Import MongoDB client
from database.mongodb_client import get_database, get_client, USE_MOCK_DB

# Import embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import Document class
from langchain_core.documents import Document

# Try to import the MongoDB Atlas Vector Search
try:
    from langchain_mongodb import MongoDBAtlasVectorSearch
    print("✅ Using updated MongoDB Atlas Vector Search from langchain_mongodb")
except ImportError:
    # Fall back to the deprecated version
    try:
        from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
        print("⚠️ Using deprecated MongoDBAtlasVectorSearch from langchain")
    except ImportError:
        from langchain_community.vectorstores import MongoDBAtlasVectorSearch
        print("⚠️ Using deprecated MongoDBAtlasVectorSearch from langchain_community")

# Sample financial knowledge data
FINANCIAL_KNOWLEDGE = [
    {
        "topic": "Budgeting",
        "category": "Personal Finance",
        "content": "Budgeting is the process of creating a plan to spend your money. This spending plan is called a budget. Creating a budget allows you to determine in advance whether you will have enough money to do the things you need to do or would like to do. Budgeting is simply balancing your expenses with your income."
    },
    {
        "topic": "Emergency Fund",
        "category": "Personal Finance",
        "content": "An emergency fund is money you set aside to pay for unexpected expenses. The purpose of an emergency fund is to improve financial security by creating a safety net that can be used to meet unanticipated expenses, such as an illness or major home repairs. Ideally, emergency funds should contain enough money to cover three to six months of expenses."
    },
    {
        "topic": "Compound Interest",
        "category": "Investing",
        "content": "Compound interest is the interest on a loan or deposit calculated based on both the initial principal and the accumulated interest from previous periods. Thought to have originated in 17th-century Italy, compound interest can be thought of as 'interest on interest,' and will make a sum grow at a faster rate than simple interest, which is calculated only on the principal amount."
    },
    {
        "topic": "Stock Market",
        "category": "Investing",
        "content": "The stock market refers to the collection of markets and exchanges where regular activities of buying, selling, and issuance of shares of publicly-held companies take place. Such financial activities are conducted through institutionalized formal exchanges or over-the-counter (OTC) marketplaces which operate under a defined set of regulations."
    },
    {
        "topic": "Mutual Funds",
        "category": "Investing",
        "content": "A mutual fund is a type of financial vehicle made up of a pool of money collected from many investors to invest in securities like stocks, bonds, money market instruments, and other assets. Mutual funds are operated by professional money managers, who allocate the fund's assets and attempt to produce capital gains or income for the fund's investors."
    },
    {
        "topic": "ETFs",
        "category": "Investing",
        "content": "An exchange-traded fund (ETF) is a type of investment fund and exchange-traded product, i.e., they are traded on stock exchanges. ETFs are similar in many ways to mutual funds, except that ETFs are bought and sold throughout the day on stock exchanges while mutual funds are bought and sold based on their price at day's end."
    },
    {
        "topic": "Retirement Planning",
        "category": "Personal Finance",
        "content": "Retirement planning is the process of determining retirement income goals, and the actions and decisions necessary to achieve those goals. Retirement planning includes identifying sources of income, estimating expenses, implementing a savings program, and managing assets and risk. Future cash flows are estimated to determine if the retirement income goal will be achieved."
    },
    {
        "topic": "401(k)",
        "category": "Retirement",
        "content": "A 401(k) is a retirement savings plan sponsored by an employer. It lets workers save and invest a piece of their paycheck before taxes are taken out. Taxes aren't paid until the money is withdrawn from the account. 401(k) plans are named for the section of the tax code that governs them."
    },
    {
        "topic": "IRA",
        "category": "Retirement",
        "content": "An Individual Retirement Account (IRA) is a tax-advantaged investing tool that individuals use to earmark funds for retirement savings. There are several types of IRAs as of 2020: Traditional IRAs, Roth IRAs, SEP IRAs, and SIMPLE IRAs. Traditional and Roth IRAs are created by individual taxpayers. SEP and SIMPLE IRAs are implemented by small business owners and self-employed individuals."
    },
    {
        "topic": "Credit Score",
        "category": "Credit",
        "content": "A credit score is a numerical expression based on a level analysis of a person's credit files, to represent the creditworthiness of an individual. A credit score is primarily based on a credit report, information typically sourced from credit bureaus. Lenders, such as banks and credit card companies, use credit scores to evaluate the potential risk posed by lending money to consumers."
    },
    {
        "topic": "Mortgage",
        "category": "Real Estate",
        "content": "A mortgage is a loan in which property or real estate is used as collateral. The borrower enters into an agreement with the lender (usually a bank) wherein the borrower receives cash upfront then makes payments over a set time span until he pays back the lender in full. A mortgage is often referred to as home loan when its used for the purchase of a home."
    },
    {
        "topic": "Inflation",
        "category": "Economics",
        "content": "Inflation is the rate at which the general level of prices for goods and services is rising and, consequently, the purchasing power of currency is falling. Central banks attempt to limit inflation — and avoid deflation — in order to keep the economy running smoothly."
    },
    {
        "topic": "Diversification",
        "category": "Investing",
        "content": "Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio. The rationale behind this technique is that a portfolio constructed of different kinds of assets will, on average, yield higher long-term returns and lower the risk of any individual holding or security."
    },
    {
        "topic": "Asset Allocation",
        "category": "Investing",
        "content": "Asset allocation is an investment strategy that aims to balance risk and reward by apportioning a portfolio's assets according to an individual's goals, risk tolerance, and investment horizon. The three main asset classes - equities, fixed-income, and cash and equivalents - have different levels of risk and return, so each will behave differently over time."
    },
    {
        "topic": "Tax-Loss Harvesting",
        "category": "Tax Planning",
        "content": "Tax-loss harvesting is a strategy used to lower your tax bill by selling investments that have lost value. The loss from the sale can be used to offset capital gains from other investments, reducing your tax liability. This strategy is typically used in taxable investment accounts, not tax-advantaged accounts like IRAs or 401(k)s."
    }
]

def create_financial_knowledge_file():
    """Create a JSON file with financial knowledge if it doesn't exist."""
    file_path = Path("data/financial_knowledge.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        with open(file_path, "w") as f:
            json.dump(FINANCIAL_KNOWLEDGE, f, indent=2)
        print(f"✅ Created financial knowledge file at {file_path}")
    else:
        print(f"ℹ️ Financial knowledge file already exists at {file_path}")
    
    return file_path

def load_financial_knowledge(file_path: Path) -> List[Dict[str, Any]]:
    """Load financial knowledge from a JSON file."""
    with open(file_path, "r") as f:
        knowledge_data = json.load(f)
    
    print(f"✅ Loaded {len(knowledge_data)} financial knowledge entries")
    return knowledge_data

def convert_to_documents(knowledge_data: List[Dict[str, Any]]) -> List[Document]:
    """Convert financial knowledge data to Document objects."""
    documents = []
    
    for item in knowledge_data:
        # Create a document with the content and metadata
        doc = Document(
            page_content=item["content"],
            metadata={
                "topic": item["topic"],
                "category": item["category"],
                "source": "financial_knowledge_base",
                "id": f"fin_knowledge_{str(uuid.uuid4())[:8]}"
            }
        )
        documents.append(doc)
    
    print(f"✅ Converted {len(documents)} entries to Document objects")
    return documents

def vectorize_and_store_knowledge(documents: List[Document]) -> bool:
    """Vectorize and store financial knowledge in MongoDB Atlas."""
    try:
        # Check MongoDB connection
        if USE_MOCK_DB:
            print("❌ MongoDB connection not available. Cannot proceed with vector storage.")
            return False
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("✅ Initialized embedding model")
        
        # Get MongoDB database
        db = get_database()
        collection = db["financial_knowledge"]
        
        # Clear existing data if any
        result = collection.delete_many({})
        print(f"🧹 Cleared {result.deleted_count} existing entries from financial_knowledge collection")
        
        # Process documents in batches to avoid memory issues
        batch_size = 10
        successful_insertions = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}...")
            
            # Get embeddings for batch
            texts = [doc.page_content for doc in batch]
            embeddings_list = embeddings.embed_documents(texts)
            
            # Store documents with embeddings
            for j, (doc, embedding_vector) in enumerate(zip(batch, embeddings_list)):
                try:
                    # Create document with vector
                    vector_doc = {
                        "page_content": doc.page_content,
                        "embedding": embedding_vector,
                        "metadata": doc.metadata,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Insert into MongoDB
                    collection.insert_one(vector_doc)
                    successful_insertions += 1
                except Exception as e:
                    print(f"❌ Error inserting document {i+j}: {e}")
        
        print(f"✅ Successfully inserted {successful_insertions}/{len(documents)} documents into financial_knowledge collection")
        
        # Create vector search index if it doesn't exist
        # Note: This is typically done through the MongoDB Atlas UI
        # This is just a reminder for the user
        print("\n⚠️ IMPORTANT: Make sure to create a vector search index in MongoDB Atlas:")
        print("1. Go to your MongoDB Atlas cluster")
        print("2. Navigate to the 'Search' tab")
        print("3. Create a new index on the 'financial_knowledge' collection")
        print("4. Name the index 'financial_knowledge_index'")
        print("5. Configure it to index the 'embedding' field as a vector")
        print("6. Set the dimensions to 768 (for 'sentence-transformers/all-mpnet-base-v2')")
        
        return True
    except Exception as e:
        print(f"❌ Error vectorizing knowledge: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to populate the financial knowledge collection."""
    print("🚀 Starting financial knowledge population process...")
    
    # Create and load financial knowledge
    knowledge_file = create_financial_knowledge_file()
    knowledge_data = load_financial_knowledge(knowledge_file)
    
    # Convert to documents
    documents = convert_to_documents(knowledge_data)
    
    # Vectorize and store in MongoDB
    success = vectorize_and_store_knowledge(documents)
    
    if success:
        print("✅ Successfully populated financial knowledge collection")
    else:
        print("❌ Failed to populate financial knowledge collection")

if __name__ == "__main__":
    main()

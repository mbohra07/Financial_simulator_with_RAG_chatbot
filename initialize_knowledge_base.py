"""
Initialize the financial knowledge base in MongoDB Atlas.
This script creates a database with basic financial concepts.
"""

import os
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document

# Import MongoDB client
from database.mongodb_client import (
    get_client,
    get_database,
    USE_MOCK_DB
)

# Load environment variables
load_dotenv()

# Basic financial concepts
FINANCIAL_CONCEPTS = [
    {
        "title": "Budgeting",
        "content": """
Budgeting is the process of creating a plan for how you will spend your money. It involves tracking your income and expenses to ensure you're not spending more than you earn.

Key components of a budget:
1. Income: Money you receive (salary, freelance work, etc.)
2. Fixed expenses: Regular payments that don't change (rent, mortgage, car payment)
3. Variable expenses: Costs that change month to month (groceries, entertainment)
4. Savings: Money set aside for future goals
5. Debt payments: Paying down credit cards, loans, etc.

Budgeting helps you:
- Understand your spending habits
- Identify areas where you can cut back
- Plan for future expenses
- Avoid or reduce debt
- Achieve financial goals

Popular budgeting methods include the 50/30/20 rule (50% needs, 30% wants, 20% savings) and zero-based budgeting (giving every dollar a job).
"""
    },
    {
        "title": "Emergency Fund",
        "content": """
An emergency fund is money set aside specifically for unexpected expenses or financial emergencies.

Key aspects of an emergency fund:
1. Purpose: To cover unexpected costs like medical emergencies, car repairs, job loss
2. Size: Typically 3-6 months of essential expenses
3. Accessibility: Should be kept in a liquid account (easily accessible)
4. Separation: Should be separate from regular checking/savings to avoid temptation

Benefits of an emergency fund:
- Provides financial security and peace of mind
- Prevents going into debt for unexpected expenses
- Reduces financial stress during difficult times
- Gives you time to find a new job if you lose your current one

Where to keep an emergency fund:
- High-yield savings account
- Money market account
- Short-term CDs (certificates of deposit)

Building an emergency fund should typically be one of your first financial priorities after paying off high-interest debt.
"""
    },
    {
        "title": "Compound Interest",
        "content": """
Compound interest is when you earn interest on both the money you've saved and the interest you earn.

How compound interest works:
1. You deposit money into an account that earns interest
2. After a period (day, month, year), interest is added to your balance
3. In the next period, you earn interest on your original deposit PLUS the previously earned interest
4. This creates a snowball effect where your money grows faster over time

The compound interest formula:
A = P(1 + r/n)^(nt)
Where:
- A = Final amount
- P = Principal (initial investment)
- r = Annual interest rate (decimal)
- n = Number of times interest compounds per year
- t = Time in years

Key factors affecting compound interest:
- Interest rate: Higher rates lead to faster growth
- Compounding frequency: More frequent compounding leads to more growth
- Time: The longer your money compounds, the more dramatic the growth

The Rule of 72:
A quick way to estimate how long it will take to double your money. Divide 72 by the interest rate to get the approximate years needed to double your investment.

Example: At 8% interest, it would take approximately 72 ÷ 8 = 9 years to double your money.
"""
    },
    {
        "title": "Debt Management",
        "content": """
Debt management involves strategies to handle and pay off what you owe effectively.

Types of debt:
1. Good debt: Potentially increases your net worth or income (mortgages, student loans, business loans)
2. Bad debt: Decreases in value or used for consumption (credit cards, payday loans, high-interest personal loans)

Debt repayment strategies:
1. Avalanche method: Pay minimum on all debts, then put extra money toward highest interest rate debt first
2. Snowball method: Pay minimum on all debts, then put extra money toward smallest balance first
3. Debt consolidation: Combine multiple debts into one loan with a lower interest rate
4. Balance transfers: Move high-interest credit card debt to a card with 0% intro APR

Tips for managing debt:
- Know exactly what you owe (balances, interest rates, minimum payments)
- Always pay at least the minimum payment on time
- Create a debt payoff plan with specific goals
- Consider negotiating with creditors for lower rates
- Avoid taking on new debt while paying off existing debt
- Consider seeking help from a credit counselor if overwhelmed

Warning signs of debt problems:
- Using credit for essential expenses like food and utilities
- Making only minimum payments
- Maxing out credit cards
- Using one form of credit to pay another
- Receiving collection calls
"""
    },
    {
        "title": "Investing Basics",
        "content": """
Investing is putting money into assets with the expectation of generating income or profit over time.

Common investment types:
1. Stocks: Ownership shares in a company
2. Bonds: Loans to a company or government that pay interest
3. Mutual funds: Professionally managed collections of stocks, bonds, or other securities
4. ETFs (Exchange-Traded Funds): Similar to mutual funds but trade like stocks
5. Real estate: Property investments
6. Retirement accounts: 401(k)s, IRAs, etc.

Key investment concepts:
1. Risk vs. return: Higher potential returns typically come with higher risk
2. Diversification: Spreading investments across different assets to reduce risk
3. Asset allocation: How you divide your portfolio among different asset types
4. Dollar-cost averaging: Investing a fixed amount regularly regardless of market conditions
5. Compound growth: Reinvesting earnings to generate more returns over time

Investment tips for beginners:
- Start early to benefit from compound growth
- Invest regularly and automatically
- Focus on long-term goals rather than short-term market movements
- Keep investment costs low (look for low-fee funds)
- Consider your risk tolerance and time horizon
- Don't try to time the market
- Rebalance your portfolio periodically

Common investment accounts:
- Brokerage accounts (taxable)
- Retirement accounts (tax-advantaged): 401(k), IRA, Roth IRA
- Education accounts: 529 plans, Coverdell ESAs
"""
    },
    {
        "title": "Retirement Planning",
        "content": """
Retirement planning involves strategies to ensure financial security when you stop working.

Key retirement planning elements:
1. Setting retirement goals: When you want to retire and lifestyle expectations
2. Estimating expenses: How much you'll need monthly/annually
3. Calculating savings needs: Total amount needed to fund retirement
4. Choosing retirement accounts: 401(k)s, IRAs, Roth accounts, etc.
5. Investment strategy: Asset allocation based on time horizon and risk tolerance
6. Social Security planning: When to claim benefits
7. Healthcare planning: Medicare, long-term care insurance

Common retirement accounts:
1. 401(k)/403(b): Employer-sponsored plans with potential matching contributions
2. Traditional IRA: Tax-deductible contributions, tax-deferred growth
3. Roth IRA: After-tax contributions, tax-free growth and withdrawals
4. SEP IRA and Solo 401(k): For self-employed individuals
5. Pension plans: Employer-funded retirement benefits

The 4% rule:
A guideline suggesting you can withdraw 4% of your retirement savings in the first year, then adjust for inflation each year after, with a high probability of not running out of money for 30 years.

Retirement planning tips:
- Start early to benefit from compound growth
- Maximize employer matches in workplace plans
- Consider tax diversification (mix of pre-tax and Roth accounts)
- Increase savings rate as income grows
- Adjust investment mix to become more conservative as retirement approaches
- Plan for healthcare costs, including long-term care
- Consider working with a financial advisor for complex situations
"""
    }
]

def create_documents_from_concepts(concepts: List[Dict[str, str]]) -> List[Document]:
    """Convert concept dictionaries to Document objects."""
    documents = []

    for concept in concepts:
        doc = Document(
            page_content=concept["content"],
            metadata={"title": concept["title"]}
        )
        documents.append(doc)

    return documents

def initialize_knowledge_base():
    """Initialize the financial knowledge base in MongoDB Atlas."""
    try:
        print("🚀 Initializing financial knowledge base...")

        # Check if MongoDB is available
        if USE_MOCK_DB:
            print("⚠️ Using mock DB, knowledge base initialization not available")
            return False

        # Create documents
        documents = create_documents_from_concepts(FINANCIAL_CONCEPTS)
        print(f"📄 Created {len(documents)} documents")

        # Get MongoDB database
        db = get_database()
        collection = db["financial_knowledge"]

        # Clear existing data
        collection.delete_many({})

        # Store documents directly without vectorization
        for i, doc in enumerate(documents):
            # Create document
            document = {
                "doc_id": f"concept_{i}",
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "title": doc.metadata.get("title", f"Concept {i}"),
                "timestamp": datetime.now().isoformat()
            }

            # Insert into MongoDB
            collection.insert_one(document)

        print("✅ Successfully initialized financial knowledge base with direct storage")
        return True
    except Exception as e:
        print(f"❌ Error initializing knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    initialize_knowledge_base()

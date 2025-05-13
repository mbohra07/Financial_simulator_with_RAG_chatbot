from crewai.tools import tool

# ***************************************************Tools used to evaluate spending***************************************************

@tool("Opportunity Cost Analyzer")
def opportunity_cost_analyzer(spending_data: str, user_name: str = "you") -> str:
    """
    Identifies where the user is spending in ways that miss out on better alternatives like investing, saving, or debt payoff.
    Applies opportunity cost principles to real spending patterns.
    """
    return (
        f"💡 Opportunity Cost Insights for {user_name}:\n"
        "- Spent ₹4000 on entertainment in 2 months → could have grown to ₹4200 via FD.\n"
        "- Impulse shopping worth ₹3000 could’ve gone to emergency fund.\n"
    )


@tool("Budget Allocator")
def budget_allocator(spending_data: str, user_name: str = "you") -> str:
    """
    Suggests ideal budget split for the user using 50/30/20 rule or custom allocation logic.
    Ensures essential, discretionary, and savings categories are balanced relative to income.
    """
    return (
        f"📋 Recommended Budget Split for {user_name}:\n"
        "- Essentials (50%): ₹30,000\n"
        "- Savings (20%): ₹12,000\n"
        "- Discretionary (30%): ₹18,000\n"
    )


@tool("Utility Evaluator")
def utility_evaluator(spending_feedback: str, user_name: str = "you") -> str:
    """
    Evaluates user’s satisfaction from discretionary spending.
    Flags low-utility purchases where emotional reward doesn't justify the expense.
    """
    return (
        f"🧠 Utility Review for {user_name}:\n"
        "- Retail therapy (₹5000): Low satisfaction (2/5). Try replacing with hobby-related spend.\n"
        "- Frequent takeout (₹3500): Moderate satisfaction (3/5), consider home cooking more often.\n"
    )

# *************************************************Tools used for Financial Analysis*****************************************************

@tool("debt_repayment_tool")
def debt_repayment_tool():
    """
    Suggests a detailed monthly debt repayment plan using the Avalanche Method 
    (paying off the highest interest debt first), or Snowball (smallest balance first),
    based on user preference or default strategy.
    """
    return {
        "debt_plan": f"Pay ₹ per month for years at % interest.",
        "monthly_debt_payment" : f"₹"
    }


@tool("portfolio_simulation_tool")
def portfolio_simulation_tool():
    """
    Simulates expected portfolio growth using a compound return model.
    Returns expected growth in one year based on risk appetite.
    """

    return {
        "portfolio_growth": f"Expected portfolio growth is % over months.",
        "final_estimated_balance": f"₹"
    }

@tool("emergency_fund_tool")
def emergency_fund_tool():
    """
    Calculates the required emergency fund target based on expenses and dependents.
    Best practice: 6 months of expenses if no dependents, 9+ months if dependents.
    """

    return {
        "fund_target": f"Recommended emergency fund is ₹ (months of expenses).",
    }

# ***************************************************Tool using RL framewok on log data**************************************************
@tool("Rl tool")
def rl_lite_tool(user_logs: list) -> dict:
    """
    Implements the RL-lite logic by scoring discipline vs greed based on user logs.
    """
    return {
        "Discipline_score": f"Discipline for the user through past logs is "
    }

# ***************************************************Tool used to do karmic analysis******************************************************
@tool("Karma Tracker")
def karma_tracker_tool(user_logs: list) -> dict:
    """
    Assigns karmic value based on sattvic/rajasic/tamasic traits using user logs and Define symbolic traits for financial behavior (e.g. excessive consumption = tamasic, charity = sattvic).
    """
    return {
        "Karmic_value": f"Karmic value through past logs is "
    }

# *******************************************************Common cache logic***************************************************************
def cache_if_long_response(args, result):
    return len(result) > 50

opportunity_cost_analyzer.cache_function = cache_if_long_response
budget_allocator.cache_function = cache_if_long_response
utility_evaluator.cache_function = cache_if_long_response
debt_repayment_tool.cache_function = cache_if_long_response
portfolio_simulation_tool.cache_function = cache_if_long_response
emergency_fund_tool.cache_function = cache_if_long_response
karma_tracker_tool.cache_function = cache_if_long_response
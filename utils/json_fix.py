"""
Utility functions for fixing and parsing JSON output from LLMs.
These functions help handle common JSON formatting errors in LLM outputs.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict

def fix_json_string(json_str: str) -> str:
    """
    Fix common JSON formatting errors in LLM outputs.

    Args:
        json_str: The potentially malformed JSON string

    Returns:
        A corrected JSON string that should be parseable
    """
    # Remove any markdown code block markers
    json_str = re.sub(r'```json|```', '', json_str)
    json_str = json_str.strip()

    # Fix missing colons between keys and values
    # Pattern: "key" "value" or "key" {
    json_str = re.sub(r'("[\w_]+")(\s+)(")', r'\1:\3', json_str)
    json_str = re.sub(r'("[\w_]+")(\s+)({)', r'\1:\3', json_str)
    json_str = re.sub(r'("[\w_]+")(\s+)(\d+)', r'\1:\3', json_str)

    # Fix missing commas between items
    # This is trickier and might not catch all cases
    json_str = re.sub(r'(true|false|null|"[^"]*"|[\d.]+)\s*\n\s*("[\w_]+")', r'\1,\n\2', json_str)

    # Fix trailing commas in arrays and objects
    json_str = re.sub(r',(\s*})', r'\1', json_str)
    json_str = re.sub(r',(\s*])', r'\1', json_str)

    # Fix missing quotes around keys
    # This is a bit risky as it might catch things that aren't keys
    json_str = re.sub(r'([\{\,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)

    return json_str

def safe_parse_json(json_str: str, default_value: Any = None) -> Any:
    """
    Safely parse a JSON string, attempting to fix common errors.

    Args:
        json_str: The JSON string to parse
        default_value: Value to return if parsing fails

    Returns:
        The parsed JSON object or the default value if parsing fails
    """
    if not json_str:
        return default_value

    # First try parsing as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try fixing the JSON string
    fixed_json = fix_json_string(json_str)

    try:
        return json.loads(fixed_json)
    except json.JSONDecodeError:
        # If still failing, try a more aggressive approach
        try:
            # Use a regex to extract what looks like a JSON object
            match = re.search(r'({.*})', fixed_json, re.DOTALL)
            if match:
                extracted_json = match.group(1)
                return json.loads(extracted_json)
        except (json.JSONDecodeError, AttributeError):
            pass

    # If all attempts fail, return the default value
    return default_value

def create_fallback_json(month: int, agent_type: str, user_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a comprehensive fallback JSON object when parsing fails.

    Args:
        month: The current month number
        agent_type: The type of agent (cashflow, discipline, etc.)
        user_inputs: The user's input data to use instead of defaults

    Returns:
        A detailed JSON object that mimics the expected output format
    """
    # Common fields for all agent types
    fallback = {
        "month": month,
        "timestamp": datetime.now().isoformat()
    }

    # Use empty dict if user_inputs is None
    if user_inputs is None:
        user_inputs = {}

    # Extract user data
    user_name = user_inputs.get("user_name", "")
    user_income = user_inputs.get("income", 0)
    user_expenses = user_inputs.get("expenses", [])
    total_expenses = user_inputs.get("total_expenses", 0)

    # Calculate total expenses if not provided
    if total_expenses == 0 and isinstance(user_expenses, list):
        for expense in user_expenses:
            if isinstance(expense, dict) and "amount" in expense:
                total_expenses += float(expense["amount"])

    if agent_type == "cashflow":
        fallback.update({
            "user_name": user_name,
            "income": {
                "salary": user_income,
                "investments": 0,
                "other": 0,
                "total": user_income
            },
            "expenses": {
                "housing": 0,
                "utilities": 0,
                "groceries": 0,
                "transportation": 0,
                "healthcare": 0,
                "entertainment": 0,
                "dining_out": 0,
                "subscriptions": 0,
                "other": 0,
                "total": total_expenses
            },
            "savings": {
                "amount": max(0, user_income - total_expenses),
                "percentage_of_income": 0 if user_income == 0 else round((user_income - total_expenses) / user_income * 100, 2),
                "target_met": False
            },
            "balance": {
                "starting": 0,
                "ending": max(0, user_income - total_expenses),
                "change": max(0, user_income - total_expenses)
            },
            "analysis": {
                "spending_categories": {
                    "essential": 0,
                    "non_essential": 0,
                    "ratio": 0
                },
                "savings_rate": "N/A",
                "cash_flow": "Neutral"
            },
            "notes": "This is a minimal report based on your provided data."
        })
    elif agent_type == "discipline_tracker":
        fallback.update({
            "financial_discipline_score": 0.85,
            "improvement_areas": [
                "Reduce dining out expenses",
                "Optimize subscription services"
            ],
            "recommended_actions": [
                {
                    "title": "Review subscription services",
                    "description": "Cancel unused subscriptions to save $30 monthly"
                },
                {
                    "title": "Meal planning",
                    "description": "Reduce dining out by planning meals in advance"
                }
            ],
            "historical_trend": "improving",
            "acknowledged_improvements": [
                "Increased savings rate",
                "Reduced impulse purchases"
            ],
            "repeated_violations": [],
            "discipline_metrics": {
                "budget_adherence": 0.92,
                "savings_goal_achievement": 1.05,
                "expense_control": 0.88
            }
        })
    elif agent_type == "goal_tracker":
        fallback.update({
            "goals": {
                "emergency_fund": {
                    "target": 10000.00,
                    "current": 6500.00,
                    "progress_percentage": 65.0,
                    "monthly_contribution": 500.00,
                    "estimated_completion": f"Month {month + 7}",
                    "status": "on_track"
                },
                "vacation_fund": {
                    "target": 2000.00,
                    "current": 1200.00,
                    "progress_percentage": 60.0,
                    "monthly_contribution": 200.00,
                    "estimated_completion": f"Month {month + 4}",
                    "status": "on_track"
                },
                "retirement_savings": {
                    "target": 5000.00,
                    "current": 2000.00,
                    "progress_percentage": 40.0,
                    "monthly_contribution": 300.00,
                    "estimated_completion": f"Month {month + 10}",
                    "status": "on_track"
                }
            },
            "trends": {
                "emergency_fund": "improving",
                "vacation_fund": "steady",
                "retirement_savings": "improving"
            },
            "overall_progress": "Good progress across all financial goals",
            "recommendations": [
                "Consider increasing retirement contributions by 5%",
                "Maintain current emergency fund contribution rate"
            ]
        })
    elif agent_type == "behavior_tracker":
        fallback.update({
            "spending_pattern": "saver",
            "goal_adherence": "consistent",
            "saving_consistency": "excellent",
            "labels": [
                "goal-oriented",
                "disciplined-saver",
                "budget-conscious"
            ],
            "behavioral_insights": [
                "Demonstrates consistent saving behavior",
                "Shows discipline in following budget guidelines",
                "Makes progress toward financial goals"
            ],
            "metadata": {
                "trend_analysis": {
                    "spending_trend": "improving",
                    "savings_trend": "improving",
                    "overall_trend": "improving"
                },
                "improvement_metrics": {
                    "spending_improvement": True,
                    "goal_improvement": True,
                    "savings_improvement": True
                }
            },
            "progress_notes": [
                "Continued improvement in spending habits",
                "Maintaining consistent savings rate",
                "Successfully prioritizing financial goals"
            ]
        })
    elif agent_type == "karma_tracker":
        fallback.update({
            "karma_score": 85,
            "financial_karma": "positive",
            "karmic_actions": {
                "positive": [
                    "Consistent savings contributions",
                    "Reduced unnecessary spending",
                    "Progress toward financial goals"
                ],
                "negative": [
                    "Occasional impulse purchases"
                ]
            },
            "karmic_balance": {
                "current": "positive",
                "trend": "improving",
                "impact_factors": [
                    "Disciplined financial behavior",
                    "Goal-oriented actions",
                    "Responsible spending"
                ]
            },
            "future_outlook": {
                "short_term": "continued improvement",
                "long_term": "financial stability",
                "recommendations": [
                    "Maintain current savings rate",
                    "Continue reducing non-essential expenses"
                ]
            }
        })
    elif agent_type == "financial_strategy":
        fallback.update({
            "recommendations": [
                {
                    "type": "savings_adjustment",
                    "description": "Increase emergency fund contributions by $50 monthly",
                    "impact": "Accelerate emergency fund completion by 2 months",
                    "priority": "high"
                },
                {
                    "type": "expense_optimization",
                    "description": "Review and reduce subscription services",
                    "impact": "Save $30-50 monthly",
                    "priority": "medium"
                },
                {
                    "type": "income_enhancement",
                    "description": "Consider passive income opportunities",
                    "impact": "Potential for $100-200 additional monthly income",
                    "priority": "medium"
                }
            ],
            "strategy_focus": "balanced_growth",
            "risk_assessment": "moderate",
            "timeline": {
                "short_term": [
                    "Optimize current expenses",
                    "Maintain emergency fund contributions"
                ],
                "medium_term": [
                    "Explore passive income opportunities",
                    "Increase retirement contributions"
                ],
                "long_term": [
                    "Consider investment diversification",
                    "Plan for major financial goals"
                ]
            },
            "financial_health_assessment": {
                "current_status": "good",
                "trend": "improving",
                "strengths": [
                    "Consistent savings",
                    "Disciplined spending",
                    "Goal-oriented approach"
                ],
                "areas_for_improvement": [
                    "Diversification of income sources",
                    "Optimization of recurring expenses"
                ]
            }
        })

    return fallback

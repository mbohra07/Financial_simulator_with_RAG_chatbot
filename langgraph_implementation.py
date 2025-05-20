"""
LangGraph implementation of the Financial Crew simulation system.
This replaces the CrewAI implementation in crew.py.
Integrated with MongoDB Atlas for persistent storage and learning from past simulations.
"""

from typing import Dict, List, Any, TypedDict, Annotated, Literal, Optional, Union
import json
import os
from datetime import datetime
import time
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug

import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
from functions.economic_context import EconomicEnvironment, simulate_monthly_market
from functions.monthly_simulation import deduplicate_and_save, assign_persona, generate_monthly_reflection_report
from functions.task_functions import (
    build_simulated_cashflow_context,
    build_discipline_report_context,
    build_goal_status_context,
    build_behavior_tracker_context,
    build_karmic_tracker_context,
    build_financial_strategy_context
)

# Import MongoDB client
from database.mongodb_client import (
    save_user_input,
    save_agent_output,
    get_previous_month_outputs,
    get_agent_outputs_for_month,
    generate_simulation_id
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
        # Default to Groq
        return ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2,
            max_tokens=4000
        )

# Define state schema
class FinancialSimulationState(TypedDict):
    """State for the financial simulation workflow."""
    # Input data
    user_inputs: Dict[str, Any]
    month_number: int
    simulation_id: str

    # Simulation results
    cashflow_result: Optional[Dict[str, Any]]
    discipline_result: Optional[Dict[str, Any]]
    goal_tracking_result: Optional[Dict[str, Any]]
    behavior_result: Optional[Dict[str, Any]]
    karma_result: Optional[Dict[str, Any]]
    financial_strategy_result: Optional[Dict[str, Any]]

    # Context for agents
    cashflow_context: Optional[str]
    discipline_context: Optional[str]
    goal_tracking_context: Optional[str]
    behavior_context: Optional[str]
    karma_context: Optional[str]
    financial_strategy_context: Optional[str]

    # Economic data
    economic_context: Dict[str, float]
    market_context: str

    # Previous month data from MongoDB
    previous_month_data: Optional[Dict[str, List[Dict[str, Any]]]]

# Define agent nodes
def simulate_cashflow_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Simulate cash flow for the current month."""
    print(f"🟢 Executing task: simulate_cashflow for month {state['month_number']}")

    # Load agent config from YAML
    with open("config/agents.yaml", "r") as f:
        import yaml
        agents_config = yaml.safe_load(f)

    # Load task config from YAML
    with open("config/tasks.yaml", "r") as f:
        import yaml
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("simulate_cashflow_task", {}).get("description", "")

    # Get previous month data if available
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_cashflow_data = state["previous_month_data"].get("cashflow", [])
        if previous_cashflow_data:
            previous_month_context = f"""
\n\nPrevious Month Cashflow Data: {json.dumps(previous_cashflow_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Analyze the trends from the previous month's cashflow data
2. Consider how income and expenses changed from previous month
3. Note any unexpected spending or income patterns
4. Identify areas where the user could improve financial management
5. Your simulation for the current month should demonstrate learning from these past patterns
6. Show progressive improvement in financial recommendations based on historical data
"""
            print(f"📊 Using previous month cashflow data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial cashflow simulation assistant. Always respond ONLY with valid JSON.

As you analyze this month's data, explicitly consider how it compares to previous months.
Show progressive learning by adapting recommendations based on what worked or didn't work before.
Your simulation should demonstrate continuity and improvement over time.
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nEconomic Context: {economic_context}\nMarket Context: {market_context}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("emotional_bias_agent", {}).get("llm", "groq/llama3-70b-8192"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        result = chain.invoke({
            "user_inputs": state["user_inputs"],
            "economic_context": state["economic_context"],
            "market_context": state["market_context"]
        })

        # Save result to file
        output_path = f"output/{user_id}_simulated_cashflow_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        # Save to file system
        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "cashflow"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "cashflow_result": result
        }
    except Exception as e:
        print(f"❌ Error in simulate_cashflow_node: {e}")
        return state

def discipline_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial discipline for the current month."""
    print(f"🟢 Executing task: discipline_tracker for month {state['month_number']}")

    # Build context from previous cashflow results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    cashflow_context = build_simulated_cashflow_context(month, user_id)

    # Load agent config from YAML
    with open("config/agents.yaml", "r") as f:
        import yaml
        agents_config = yaml.safe_load(f)

    # Load task config from YAML
    with open("config/tasks.yaml", "r") as f:
        import yaml
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("discipline_tracker_task", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_discipline_data = state["previous_month_data"].get("discipline", [])
        if previous_discipline_data:
            previous_month_context = f"""
\n\nPrevious Month Discipline Data: {json.dumps(previous_discipline_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Compare current month's financial behavior with previous month's discipline score and violations
2. Identify if the user has improved or worsened in specific areas
3. Recognize patterns of repeated violations or improvements
4. Provide more targeted recommendations based on historical discipline issues
5. Acknowledge improvements where the user has followed previous recommendations
6. Adjust discipline scoring to reflect progressive learning and improvement
"""
            print(f"📊 Using previous month discipline data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial discipline tracking assistant. Always respond ONLY with valid JSON.

Your role is to track financial discipline over time and show progressive learning:
1. Compare current behavior with previous months' patterns
2. Recognize improvements or regressions in financial discipline
3. Provide increasingly personalized recommendations based on historical data
4. Adjust your scoring to reflect the user's learning journey
5. Be more strict about repeated violations and more rewarding of consistent improvements
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Context: {cashflow_context}\nCashflow Result: {cashflow_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("discipline_tracker_agent", {}).get("llm", "groq/llama3-70b-8192"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        result = chain.invoke({
            "user_inputs": state["user_inputs"],
            "cashflow_context": cashflow_context,
            "cashflow_result": state["cashflow_result"]
        })

        # Save result to file
        output_path = f"output/{user_id}_discipline_report_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        # Save to file system
        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "discipline_tracker"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "discipline_result": result,
            "discipline_context": cashflow_context
        }
    except Exception as e:
        print(f"❌ Error in discipline_tracker_node: {e}")
        return state

def goal_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial goals for the current month."""
    print(f"🟢 Executing task: goal_tracker for month {state['month_number']}")

    # Build context from previous results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    goal_context = build_goal_status_context(month, user_id)

    # Load agent config from YAML
    with open("config/agents.yaml", "r") as f:
        import yaml
        agents_config = yaml.safe_load(f)

    # Load task config from YAML
    with open("config/tasks.yaml", "r") as f:
        import yaml
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("track_goals", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_goal_data = state["previous_month_data"].get("goal", [])
        if previous_goal_data:
            previous_month_context = f"""
\n\nPrevious Month Goal Data: {json.dumps(previous_goal_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Track progress toward goals across multiple months, not just the current month
2. Compare current goal progress with previous month's status
3. Identify if the user is consistently meeting savings targets for each goal
4. Adjust expectations and recommendations based on historical performance
5. Provide more targeted goal adjustment suggestions based on past behavior
6. Recognize and reward consistent progress toward goals
"""
            print(f"📊 Using previous month goal data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial goal tracking assistant. Always respond ONLY with valid JSON.

Your role is to track financial goals over time and demonstrate progressive learning:
1. Monitor cumulative progress toward goals across all months
2. Identify trends in goal achievement (improving, declining, or stagnant)
3. Provide increasingly personalized goal adjustments based on historical performance
4. Recognize when goals need to be recalibrated based on consistent patterns
5. Show how current month's progress builds on previous months' achievements
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Result: {cashflow_result}\nDiscipline Result: {discipline_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("goal_tracker_agent", {}).get("llm", "groq/llama-3.1-70b-versatile"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        result = chain.invoke({
            "user_inputs": state["user_inputs"],
            "cashflow_result": state["cashflow_result"],
            "discipline_result": state["discipline_result"]
        })

        # Save result to file
        output_path = f"output/{user_id}_goal_status_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "goal_tracker"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "goal_tracking_result": result,
            "goal_tracking_context": goal_context
        }
    except Exception as e:
        print(f"❌ Error in goal_tracker_node: {e}")
        return state

def behavior_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial behavior for the current month."""
    print(f"🟢 Executing task: behavior_tracker for month {state['month_number']}")

    # Build context from previous results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    behavior_context = build_behavior_tracker_context(month, user_id)

    # Load agent config from YAML
    with open("config/agents.yaml", "r") as f:
        import yaml
        agents_config = yaml.safe_load(f)

    # Load task config from YAML
    with open("config/tasks.yaml", "r") as f:
        import yaml
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("behavior_tracker_task", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_behavior_data = state["previous_month_data"].get("behavior", [])
        if previous_behavior_data:
            previous_month_context = f"""
\n\nPrevious Month Behavior Data: {json.dumps(previous_behavior_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Compare current financial behaviors with patterns from previous months
2. Identify behavioral improvements or regressions over time
3. Note recurring behavioral patterns that affect financial outcomes
4. Provide more targeted behavioral insights based on historical patterns
5. Recognize when the user has successfully changed problematic behaviors
6. Adjust behavioral recommendations based on what has or hasn't worked in the past
"""
            print(f"📊 Using previous month behavior data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial behavior tracking assistant. Always respond ONLY with valid JSON.

Your role is to analyze financial behaviors over time and demonstrate progressive learning:
1. Track behavioral patterns across multiple months, not just the current month
2. Identify trends in financial decision-making (improving, declining, or stagnant)
3. Provide increasingly personalized behavioral insights based on historical patterns
4. Recognize when behavioral patterns need special attention based on consistent issues
5. Show how current behaviors compare to previous months and highlight improvements
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Result: {cashflow_result}\nDiscipline Result: {discipline_result}\nGoal Tracking Result: {goal_tracking_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("behavior_tracker_agent", {}).get("llm", "groq/llama-3.1-70b-versatile"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        result = chain.invoke({
            "user_inputs": state["user_inputs"],
            "cashflow_result": state["cashflow_result"],
            "discipline_result": state["discipline_result"],
            "goal_tracking_result": state["goal_tracking_result"]
        })

        # Save result to file
        output_path = f"output/{user_id}_behavior_tracker_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "behavior_tracker"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "behavior_result": result,
            "behavior_context": behavior_context
        }
    except Exception as e:
        print(f"❌ Error in behavior_tracker_node: {e}")
        return state

def karma_tracker_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Track financial karma for the current month."""
    print(f"🟢 Executing task: karma_tracker for month {state['month_number']}")

    # Build context from previous results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    karma_context = build_karmic_tracker_context(month, user_id)

    # Load agent config from YAML
    with open("config/agents.yaml", "r") as f:
        import yaml
        agents_config = yaml.safe_load(f)

    # Load task config from YAML
    with open("config/tasks.yaml", "r") as f:
        import yaml
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("karma_tracker_task", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_karma_data = state["previous_month_data"].get("karma", [])
        if previous_karma_data:
            previous_month_context = f"""
\n\nPrevious Month Karma Data: {json.dumps(previous_karma_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Compare current financial karma with patterns from previous months
2. Identify karmic improvements or regressions over time
3. Note recurring patterns in how financial decisions affect overall well-being
4. Provide more targeted karmic insights based on historical patterns
5. Recognize when the user has successfully improved their financial karma
6. Adjust karmic recommendations based on what has or hasn't worked in the past
"""
            print(f"📊 Using previous month karma data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial karma tracking assistant. Always respond ONLY with valid JSON.

Your role is to analyze financial karma over time and demonstrate progressive learning:
1. Track karmic patterns across multiple months, not just the current month
2. Identify trends in how financial decisions affect overall well-being
3. Provide increasingly personalized karmic insights based on historical patterns
4. Recognize when karmic patterns need special attention based on consistent issues
5. Show how current karma compares to previous months and highlight improvements
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Result: {cashflow_result}\nDiscipline Result: {discipline_result}\nGoal Tracking Result: {goal_tracking_result}\nBehavior Result: {behavior_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("karma_tracker_agent", {}).get("llm", "groq/llama-3.1-70b-versatile"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        result = chain.invoke({
            "user_inputs": state["user_inputs"],
            "cashflow_result": state["cashflow_result"],
            "discipline_result": state["discipline_result"],
            "goal_tracking_result": state["goal_tracking_result"],
            "behavior_result": state["behavior_result"]
        })

        # Save result to file
        output_path = f"output/{user_id}_karmic_tracker_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "karma_tracker"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "karma_result": result,
            "karma_context": karma_context
        }
    except Exception as e:
        print(f"❌ Error in karma_tracker_node: {e}")
        return state

def financial_strategy_node(state: FinancialSimulationState) -> FinancialSimulationState:
    """Generate financial strategy for the current month."""
    print(f"🟢 Executing task: financial_strategy for month {state['month_number']}")

    # Build context from previous results
    user_id = state["user_inputs"].get("user_id", "default_user")
    month = state["month_number"]
    strategy_context = build_financial_strategy_context(month, user_id)

    # Load agent config from YAML
    with open("config/agents.yaml", "r") as f:
        import yaml
        agents_config = yaml.safe_load(f)

    # Load task config from YAML
    with open("config/tasks.yaml", "r") as f:
        import yaml
        tasks_config = yaml.safe_load(f)

    # Get the task description
    task_description = tasks_config.get("financial_strategy_task", {}).get("description", "")

    # Get previous month data if available
    previous_month_context = ""

    if month > 1 and state.get("previous_month_data"):
        previous_strategy_data = state["previous_month_data"].get("strategy", [])
        if previous_strategy_data:
            previous_month_context = f"""
\n\nPrevious Month Strategy Data: {json.dumps(previous_strategy_data, indent=2)}

IMPORTANT LEARNING INSTRUCTIONS:
1. Evaluate the effectiveness of previous month's financial strategies
2. Identify which strategies worked well and which didn't achieve desired outcomes
3. Build upon successful strategies and modify or replace unsuccessful ones
4. Consider how the user's financial situation has evolved over time
5. Provide more targeted and personalized strategies based on historical performance
6. Show progressive improvement in strategy recommendations based on what you've learned
"""
            print(f"📊 Using previous month strategy data for month {month-1}")

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a financial strategy assistant. Always respond ONLY with valid JSON.

Your role is to develop financial strategies that demonstrate progressive learning over time:
1. Evaluate the effectiveness of previous strategies before making new recommendations
2. Build upon what has worked well in the past and avoid repeating unsuccessful approaches
3. Provide increasingly personalized strategies based on the user's unique financial journey
4. Show how your recommendations have evolved based on the user's changing circumstances
5. Create a sense of continuity and improvement in financial planning across months
"""),
        HumanMessage(content=task_description +
                    "\n\nUser Inputs: {user_inputs}\nCashflow Result: {cashflow_result}\nDiscipline Result: {discipline_result}\nGoal Tracking Result: {goal_tracking_result}\nBehavior Result: {behavior_result}\nKarma Result: {karma_result}" +
                    previous_month_context)
    ])

    # Get LLM
    llm = get_llm(agents_config.get("financial_strategy_agent", {}).get("llm", "groq/llama-3.1-70b-versatile"))

    # Create chain
    chain = prompt | llm | JsonOutputParser()

    # Execute chain
    try:
        result = chain.invoke({
            "user_inputs": state["user_inputs"],
            "cashflow_result": state["cashflow_result"],
            "discipline_result": state["discipline_result"],
            "goal_tracking_result": state["goal_tracking_result"],
            "behavior_result": state["behavior_result"],
            "karma_result": state["karma_result"]
        })

        # Save result to file
        output_path = f"output/{user_id}_financial_strategy_simulation.json"

        # Ensure it's a list with the month number
        if isinstance(result, dict):
            result["month"] = month
            result = [result]
        elif isinstance(result, list) and result:
            for item in result:
                if isinstance(item, dict):
                    item["month"] = month

        deduplicate_and_save(output_path, result)

        # Save to MongoDB
        agent_name = "financial_strategy"
        save_agent_output(
            user_id=user_id,
            simulation_id=state["simulation_id"],
            month=month,
            agent_name=agent_name,
            output_data=result[0] if result else {}
        )
        print(f"💾 Saved {agent_name} output to MongoDB for month {month}")

        # Update state
        return {
            **state,
            "financial_strategy_result": result,
            "financial_strategy_context": strategy_context
        }
    except Exception as e:
        print(f"❌ Error in financial_strategy_node: {e}")
        return state

# Define the LangGraph workflow
def create_financial_simulation_graph():
    """Create the financial simulation workflow graph."""
    # Create the graph
    workflow = StateGraph(FinancialSimulationState)

    # Add nodes
    workflow.add_node("simulate_cashflow", simulate_cashflow_node)
    workflow.add_node("discipline_tracker", discipline_tracker_node)
    workflow.add_node("goal_tracker", goal_tracker_node)
    workflow.add_node("behavior_tracker", behavior_tracker_node)
    workflow.add_node("karma_tracker", karma_tracker_node)
    workflow.add_node("financial_strategy", financial_strategy_node)

    # Define the edges (sequential workflow)
    workflow.add_edge("simulate_cashflow", "discipline_tracker")
    workflow.add_edge("discipline_tracker", "goal_tracker")
    workflow.add_edge("goal_tracker", "behavior_tracker")
    workflow.add_edge("behavior_tracker", "karma_tracker")
    workflow.add_edge("karma_tracker", "financial_strategy")
    workflow.add_edge("financial_strategy", END)

    # Set the entry point
    workflow.set_entry_point("simulate_cashflow")

    # Compile the graph
    return workflow.compile()

# Main simulation function
def simulate_timeline_langgraph(n_months: int, simulation_unit: str, user_inputs: dict, task_id: str = None):
    """Run the financial simulation for multiple months using LangGraph."""
    print(f"🚀 Starting LangGraph Financial Simulation for {n_months} {simulation_unit}...")

    # Create the workflow graph
    workflow = create_financial_simulation_graph()

    # Ensure user_id exists
    if "user_id" not in user_inputs:
        user_inputs["user_id"] = "default_user"

    # Ensure user_name exists
    if "user_name" not in user_inputs:
        user_inputs["user_name"] = "Default User"

    # Generate a unique simulation ID
    simulation_id = generate_simulation_id()
    print(f"📝 Simulation ID: {simulation_id}")

    # Save user input to MongoDB
    save_user_input(user_inputs, simulation_id)
    print(f"💾 Saved user input to MongoDB")

    # Run simulation for each month
    for month in range(1, n_months + 1):
        print(f"\n🔄 Simulating Month {month} of {n_months}")

        # Update task status if task_id is provided
        if task_id:
            print(f"📊 Updating task status: {month}/{n_months} months completed")

        # Simulate economic environment
        eco_env = EconomicEnvironment(unit=simulation_unit)
        eco_env.simulate_step()
        eco_context = eco_env.get_context()

        # Simulate market conditions
        market_snapshot, market_context_summary = simulate_monthly_market()

        # Update user inputs with economic data
        month_inputs = user_inputs.copy()
        month_inputs["Month"] = month
        month_inputs["market_context"] = market_context_summary
        month_inputs["inflation"] = eco_context["inflation_rate"]
        month_inputs["interest_rate"] = eco_context["interest_rate"]
        month_inputs["cost_of_living_index"] = eco_context["cost_of_living_index"]

        # Fetch previous month data from MongoDB if not month 1
        previous_month_data = None
        if month > 1:
            user_id = user_inputs["user_id"]
            previous_month = month - 1

            # Get cashflow data
            cashflow_data = get_agent_outputs_for_month(user_id, previous_month, "cashflow")

            # Get discipline data
            discipline_data = get_agent_outputs_for_month(user_id, previous_month, "discipline_tracker")

            # Get goal tracking data
            goal_data = get_agent_outputs_for_month(user_id, previous_month, "goal_tracker")

            # Get behavior data
            behavior_data = get_agent_outputs_for_month(user_id, previous_month, "behavior_tracker")

            # Get karma data
            karma_data = get_agent_outputs_for_month(user_id, previous_month, "karma_tracker")

            # Get financial strategy data
            strategy_data = get_agent_outputs_for_month(user_id, previous_month, "financial_strategy")

            # Compile all data
            previous_month_data = {
                "cashflow": [item["data"] for item in cashflow_data],
                "discipline": [item["data"] for item in discipline_data],
                "goal": [item["data"] for item in goal_data],
                "behavior": [item["data"] for item in behavior_data],
                "karma": [item["data"] for item in karma_data],
                "strategy": [item["data"] for item in strategy_data]
            }

            print(f"📊 Fetched previous month data from MongoDB for month {previous_month}")

        # Initialize state
        initial_state = {
            "user_inputs": month_inputs,
            "month_number": month,
            "simulation_id": simulation_id,
            "cashflow_result": None,
            "discipline_result": None,
            "goal_tracking_result": None,
            "behavior_result": None,
            "karma_result": None,
            "financial_strategy_result": None,
            "cashflow_context": None,
            "discipline_context": None,
            "goal_tracking_context": None,
            "behavior_context": None,
            "karma_context": None,
            "financial_strategy_context": None,
            "economic_context": eco_context,
            "market_context": market_context_summary,
            "previous_month_data": previous_month_data
        }

        # Run the workflow
        try:
            # Execute the workflow directly instead of streaming
            result = workflow.invoke(initial_state)

            # Generate monthly reflection report
            user_name = user_inputs["user_name"]
            assign_persona(user_name, month)
            generate_monthly_reflection_report(user_name, month)

            print(f"✅ Month {month} simulation completed successfully")

        except Exception as e:
            print(f"❌ Error in month {month} simulation: {e}")
            import traceback
            traceback.print_exc()

        # Add delay between months to avoid rate limits
        if month < n_months:
            print(f"⏳ Waiting before starting next month simulation...")
            time.sleep(15)

    print(f"🎉 Financial simulation completed for {n_months} {simulation_unit}")
    return True

# For testing
if __name__ == "__main__":
    # Test inputs
    test_inputs = {
        "user_id": "test_user",
        "user_name": "Test User",
        "age": 30,
        "occupation": "Software Engineer",
        "income_level": "50,000-100,000",
        "goal": "Save $10,000 for emergency fund",
        "starting_balance": 5000,
        "monthly_earning": 6000,
        "monthly_expenses": 4500,
        "savings_target": 1500
    }

    # Run simulation for 2 months
    simulate_timeline_langgraph(2, "Months", test_inputs)

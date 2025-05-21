"""
Streamlit application for the Financial Crew simulation using LangGraph.
This replaces the CrewAI implementation in streamlit_app.py.
Integrated with MongoDB Atlas for persistent storage and learning from past simulations.
"""

import streamlit as st
import json
import os
import pandas as pd
import time
import agentops
from langgraph_implementation import simulate_timeline_langgraph
from functions.economic_context import EconomicEnvironment

# Import MongoDB client
from database.mongodb_client import (
    get_all_agent_outputs_for_user,
    get_agent_outputs_for_month
)

# Initialize AgentOps
agentops.init(
    api_key='4be58a32-e415-4142-82b7-834ae6b95422',
    default_tags=['langgraph']
)

# ************************************************Streamlit configuration************************************************************

st.set_page_config(
    page_title="🧠 Financial Agent Simulator (LangGraph)",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("📈 Financial Agent Simulation")

st.markdown("""
Welcome to your **Personal Financial Simulation** powered by LangGraph. Simulate months of financial life, get guidance,
and improve your money habits with AI agents!
""")

# Sidebar for navigation
with st.sidebar:
    st.header("Simulation Navigation")
    display_option = st.radio(
        "View Results",
        options = [
            "💰 Cash Flow",
            "🎯 Goal Tracking",
            "✅ Discipline Tracker",
            "🧠 Behavior Tracker",
            "🌱 Karma Tracker",
            "📈 Financial Strategy"
        ],
        index=0
    )
    st.markdown("---")
    st.caption("ℹ️ Run a new simulation to update all reports")

# Function to display cash flow data
def display_cash_flow():
    st.header("💰 Monthly Cash Flow")

    try:
        # First try to get data from MongoDB
        mongo_data = get_all_agent_outputs_for_user(st.session_state.user_id)

        # Filter for cashflow data
        cashflow_data = [item["data"] for item in mongo_data if item.get("agent_name") == "cashflow_simulator"]

        if cashflow_data:
            data = cashflow_data
            st.success("Using data from MongoDB")
        else:
            # Fallback to file system
            file_path = f"output/{st.session_state.user_id}_simulated_cashflow_simulation.json"
            if not os.path.exists(file_path):
                st.warning("No cash flow data available. Please run a simulation first.")
                return

            with open(file_path, "r") as f:
                data = json.load(f)

            if not data:
                st.warning("No cash flow data available. Please run a simulation first.")
                return

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Summary", "Income Breakdown", "Expense Breakdown"])

        with tab1:
            # Create a summary table
            summary_data = []
            for entry in data:
                month = entry.get("month", "N/A")
                income = entry.get("income", {}).get("total", 0)
                expenses = entry.get("expenses", {}).get("total", 0)
                savings = entry.get("savings", 0)
                savings_rate = entry.get("savings_rate", 0)

                summary_data.append({
                    "Month": month,
                    "Income": f"₹{income:,.2f}",
                    "Expenses": f"₹{expenses:,.2f}",
                    "Savings": f"₹{savings:,.2f}",
                    "Savings Rate": f"{savings_rate:.1f}%"
                })

            if summary_data:
                st.dataframe(pd.DataFrame(summary_data))

                # Create a chart
                chart_data = pd.DataFrame({
                    "Month": [entry["Month"] for entry in summary_data],
                    "Income": [float(entry["Income"].replace("₹", "").replace(",", "")) for entry in summary_data],
                    "Expenses": [float(entry["Expenses"].replace("₹", "").replace(",", "")) for entry in summary_data],
                    "Savings": [float(entry["Savings"].replace("₹", "").replace(",", "")) for entry in summary_data]
                })

                st.line_chart(chart_data.set_index("Month"))

        with tab2:
            # Income breakdown
            st.subheader("Income Sources")

            for entry in data:
                month = entry.get("month", "N/A")
                income_sources = entry.get("income", {}).get("sources", [])

                if income_sources:
                    st.write(f"### Month {month}")

                    income_data = []
                    for source in income_sources:
                        income_data.append({
                            "Source": source.get("name", "Unknown"),
                            "Amount": f"₹{source.get('amount', 0):,.2f}"
                        })

                    st.dataframe(pd.DataFrame(income_data))

        with tab3:
            # Expense breakdown
            st.subheader("Expense Categories")

            for entry in data:
                month = entry.get("month", "N/A")
                expense_categories = entry.get("expenses", {}).get("categories", [])

                if expense_categories:
                    st.write(f"### Month {month}")

                    expense_data = []
                    for category in expense_categories:
                        expense_data.append({
                            "Category": category.get("name", "Unknown"),
                            "Amount": f"₹{category.get('amount', 0):,.2f}"
                        })

                    st.dataframe(pd.DataFrame(expense_data))

                    # Create a pie chart
                    import plotly.express as px
                    fig = px.pie(
                        values=[category.get("amount", 0) for category in expense_categories],
                        names=[category.get("name", "Unknown") for category in expense_categories],
                        title=f"Month {month} Expense Breakdown"
                    )
                    st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error displaying cash flow data: {e}")

# Function to display goal tracking data
def display_goal_tracking():
    st.header("🎯 Goal Tracking")

    try:
        # First try to get data from MongoDB
        mongo_data = get_all_agent_outputs_for_user(st.session_state.user_id)

        # Filter for goal tracking data
        goal_data = [item["data"] for item in mongo_data if item.get("agent_name") == "goal_tracker"]

        if goal_data:
            data = goal_data
            st.success("Using data from MongoDB")
        else:
            # Fallback to file system
            file_path = f"output/{st.session_state.user_id}_goal_status_simulation.json"
            if not os.path.exists(file_path):
                st.warning("No goal tracking data available. Please run a simulation first.")
                return

            with open(file_path, "r") as f:
                data = json.load(f)

            if not data:
                st.warning("No goal tracking data available. Please run a simulation first.")
                return

        # Display goals for each month
        for entry in data:
            month = entry.get("month", "N/A")
            goals = entry.get("goals", [])
            summary = entry.get("summary", {})

            st.write(f"## Month {month}")

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("On Track Goals", summary.get("on_track_goals", 0))
            with col2:
                st.metric("Behind Goals", summary.get("behind_goals", 0))
            with col3:
                st.metric("Total Saved", f"₹{summary.get('total_saved', 0):,.2f}")
            with col4:
                st.metric("Required by Now", f"₹{summary.get('total_required_by_now', 0):,.2f}")

            # Display individual goals
            for goal in goals:
                with st.expander(f"{goal.get('name', 'Unknown Goal')} - {goal.get('status', 'Unknown')}"):
                    st.write(f"**Target Amount:** ₹{goal.get('target_amount', 0):,.2f}")
                    st.write(f"**Saved So Far:** ₹{goal.get('saved_so_far', 0):,.2f}")
                    st.write(f"**Expected by Now:** ₹{goal.get('expected_by_now', 0):,.2f}")
                    st.write(f"**Priority:** {goal.get('priority', 'N/A')}")
                    st.write(f"**Adjustment Suggestion:** {goal.get('adjustment_suggestion', 'None')}")

                    # Create a progress bar
                    progress = goal.get('saved_so_far', 0) / goal.get('target_amount', 1)
                    st.progress(min(progress, 1.0))

    except Exception as e:
        st.error(f"Error displaying goal tracking data: {e}")

# Add other display functions for the remaining tabs
def display_discipline_tracker():
    st.header("✅ Discipline Tracker")

    try:
        # First try to get data from MongoDB
        mongo_data = get_all_agent_outputs_for_user(st.session_state.user_id)

        # Filter for discipline data
        discipline_data = [item["data"] for item in mongo_data if item.get("agent_name") == "discipline_tracker"]

        if discipline_data:
            data = discipline_data
            st.success("Using data from MongoDB")
        else:
            # Fallback to file system
            file_path = f"output/{st.session_state.user_id}_discipline_report_simulation.json"
            if not os.path.exists(file_path):
                st.warning("No discipline tracker data available. Please run a simulation first.")
                return

            with open(file_path, "r") as f:
                data = json.load(f)

            if not data:
                st.warning("No discipline tracker data available. Please run a simulation first.")
                return

        # Display discipline data for each month
        for entry in data:
            month = entry.get("month", "N/A")
            discipline_score = entry.get("discipline_score", 0)
            violations = entry.get("violations", [])
            recommendations = entry.get("recommendations", [])

            st.write(f"## Month {month}")

            # Display discipline score with gauge
            st.metric("Discipline Score", f"{discipline_score}/100")

            # Create a progress bar for the score
            st.progress(discipline_score / 100)

            # Display violations and recommendations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Violations")
                if violations:
                    for violation in violations:
                        st.write(f"- {violation}")
                else:
                    st.write("No violations detected!")

            with col2:
                st.subheader("Recommendations")
                if recommendations:
                    for recommendation in recommendations:
                        st.write(f"- {recommendation}")
                else:
                    st.write("No recommendations provided.")

    except Exception as e:
        st.error(f"Error displaying discipline tracker data: {e}")

# Main app logic
if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"

# Input form for simulation
with st.expander("Run New Simulation", expanded=True):
    with st.form("simulation_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Your Name")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            occupation = st.text_input("Occupation", "Software Engineer")
        with col2:
            income_level = st.selectbox("Income Level", ["<10,000", "10,000-50,000", "50,000-100,000", ">100,000"])
            goal = st.text_input("Financial Goal", "Save ₹50,000 for emergency fund")
            risk_level = st.select_slider("Risk Tolerance", options=["Low", "Medium", "High"])

        st.subheader("Financial Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            starting_balance = st.number_input("Starting Balance (₹)", min_value=0, value=10000)
            monthly_earning = st.number_input("Monthly Income (₹)", min_value=0, value=15000)
        with col2:
            monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=12000)
            savings_target = st.number_input("Monthly Savings Target (₹)", min_value=0, value=3000)
        with col3:
            simulation_months = st.number_input("Simulation Months", min_value=1, max_value=12, value=3)

        submit_button = st.form_submit_button("Run Simulation")

        if submit_button:
            with st.spinner("Running financial simulation..."):
                # Prepare user inputs
                user_inputs = {
                    "user_id": st.session_state.user_id,
                    "user_name": user_name,
                    "age": age,
                    "occupation": occupation,
                    "income_level": income_level,
                    "goal": goal,
                    "risk_level": risk_level,
                    "starting_balance": starting_balance,
                    "monthly_earning": monthly_earning,
                    "monthly_expenses": monthly_expenses,
                    "savings_target": savings_target,
                    "financial_type": "Balanced",  # Default value
                    "cashflow_context": "There is no previous summary"
                }

                # Show MongoDB info
                st.info("Simulation will store data in MongoDB for continuous learning")

                # Run the simulation
                result = simulate_timeline_langgraph(simulation_months, "Months", user_inputs)

                if result:
                    st.success("✅ Simulation Complete!")
                    st.balloons()

                    # Show MongoDB info
                    st.info("Data has been stored in MongoDB. Future simulations will learn from this data.")
                else:
                    st.error("Simulation failed after multiple attempts.")

# Display the selected content based on navigation
if display_option == "💰 Cash Flow":
    display_cash_flow()
elif display_option == "🎯 Goal Tracking":
    display_goal_tracking()
elif display_option == "✅ Discipline Tracker":
    display_discipline_tracker()
elif display_option == "🧠 Behavior Tracker":
    st.header("🧠 Behavior Tracker")
    st.info("This section is under development in the LangGraph version.")
elif display_option == "🌱 Karma Tracker":
    st.header("🌱 Karma Tracker")
    st.info("This section is under development in the LangGraph version.")
elif display_option == "📈 Financial Strategy":
    st.header("📈 Financial Strategy")
    st.info("This section is under development in the LangGraph version.")

st.markdown("---")
st.caption("""
ℹ️ This is a simulation tool powered by LangGraph. Actual financial results may vary based on real-world circumstances.
Use the insights to inform your decisions, but consult a financial advisor for personalized advice.
""")

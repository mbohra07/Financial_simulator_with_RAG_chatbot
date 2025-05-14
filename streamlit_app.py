# import streamlit as st
# from functions.streamlit_functions import *
# import agentops
# from functions.crew_functions import *

# agentops.init(
#     api_key='0c4bf935-54bd-42f0-8c82-9a044f1afe10',
#     default_tags=['crewai']
# )

# # ************************************************Streamlit configuration************************************************************

# st.set_page_config(
#     page_title="🧠 Financial Agent Simulator", 
#     layout="centered",
#     initial_sidebar_state="expanded"
# )
# st.title("📈 Financial Agent Simulation")

# st.markdown("""
# Welcome to your **Personal Financial Simulation**. Simulate months of financial life, get guidance, 
# and improve your money habits with AI agents!
# """)

# # Sidebar for navigation
# with st.sidebar:
#     st.header("Simulation Navigation")
#     display_option = st.radio(
#         "View Results",
#         options = [
#             "💰 Cash Flow",
#             "🎯 Goal Tracking",
#             "✅ Discipline Tracker",
#             "🧠 Behavior Tracker",
#             "🌱 Karma Tracker",
#             "📈 Financial Strategy"
#         ],
#         index=0
#     )
#     st.markdown("---")
#     st.caption("ℹ️ Run a new simulation to update all reports")

# # *************************************************Collecting user profile details*****************************************************
# # --- Collecting user profile details ---
# with st.form("financial_profile_form"):
#     st.subheader("👤 Basic Financial Profile")
#     user_id = st.number_input("Unique ID", min_value=1,)
#     user_name = st.text_input("Your Name")
#     income = st.number_input("Monthly Income (₹)", min_value=0.0, format="%.2f")

#     st.subheader("💸 Monthly Expenses")
#     expenses = []
#     total_expenses = 0.0
#     # Use a slider for selecting the number of expense categories
#     num_expenses = st.slider(
#         "Number of Expense Categories", 
#         min_value=1, 
#         max_value=10, 
#         value=2, 
#         step=1
#     )

#     expenses = []
#     total_expenses = 0.0

#     for i in range(num_expenses):
#         exp_col1, exp_col2 = st.columns(2)
#         with exp_col1:
#             exp_name = st.text_input(f"Expense {i+1} Name", key=f"exp_name_{i}")
#         with exp_col2:
#             exp_amount = st.number_input(f"Amount (₹)", min_value=0.0, format="%.2f", key=f"exp_amt_{i}")
#         if exp_name:
#             expenses.append({"name": exp_name, "amount": exp_amount})
#             total_expenses += exp_amount

#     st.subheader("🎯 Financial Goal")
#     goal = st.text_input("What's your financial goal? (e.g., 'Save ₹50,000 for emergency fund')")

#     st.subheader("🧑‍💼 Financial Type")
#     financial_type = st.selectbox(
#         "Choose your financial type:",
#         ["Conservative", "Balanced", "Aggressive"]
#     )

#     st.subheader("⚠️ Risk Level")
#     risk_level = st.select_slider(
#         "Select your risk tolerance:",
#         options=["Low", "Medium", "High"],
#         value="Medium"
#     )

#     submit = st.form_submit_button("💡 Run Financial Simulation")

# # ***************************************************Running my Crew workflow**********************************************************
# if submit:
#     with st.spinner("🔍 Simulating your financial journey..."):

#         balance = income - total_expenses
#         user_inputs = {
#             "user_id": user_id,
#             "user_name": user_name,
#             "income": income,
#             "expenses": expenses,
#             "total_expenses": total_expenses,
#             "goal": goal,
#             "financial_type": financial_type,
#             "risk_level": risk_level,
#             "balance": balance
#         }
#         custom_agents = {
#             'spending_advisor': {'goal': 'Save more money this month'},
#             'goal_tracker': {'goal': 'Increase savings target by 20%'}
#         }
#         custom_tasks = {
#             'simulate_cash_flow': {'expected_output': 'Custom output for this run'}
#         }
        
#         result = simulate_timeline(6, "Months" ,user_inputs)

#         if result:
#             st.success("✅ Simulation Complete!")
#             st.balloons()
#         else:
#             st.error("Simulation failed after multiple attempts.")

# # **********************************************Display the selected content based on navigation*******************************************

# elif display_option == "💰 Cash Flow":
#     display_cash_flow()
# elif display_option == "🎯 Goal Tracking":
#     display_goal_tracking()
# elif display_option == "✅ Discipline Tracker":
#     display_discipline_tracker()
# elif display_option == "🧠 Behavior Tracker":
#     display_behavior_tracker()
# elif display_option == "🌱 Karma Tracker":
#     display_karma_tracker()
# elif display_option == "📈 Financial Strategy":
#     display_financial_strategy()

# st.markdown("---")
# st.caption("""
# ℹ️ This is a simulation tool. Actual financial results may vary based on real-world circumstances.
# Use the insights to inform your decisions, but consult a financial advisor for personalized advice.
# """)
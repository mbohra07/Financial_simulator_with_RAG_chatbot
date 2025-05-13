import os
import json
import streamlit as st
import pandas as pd

OUTPUT_DIR = "output"

def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        st.warning(f"{filename} not found. Please run the simulation first.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # If data is a list, use the first item
        if isinstance(data, list):
            if len(data) > 0:
                return data[0]
            else:
                return None
        return data
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

# --- Cash Flow Simulation ---
def display_cash_flow():
    data = load_json("1_simulated_cashflow_simulation.json")
    if not data:
        return

    st.subheader("💰 Monthly Cash Flow Simulation")

    # Income
    income = data.get("income", {})
    st.write(f"**Salary:** ₹{income.get('salary', 0):,.2f}")
    st.write(f"**Freelance:** ₹{income.get('freelance', 0):,.2f}")
    st.write(f"**Total Income:** ₹{income.get('total', 0):,.2f}")

    # Expenses
    expenses = data.get("expenses", {})
    st.write(f"**Needs:** ₹{expenses.get('needs', 0):,.2f}")
    st.write(f"**Wants:** ₹{expenses.get('wants', 0):,.2f}")
    st.write(f"**Luxury:** ₹{expenses.get('luxury', 0):,.2f}")
    st.write(f"**Emergency:** ₹{expenses.get('emergency', 0):,.2f}")
    st.write(f"**Total Expenses:** ₹{expenses.get('total', 0):,.2f}")

    # Savings & Debt
    st.write(f"**Savings:** ₹{data.get('savings', 0):,.2f}")
    st.write(f"**Debt Taken:** ₹{data.get('debt_taken', 0):,.2f}")
    st.write(f"**Debt Repaid:** ₹{data.get('debt_repaid', 0):,.2f}")

    # Notes
    st.info(data.get("notes", ""))

# --- Discipline Tracker ---
def display_discipline_tracker():
    data = load_json("1_discipline_report_simulation.json")
    if not data:
        return

    st.subheader("✅ Discipline Tracker")
    rules = data.get("rules_checked", {})
    st.write(f"**Expenses within income:** {'✅' if rules.get('expenses_within_income', False) else '❌'}")
    st.write(f"**Minimum savings met:** {'✅' if rules.get('minimum_savings_met', False) else '❌'}")
    st.write(f"**Unnecessary debt taken:** {'❌' if rules.get('unnecessary_debt_taken', False) else '✅'}")

    st.write("**Violations:**")
    for v in data.get("violations", []):
        st.markdown(f"- {v}")

    st.metric("Discipline Score", data.get("discipline_score", 0))
    st.write("**Recommendations:**")
    for rec in data.get("recommendations", []):
        st.markdown(f"- {rec}")

# --- Goal Tracking ---
def display_goal_tracking():
    data = load_json("1_goal_status_simulation.json")
    if not data:
        return

    st.subheader("🎯 Goal Tracking")
    goals = data.get("goals", [])
    for goal in goals:
        st.write(f"**Goal:** {goal.get('name', '')}")
        st.write(f" - Target: ₹{goal.get('target_amount', 0):,.2f}")
        st.write(f" - Saved so far: ₹{goal.get('saved_so_far', 0):,.2f}")
        st.write(f" - Expected by now: ₹{goal.get('expected_by_now', 0):,.2f}")
        st.write(f" - Status: {goal.get('status', '')}")
        st.write(f" - Priority: {goal.get('priority', '')}")
        st.write(f" - Adjustment: {goal.get('adjustment_suggestion', '')}")
        st.markdown("---")

    summary = data.get("summary", {})
    st.write(f"**On Track Goals:** {summary.get('on_track_goals', 0)}")
    st.write(f"**Behind Goals:** {summary.get('behind_goals', 0)}")
    st.write(f"**Total Saved:** ₹{summary.get('total_saved', 0):,.2f}")
    st.write(f"**Total Required by Now:** ₹{summary.get('total_required_by_now', 0):,.2f}")

# --- Behavior Tracker ---
def display_behavior_tracker():
    data = load_json("1_behavior_tracker_simulation.json")
    if not data:
        return

    st.subheader("🧠 Behavior Tracker")
    traits = data.get("traits", {})
    st.write(f"**Spending Pattern:** {traits.get('spending_pattern', '')}")
    st.write(f"**Goal Adherence:** {traits.get('goal_adherence', '')}")
    st.write(f"**Saving Consistency:** {traits.get('saving_consistency', '')}")
    st.write(f"**Labels:** {', '.join(traits.get('labels', []))}")

# --- Karma Tracker ---
def display_karma_tracker():
    data = load_json("1_karmic_tracker_simulation.json")
    if not data:
        return

    st.subheader("🌱 Karma Tracker")
    traits = data.get("traits", {})
    st.write(f"**Sattvic Traits:** {', '.join(traits.get('sattvic_traits', []))}")
    st.write(f"**Rajasic Traits:** {', '.join(traits.get('rajasic_traits', []))}")
    st.write(f"**Tamasic Traits:** {', '.join(traits.get('tamasic_traits', []))}")
    st.write(f"**Karma Score:** {traits.get('karma_score', 0)}")
    st.write(f"**Trend:** {traits.get('trend', '')}")

# --- Financial Strategy ---
def display_financial_strategy():
    data = load_json("1_financial_strategy_simulation.json")
    if not data:
        return

    st.subheader("📈 Financial Strategy")
    traits = data.get("traits", {})
    st.write("**Recommendations:**")
    for rec in traits.get("recommendations", []):
        st.markdown(f"- {rec}")
    st.write(f"**Reasoning:** {traits.get('reasoning', '')}")

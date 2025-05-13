from functions.economic_context import EconomicEnvironment, simulate_monthly_market
from functions.monthly_simulation import simulate_month
import os
import json
from litellm import RateLimitError, completion
from crew import FinancialCrew
import time
from crewai import Crew
import traceback
import streamlit as st
import requests
from functions.task_functions import *

# *******************************************Functions to sequentially and hierarchically run my crew workflow************************

def kickoff_sequential(self, inputs, sleep_between_calls=15, max_retries=20):
    print("🚀 Starting Sequential Crew Execution...")
    results = []

    month_number = inputs.get('Month', 1)
    user_id = inputs.get('user_id')

    for task in self.tasks:
        print(f"\n🟢 Executing task: {task.name} for agent: {task.agent.role}")

        # 🔎 Check tool status
        if hasattr(task, 'tool') and task.tool:
            try:
                tool_status = task.tool.check_status()
                if tool_status:
                    print(f"🛠️ Tool '{task.tool.name}' is working properly ✅")
                else:
                    print(f"⚠️ Tool '{task.tool.name}' is NOT responding ❌")
            except Exception as e:
                print(f"❗ Error checking tool '{task.tool.name}': {e}")
        else:
            print(f"ℹ️ No tool assigned to task '{task.name}'")

        # ➕ Load context
        context_inputs = inputs.copy()
        prev_result = None
        cashflow_data = None

        if hasattr(task, 'output_file') and task.output_file:
            output_dir = os.path.dirname(task.output_file)
            base_file_name = os.path.basename(task.output_file)
            file_name_without_ext, ext = os.path.splitext(base_file_name)

        # Determine the required context based on task dependencies
        task_context = []
        
        if task.name == "simulate_cashflow_task":
            # Only need previous simulate_cashflow and financial_strategy
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            print("DEBUG: task_context =", task_context)
            task_context.append(build_financial_strategy_context(month_number, user_id))
        elif task.name == "discipline_tracker_task":
            # Needs context for previous simulate_cashflow and discipline_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_discipline_report_context(month_number, user_id))
        
        elif task.name == "track_goals":
            # Needs context for previous simulate_cashflow, discipline_tracker, and goal_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_discipline_report_context(month_number, user_id))
            task_context.append(build_goal_status_context(month_number, user_id))
        
        elif task.name == "behavior_tracker_task":
            # Needs context for previous simulate_cashflow and behavior_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_behavior_tracker_context(month_number, user_id))
        
        elif task.name == "karma_tracker_task":
            # Needs context for previous simulate_cashflow, behavior_tracker, and karma_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_behavior_tracker_context(month_number, user_id))
            task_context.append(build_karmic_tracker_context(month_number, user_id))
        
        elif task.name == "financial_strategy_task":
            # Needs all previous context
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_financial_strategy_context(month_number, user_id))
            task_context.append(build_discipline_report_context(month_number, user_id))
            task_context.append(build_karmic_tracker_context(month_number, user_id))
            task_context.append(build_goal_status_context(month_number, user_id))
            task_context.append(build_behavior_tracker_context(month_number, user_id))

        # Combine all context strings for the task
        context_inputs['context'] = "\n\n".join([c for c in task_context if c])

        # 🚀 Execute task
        retries = 0
        task_result = None
        parsed_result = None
        while retries < max_retries:
            task_result = task.agent.execute_task(task, context_inputs)
            try:
                if isinstance(task_result, str):
                    parsed_result = json.loads(task_result)
                else:
                    parsed_result = task_result
                print(f"✅ Finished task: {task.name}\nResult is valid JSON ✅")
                break
            except json.JSONDecodeError:
                retries += 1
                print(f"❗ Task result is not valid JSON (Attempt {retries}/{max_retries}). Retrying...")
                time.sleep(10)

        if parsed_result is None:
            print(f"❌ Failed to get valid JSON for task '{task.name}' after {max_retries} retries. Saving last result as plain text.")
            parsed_result = task_result

        results.append({
            "task_name": task.name,
            "result": parsed_result
        })

        # 💾 Write output (append mode for every task)
        if hasattr(task, 'output_file') and task.output_file:
            output_dir = os.path.dirname(task.output_file)
            base_file_name = os.path.basename(task.output_file)
            file_name_without_ext, ext = os.path.splitext(base_file_name)
            dynamic_file_name = f"{user_id}_{file_name_without_ext}_simulation{ext}"
            output_path = os.path.join(output_dir, dynamic_file_name)

            os.makedirs(output_dir, exist_ok=True)

            try:
                # Load existing data if exists
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                else:
                    existing_data = []

                # If parsed_result is a list, extend, else append
                if isinstance(parsed_result, list):
                    existing_data.extend(parsed_result)
                else:
                    existing_data.append(parsed_result)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=4, ensure_ascii=False)

                print(f"💾 Appended result to {output_path}")
            except Exception as e:
                print(f"❗ Error appending to file '{output_path}': {e}")

        time.sleep(sleep_between_calls)

    print("🎉 All tasks completed sequentially!")

    return results

# Patch the method into your Crew instance
Crew.kickoff_sequential = kickoff_sequential

# *****************************************************Simulation for my Crew workflow****************************************************

def run_simulation_with_retries(inputs, custom_agents=None, custom_tasks=None, max_attempts=3):
    hashable_inputs = json.dumps(inputs, sort_keys=True)
    customized_agents = custom_agents
    customized_tasks = custom_tasks
    for attempt in range(max_attempts):
        try:
            result = FinancialCrew().flexible_crew(
                input_data=hashable_inputs,
                agent_overrides=customized_agents,
                task_overrides=customized_tasks
            ).kickoff_sequential(inputs=inputs)
            return result
        except RateLimitError:
            st.warning(f"Rate limit hit. Retrying in 10 seconds... (Attempt {attempt + 1}/{max_attempts})")
            time.sleep(10)
        except Exception as e:
            print("Full Traceback:")
            traceback.print_exc()
            st.error(f"An unexpected error occurred: {e}")
            break
    return None

def load_json(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_persona_title(karma_score, behavior_pattern):
    if karma_score > 75 and behavior_pattern == "Consistent Saver":
        return "Wise Sage"
    elif 50 <= karma_score <= 75 and behavior_pattern in ["Consistent Saver", "Occasional Spender"]:
        return "Disciplined Hustler"
    elif karma_score < 50 or behavior_pattern == "Inconsistent Behavior":
        return "Reckless Drifter"
    else:
        return "Balanced Explorer"

def get_month_entries(data, user_name, month):
    # Filter entries for user and month
    return [entry for entry in data if entry.get("user_name") == user_name and entry.get("month") == month]

def generate_monthly_reflection_report(user_name, month):
    # Paths
    output_dir = "output"
    data_dir = "data/"
    os.makedirs(data_dir, exist_ok=True)

    # Load logs
    karma_log = load_json(os.path.join(output_dir, "karmic_tracker_simulation.json")) or []
    behavior_log = load_json(os.path.join(output_dir, "behavior_tracker_simulation.json")) or []
    persona_log = load_json(os.path.join("data", "persona_history.json")) or []

    # Filter for current user and month
    karma_entries = get_month_entries(karma_log, user_name, month)
    behavior_entries = get_month_entries(behavior_log, user_name, month)
    persona_entries = get_month_entries(persona_log, user_name, month)

    # Monthly karma score (average)
    if karma_entries:
        karma_scores = [entry.get("traits", {}).get("karma_score", 0) for entry in karma_entries]
        monthly_karma_score = round(sum(karma_scores) / len(karma_scores), 2)
    else:
        monthly_karma_score = None

    # Persona title + transition
    persona_title = persona_entries[-1]["persona_title"] if persona_entries else "Unassigned"
    transition_note = persona_entries[-1].get("transition_reason", "No transition noted") if persona_entries else "No transition noted"

    # Key behavior observations
    if behavior_entries:
        behavior_traits = behavior_entries[-1].get("traits", {})
        behavior_obs = {
            "spending_pattern": behavior_traits.get("spending_pattern", ""),
            "goal_adherence": behavior_traits.get("goal_adherence", ""),
            "saving_consistency": behavior_traits.get("saving_consistency", ""),
            "labels": behavior_traits.get("labels", [])
        }
    else:
        behavior_obs = {}

    # Summary message
    summary_message = f"You’ve evolved into a {persona_title} due to {transition_note}."

    # Assemble report
    report = {
        "month": month,
        "user_name": user_name,
        "monthly_karma_score": monthly_karma_score,
        "persona_title": persona_title,
        "transition_note": transition_note,
        "key_behavior_observations": behavior_obs,
        "summary_message": summary_message
    }

    # Save report
    save_path = os.path.join(data_dir, f"reflection_month_{month}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ Reflection report saved: {save_path}")
    return report

def assign_persona(user_name, month):
    karma_data = load_json('output/karmic_tracker_simulation.json')
    behavior_data = load_json('output/behavior_tracker_simulatiom.json')
    history_data = load_json('data/persona_history.json')

    # Extract average karma score for the month
    karmic_scores = [entry.get('traits', {}).get('karma_score', 0) for entry in karma_data if entry.get('user_name') == user_name and entry.get('month') == month]
    avg_karmic_score = sum(karmic_scores) / len(karmic_scores) if karmic_scores else 50

    # Extract behavior pattern
    behavior_entry = next((entry for entry in behavior_data if entry.get('user_name') == user_name and entry.get('month') == month), None)
    behavior_pattern = behavior_entry.get('traits', {}).get('spending_pattern', "Inconsistent Behavior") if behavior_entry else "Inconsistent Behavior"

    persona_title = compute_persona_title(avg_karmic_score, behavior_pattern)

    # Check if persona changed
    last_persona = history_data[-1]['persona_title'] if history_data else None
    change_flag = persona_title != last_persona

    record = {
        "user_name": user_name,
        "month": month,
        "persona_title": persona_title,
        "avg_karmic_score": avg_karmic_score,
        "behavior_pattern": behavior_pattern,
        "change_flag": change_flag
    }

    history_data.append(record)
    save_json('data/persona_history.json', history_data)

    print(f"🔮 Persona Assigned for Month {month}: {persona_title} (Change: {change_flag})")
    return record

def simulate_timeline(n_months, simulation_unit, user_inputs):
    previous_result = None

    for month in range(1, n_months + 1):
        eco_env = EconomicEnvironment(unit=simulation_unit)
        eco_env.simulate_step()
        context = eco_env.get_context()
        market_snapshot, market_context_summary = simulate_monthly_market()
        user_inputs["market_context"] = market_context_summary 
        user_name = user_inputs['user_name']
        user_inputs['inflation'] = context['inflation_rate']
        user_inputs['interest_rate'] = context['interest_rate']
        user_inputs['cost_of_living_index'] = context['cost_of_living_index']
        user_inputs['Month'] = month

        print(f"\n🚀 Simulating Month {month}")
        result = run_simulation_with_retries(inputs=user_inputs)
        assign_persona(user_name, month)
        generate_monthly_reflection_report(user_name, month)
    print("\n🎉 All tasks completed sequentially!")
    return True